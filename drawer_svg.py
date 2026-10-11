import struct
import math
import numpy as np
from collections import deque, defaultdict
from pathlib import Path
from tqdm import tqdm
from load import load_nets, NetData
import shutil
import click
from multiprocessing import Pool
import io
from fpdf import FPDF


# ============================================================
# 1. 通用 OFF 解析（含顶点坐标）
# ============================================================
def parse_off_full(raw_off: str):
    lines = raw_off.strip().splitlines()
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines) or lines[idx].strip() != "OFF":
        raise ValueError("Invalid OFF: missing OFF header")
    idx += 1
    while idx < len(lines) and (lines[idx].strip() == "" or lines[idx].strip().startswith("#")):
        idx += 1
    V, F, E = map(int, lines[idx].split())
    idx += 1
    vertices = []
    for _ in range(V):
        parts = lines[idx].split()
        if len(parts) >= 3:
            vertices.append(tuple(map(float, parts[:3])))
        idx += 1
    faces = []
    for _ in range(F):
        parts = list(map(int, lines[idx].split()))
        n = parts[0]
        faces.append(parts[1:1 + n])
        idx += 1
    return vertices, faces


# ============================================================
# 2. 局部 2D 参数化（将三维面展平到二维）
# ============================================================
def face_local_2d(vertices, face):
    pts = [np.array(vertices[i], dtype=np.float64) for i in face]
    n = len(pts)
    p0 = pts[0]
    p1 = pts[1] if n > 1 else p0 + np.array([1.0, 0.0, 0.0])
    normal = None
    for i in range(2, n):
        v1 = pts[1] - p0
        v2 = pts[i] - p0
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) > 1e-12:
            break
    if normal is None or np.linalg.norm(normal) < 1e-12:
        raise ValueError(f"面的顶点共线或退化为点：{face}")
    normal = normal / np.linalg.norm(normal)
    x_axis = pts[1] - p0
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    local_coords = []
    for pt in pts:
        vec = pt - p0
        x = np.dot(vec, x_axis)
        y = np.dot(vec, y_axis)
        local_coords.append((x, y))
    return local_coords


# ============================================================
# 3. 2D 变换工具
# ============================================================
def rotate_point(p, angle):
    x, y = p
    c = math.cos(angle)
    s = math.sin(angle)
    return (c * x - s * y, s * x + c * y)


def reflect_point(p, p_start, p_end):
    ax, ay = p_start
    bx, by = p_end
    px, py = p
    dx = bx - ax
    dy = by - ay
    a = dx * dx + dy * dy
    if a < 1e-12:
        return p
    t = ((px - ax) * dx + (py - ay) * dy) / a
    qx = ax + t * dx
    qy = ay + t * dy
    return (2 * qx - px, 2 * qy - py)


def transform_face(local_coords, angle, translation, do_reflect, ref_start, ref_end):
    result = [rotate_point(pt, angle) for pt in local_coords]
    result = [(x + translation[0], y + translation[1]) for x, y in result]
    if do_reflect:
        result = [reflect_point(pt, ref_start, ref_end) for pt in result]
    return result


# ============================================================
# 4. 通用网格展开（已添加缓存参数）
# ============================================================
def unfold_generic(vertices, faces, edges, mask, root_face=0, scale=1.0,
                   local_coords=None, face_vertex_pos=None):
    N = len(faces)
    M = len(edges)

    tree_adj = [[] for _ in range(N)]
    for e_idx in range(M):
        if (mask >> e_idx) & 1:
            u, v = edges[e_idx]
            tree_adj[u].append(v)
            tree_adj[v].append(u)

    if local_coords is None:
        local_coords = [face_local_2d(vertices, f) for f in faces]

    if face_vertex_pos is None:
        face_vertex_pos = []
        for f_idx, face in enumerate(faces):
            pos_dict = defaultdict(list)
            for i, v in enumerate(face):
                pos_dict[v].append(i)
            face_vertex_pos.append(pos_dict)

    global_coords = [None] * N
    global_coords[root_face] = local_coords[root_face][:]

    visited = [False] * N
    visited[root_face] = True
    queue = deque([root_face])

    while queue:
        f = queue.popleft()
        f_global = global_coords[f]
        f_face = faces[f]
        f_pos = face_vertex_pos[f]

        for g in tree_adj[f]:
            if visited[g]:
                continue

            g_face = faces[g]
            g_local = local_coords[g]
            g_pos = face_vertex_pos[g]

            set_f = set(f_face)
            set_g = set(g_face)
            common = set_f & set_g
            if len(common) < 2:
                raise RuntimeError(f"面 {f} 和 {g} 没有共享边？")

            common_list = list(common)[:2]
            v_a, v_b = common_list[0], common_list[1]

            indices_f_a = f_pos[v_a]
            indices_f_b = f_pos[v_b]
            i_f = j_f = None
            n_f = len(f_face)
            for ia in indices_f_a:
                for ib in indices_f_b:
                    if (ia + 1) % n_f == ib or (ib + 1) % n_f == ia:
                        i_f, j_f = ia, ib
                        break
                if i_f is not None:
                    break
            if i_f is None:
                raise RuntimeError(f"共享边不在面 {f} 中连续？")

            indices_g_a = g_pos[v_a]
            indices_g_b = g_pos[v_b]
            i_g = j_g = None
            n_g = len(g_face)
            for ia in indices_g_a:
                for ib in indices_g_b:
                    if (ia + 1) % n_g == ib or (ib + 1) % n_g == ia:
                        i_g, j_g = ia, ib
                        break
                if i_g is not None:
                    break
            if i_g is None:
                raise RuntimeError(f"共享边不在面 {g} 中连续？")

            p_f_start = f_global[i_f]
            p_f_end = f_global[j_f]

            p_g_local_start = g_local[i_g]
            p_g_local_end = g_local[j_g]

            vec_f = (p_f_end[0] - p_f_start[0], p_f_end[1] - p_f_start[1])
            vec_g_local = (p_g_local_end[0] - p_g_local_start[0], p_g_local_end[1] - p_g_local_start[1])

            angle_f = math.atan2(vec_f[1], vec_f[0])
            angle_g = math.atan2(vec_g_local[1], vec_g_local[0])
            angle = angle_f - angle_g

            trans_x = p_f_start[0] - (math.cos(angle) * p_g_local_start[0] - math.sin(angle) * p_g_local_start[1])
            trans_y = p_f_start[1] - (math.sin(angle) * p_g_local_start[0] + math.cos(angle) * p_g_local_start[1])

            g_global_candidate = transform_face(g_local, angle, (trans_x, trans_y), False, (0,0), (0,0))

            f_other_idx = None
            for idx in range(n_f):
                if idx != i_f and idx != j_f:
                    f_other_idx = idx
                    break
            if f_other_idx is None:
                f_other_idx = (i_f + 2) % n_f

            g_other_idx = None
            for idx in range(n_g):
                if idx != i_g and idx != j_g:
                    g_other_idx = idx
                    break
            if g_other_idx is None:
                g_other_idx = (i_g + 2) % n_g

            p_f_other = f_global[f_other_idx]
            p_g_other = g_global_candidate[g_other_idx]

            def side(p, a, b):
                return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

            side_f = side(p_f_other, p_f_start, p_f_end)
            side_g = side(p_g_other, p_f_start, p_f_end)

            need_reflect = (side_f * side_g >= 0)

            g_global_final = transform_face(g_local, angle, (trans_x, trans_y),
                                            need_reflect, p_f_start, p_f_end)

            global_coords[g] = g_global_final
            visited[g] = True
            queue.append(g)

    scaled_coords = []
    for face in global_coords:
        if face:
            scaled_coords.append([(x * scale, y * scale) for x, y in face])
        else:
            scaled_coords.append(None)
    return scaled_coords


# ============================================================
# 5. 绘制函数（SVG 字符串，黑色线框，透明背景，无文字）
# ============================================================
# 注：文字在 PDF 层用 fpdf2 直接绘制，避免 SVG <text> 兼容性问题
def draw_net_svg(face_global_coords, N, stroke_width=0.4):
    xs, ys = [], []
    for coords in face_global_coords:
        if coords:
            for x, y in coords:
                xs.append(x)
                ys.append(y)

    if not xs:
        return "", 0, 0

    margin = 10
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin

    w = int(math.ceil(max_x - min_x))
    h = int(math.ceil(max_y - min_y))

    ox = -min_x
    oy = -min_y

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]

    for coords in face_global_coords:
        if not coords:
            continue

        pts = [(x + ox, y + oy) for x, y in coords]
        points_str = " ".join(f"{px:.2f},{py:.2f}" for px, py in pts)

        svg_parts.append(
            f'<polygon points="{points_str}" fill="none" stroke="#000000" stroke-width="{stroke_width}" stroke-linejoin="round"/>'
        )

    svg_parts.append('</svg>')
    svg_str = "\n".join(svg_parts)

    return svg_str, w, h


# ============================================================
# 6. 多进程并行计算（只计算 SVG 和尺寸，不写文件）
# ============================================================
_worker_data = {}

def _init_worker(vertices, faces, edges, scale, total, stroke_width):
    """每个工作进程初始化时调用，缓存共享数据避免重复序列化与重复计算。"""
    _worker_data['vertices'] = vertices
    _worker_data['faces'] = faces
    _worker_data['edges'] = edges
    _worker_data['scale'] = scale
    _worker_data['total_count'] = total
    _worker_data['N'] = len(faces)
    _worker_data['stroke_width'] = stroke_width

    # 预计算与 mask 无关的数据
    _worker_data['local_coords'] = [face_local_2d(vertices, f) for f in faces]
    _worker_data['face_vertex_pos'] = []
    for face in faces:
        pos_dict = defaultdict(list)
        for i, v in enumerate(face):
            pos_dict[v].append(i)
        _worker_data['face_vertex_pos'].append(pos_dict)


def _compute_single(args):
    """处理单个 mask：计算展开图 → 返回 SVG 和尺寸。
    返回 (idx, mask, svg_str, w, h, success, error_msg)"""
    idx, mask = args
    d = _worker_data

    try:
        global_coords = unfold_generic(d['vertices'], d['faces'], d['edges'], mask,
                                       root_face=0, scale=d['scale'],
                                       local_coords=d['local_coords'],
                                       face_vertex_pos=d['face_vertex_pos'])
        svg_str, img_w, img_h = draw_net_svg(global_coords, d['N'], stroke_width=d['stroke_width'])
        if not svg_str:
            return (idx, mask, None, 0, 0, False, "empty svg")
        return (idx, mask, svg_str, img_w, img_h, True, None)
    except Exception as e:
        return (idx, mask, None, 0, 0, False, str(e))


# ============================================================
# 7. PDF 写入（单线程，带 tqdm）
# ============================================================
def write_pdf(results, out_pdf, page_w, page_h, cols, rows, padding, text_height,
              label_font_size, mask_font_size, grid_line_width):
    """
    单线程将展开图写入 PDF。
    results: list of (idx, mask, svg_str, w, h, success, error)
    """
    valid = [r for r in results if r[5]]
    if not valid:
        print("没有有效的展开图，跳过 PDF 生成。")
        return

    # 统一缩放比：从尺寸最大（长或宽）的展开图计算
    max_dim = max(max(r[3], r[4]) for r in valid)
    cell_w = page_w / cols
    cell_h = page_h / rows
    avail_w = cell_w - 2 * padding
    avail_h = cell_h - text_height - 2 * padding
    avail_size = min(avail_w, avail_h)

    if max_dim <= 0:
        scale = 1.0
    else:
        scale = avail_size / max_dim
    scale = 0.876190

    print(f"页面: {page_w}mm × {page_h}mm, 网格: {cols}×{rows}")
    print(f"单元格: {cell_w:.1f}mm × {cell_h:.1f}mm, 统一缩放比例: {scale:.6f}")
    print(f"最大展开图 max(宽,高): {max_dim:.1f}px, 有效展开图: {len(valid)}")

    pdf = FPDF(unit="mm", format=(page_w, page_h))
    pdf.set_auto_page_break(False)

    nets_per_page = cols * rows
    total_pages = math.ceil(len(valid) / nets_per_page)

    pbar = tqdm(total=len(valid), desc="写入 PDF")

    for page_idx in range(total_pages):
        pdf.add_page()
        page_nets = valid[page_idx * nets_per_page:(page_idx + 1) * nets_per_page]

        for i, (idx, mask, svg_str, w, h, _, _) in enumerate(page_nets):
            row = i // cols
            col = i % cols
            cell_x = col * cell_w
            cell_y = row * cell_h

            # ---- 绘制网格线（cell 边框）----
            if grid_line_width > 0:
                pdf.set_draw_color(180, 180, 180)
                pdf.set_line_width(grid_line_width)
                pdf.rect(cell_x, cell_y, cell_w, cell_h, style="D")

            # ---- 左上角标签：序号 + 16 位 mask ----
            label_x = cell_x + padding
            label_y = cell_y + padding

            pdf.set_xy(label_x, label_y)
            pdf.set_font("Helvetica", "B", label_font_size)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, label_font_size * 0.45, f"#{idx + 576*1000}", new_x="LMARGIN", new_y="NEXT")

            pdf.set_xy(label_x, label_y + label_font_size * 0.5)
            pdf.set_font("Helvetica", "", mask_font_size)
            pdf.cell(0, mask_font_size * 0.4, f"mask={mask:016X}", new_x="LMARGIN", new_y="NEXT")

            # ---- 嵌入 SVG（统一比例缩放，cell 内居中）----
            scaled_w = w * scale
            scaled_h = h * scale

            # 在 cell 内居中（标签区域下方）
            img_x = cell_x + (cell_w - scaled_w) / 2
            img_y = cell_y + text_height + (cell_h - text_height - scaled_h) / 2

            # 边界保护
            if scaled_w > avail_w:
                scaled_w = avail_w
            if scaled_h > avail_h:
                scaled_h = avail_h
            if img_x < cell_x + padding:
                img_x = cell_x + padding
            if img_y < cell_y + text_height:
                img_y = cell_y + text_height

            svg_bytes = io.BytesIO(svg_str.encode('utf-8'))
            pdf.image(svg_bytes, x=img_x, y=img_y, w=scaled_w, h=scaled_h)

            pbar.update(1)

    pbar.close()
    pdf.output(out_pdf)
    print(f"PDF 已保存: {out_pdf}")


# ============================================================
# 8. 主程序（CLI）
# ============================================================
@click.command()
@click.option(
    '--nets_path', '-p',
    default='nets.bin',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=str
    )
)
@click.option(
    '--scale', '-s',
    default='30.0',
    type=float,
    help='展开图内部坐标缩放（默认 30.0）'
)
@click.option(
    '--workers', '-w',
    default=4,
    type=int,
    help='并行工作进程数（默认4，建议设为 CPU 核心数）'
)
@click.option(
    '--cols', '-c',
    default=8,
    type=int,
    help='每行网格数（默认 8）'
)
@click.option(
    '--rows', '-r',
    default=8,
    type=int,
    help='每列网格数（默认 8）'
)
@click.option(
    '--page_w', '-pw',
    default=1600.0,
    type=float,
    help='页面宽度 mm（默认 1600）'
)
@click.option(
    '--page_h', '-ph',
    default=1600.0,
    type=float,
    help='页面高度 mm（默认 1600）'
)
@click.option(
    '--stroke_width', '-sw',
    default=0.4,
    type=float,
    help='SVG 描边粗细，单位 px（默认 0.4）'
)
@click.option(
    '--label_font_size', '-lfs',
    default=16,
    type=float,
    help='左上角序号标签字体大小（默认 16）'
)
@click.option(
    '--mask_font_size', '-mfs',
    default=14,
    type=float,
    help='左上角 mask 标签字体大小（默认 14）'
)
@click.option(
    '--grid_line_width', '-glw',
    default=0.2,
    type=float,
    help='网格线粗细，单位 mm（默认 0.2，设为 0 则不绘制）'
)
@click.option(
    '--out_pdf', '-o',
    default='nets.pdf',
    type=str,
    help='输出 PDF 文件名'
)
def main(nets_path, scale, workers, cols, rows, page_w, page_h, stroke_width,
         label_font_size, mask_font_size, grid_line_width, out_pdf):
    nd = load_nets(nets_path)
    nd.masks = nd.masks[576*1000:576*1200]
    print(f"网格面数 (N): {nd.N}")
    print(f"邻接边数 (M): {nd.M}")
    print(f"总展开图数量: {len(nd.masks)}")

    vertices, faces = parse_off_full(nd.raw_off)
    print(f"顶点数: {len(vertices)}, 面数: {len(faces)}")

    total = len(nd.masks)
    task_gen = ((idx, mask) for idx, mask in enumerate(nd.masks))

    pool_kwargs = {
        'processes': workers,
        'initializer': _init_worker,
        'initargs': (vertices, faces, nd.edges, scale, total, stroke_width)
    }
    try:
        pool = Pool(**pool_kwargs, maxtasksperchild=100)
    except TypeError:
        pool = Pool(**pool_kwargs)

    results = []
    with pool:
        for result in tqdm(
            pool.imap_unordered(_compute_single, task_gen, chunksize=1),
            total=total, desc="计算展开图"
        ):
            results.append(result)
            idx, _, _, _, _, success, error = result
            if not success:
                if error != "empty svg":
                    print(f"展开 mask {idx} 失败: {error}")

    # 按 idx 排序，修复多线程乱序问题
    results.sort(key=lambda r: r[0])

    # 单线程写入 PDF（内部带 tqdm）
    padding = 5.0       # mm，cell 内边距
    text_height = max(label_font_size, mask_font_size) * 0.9 + 2.0  # mm，标签区域高度
    write_pdf(results, out_pdf, page_w, page_h, cols, rows, padding, text_height,
              label_font_size, mask_font_size, grid_line_width)

    print("完成。")


if __name__ == "__main__":
    main()
