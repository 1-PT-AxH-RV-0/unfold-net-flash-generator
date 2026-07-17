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
# 5. 绘制函数（SVG 字符串，白色线框 + 白色文字，透明背景）
# ============================================================
def draw_net_svg(face_global_coords, N):
    xs, ys = [], []
    for coords in face_global_coords:
        if coords:
            for x, y in coords:
                xs.append(x)
                ys.append(y)

    if not xs:
        return "", 0, 0, (-1, -1)

    margin = 60
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin

    w = int(math.ceil(max_x - min_x))
    h = int(math.ceil(max_y - min_y))

    ox = -min_x
    oy = -min_y

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]

    face0_center = (-1, -1)

    for i, coords in enumerate(face_global_coords):
        if not coords:
            continue

        pts = [(x + ox, y + oy) for x, y in coords]
        points_str = " ".join(f"{px:.2f},{py:.2f}" for px, py in pts)

        svg_parts.append(
            f'<polygon points="{points_str}" fill="none" stroke="#F0F0F0" stroke-width="2" stroke-linejoin="round"/>'
        )

        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        # 恢复面号文字标注
        svg_parts.append(
            f'<text x="{cx:.2f}" y="{cy:.2f}" fill="#F0F0F0" font-family="sans-serif" font-size="24" '
            f'text-anchor="middle" dominant-baseline="central">{i}</text>'
        )

        if i == 0:
            face0_center = (int(round(cx)), int(round(cy)))

    svg_parts.append('</svg>')
    svg_str = "\n".join(svg_parts)

    return svg_str, w, h, face0_center


# ============================================================
# 6. 多进程并行处理（内存友好）
# ============================================================
_worker_data = {}

def _init_worker(vertices, faces, edges, scale, out_dir, total):
    """每个工作进程初始化时调用，缓存共享数据避免重复序列化与重复计算。"""
    _worker_data['vertices'] = vertices
    _worker_data['faces'] = faces
    _worker_data['edges'] = edges
    _worker_data['scale'] = scale
    _worker_data['out_dir'] = out_dir
    _worker_data['total_count'] = total
    _worker_data['N'] = len(faces)

    # 预计算与 mask 无关的数据
    _worker_data['local_coords'] = [face_local_2d(vertices, f) for f in faces]
    _worker_data['face_vertex_pos'] = []
    for face in faces:
        pos_dict = defaultdict(list)
        for i, v in enumerate(face):
            pos_dict[v].append(i)
        _worker_data['face_vertex_pos'].append(pos_dict)


def _process_single_mask(args):
    """处理单个 mask：计算 → 绘制 → 保存。返回 (idx, success, error_msg)"""
    idx, mask = args
    d = _worker_data
    subdir = d['out_dir'] / f"{(idx // 1000) * 1000 + 1}~{min((idx // 1000 + 1) * 1000, d['total_count'])}"

    try:
        global_coords = unfold_generic(d['vertices'], d['faces'], d['edges'], mask,
                                       root_face=0, scale=d['scale'],
                                       local_coords=d['local_coords'],
                                       face_vertex_pos=d['face_vertex_pos'])
        svg_str, img_w, img_h, (f0_cx, f0_cy) = draw_net_svg(global_coords, d['N'])
        if not svg_str:
            return (idx, False, "empty svg")

        subdir.mkdir(exist_ok=True, parents=True)
        filename = f"net_{idx:05d}_mask_{mask}_size_{img_w}x{img_h}_f0_{f0_cx}_{f0_cy}.svg"
        filepath = subdir / filename
        filepath.write_text(svg_str, encoding='utf-8')
        return (idx, True, None)
    except Exception as e:
        return (idx, False, str(e))


# ============================================================
# 7. 主程序（CLI）
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
    type=float
)
@click.option(
    '--workers', '-w',
    default=4,
    type=int,
    help='并行工作进程数（默认4，建议设为 CPU 核心数）'
)
def main(nets_path, scale, workers):
    nd = load_nets(nets_path)
    print(f"网格面数 (N): {nd.N}")
    print(f"邻接边数 (M): {nd.M}")
    print(f"总展开图数量: {len(nd.masks)}")

    vertices, faces = parse_off_full(nd.raw_off)
    print(f"顶点数: {len(vertices)}, 面数: {len(faces)}")

    out_dir = Path("nets_svg")
    out_dir.mkdir(exist_ok=True)
    for item in out_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)

    total = len(nd.masks)
    task_gen = ((idx, mask) for idx, mask in enumerate(nd.masks))

    pool_kwargs = {
        'processes': workers,
        'initializer': _init_worker,
        'initargs': (vertices, faces, nd.edges, scale, out_dir, total)
    }
    try:
        pool = Pool(**pool_kwargs, maxtasksperchild=100)
    except TypeError:
        pool = Pool(**pool_kwargs)

    with pool:
        for idx, success, error in tqdm(
            pool.imap_unordered(_process_single_mask, task_gen, chunksize=1),
            total=total, desc="绘制展开图"
        ):
            if not success:
                if error != "empty svg":
                    print(f"展开 mask {idx} 失败: {error}")

    print("完成。")


if __name__ == "__main__":
    main()
