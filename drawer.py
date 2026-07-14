import struct
import math
import numpy as np
import cv2
import shutil
from collections import deque, defaultdict
from pathlib import Path
from tqdm import tqdm
from load import load_nets, NetData
import click

# ============================================================
# 1. 通用 OFF 解析（含顶点坐标）
# ============================================================
def parse_off_full(raw_off):
    lines = raw_off.strip().splitlines()
    idx = 0
    # 跳过空行和 OFF 头
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines) or lines[idx].strip() != "OFF":
        raise ValueError("Invalid OFF: missing OFF header")
    idx += 1
    # 跳过注释和空行
    while idx < len(lines) and (lines[idx].strip() == "" or lines[idx].strip().startswith("#")):
        idx += 1
    V, F, E = map(int, lines[idx].split())
    idx += 1
    # 读顶点
    vertices = []
    for _ in range(V):
        parts = lines[idx].split()
        if len(parts) >= 3:
            vertices.append(tuple(map(float, parts[:3])))
        idx += 1
    # 读面
    faces = []
    for _ in range(F):
        parts = list(map(int, lines[idx].split()))
        n = parts[0]
        faces.append(parts[1:1 + n])
        idx += 1
    return vertices, faces


# ============================================================
# 2. 文字工具
# ============================================================
def put_text_center(img, text, center, font, font_scale, color, thickness):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = center[0] - text_w // 2
    y = center[1] + text_h // 2
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


# ============================================================
# 3. 局部 2D 参数化（将三维面展平到二维）
# ============================================================
def face_local_2d(vertices, face):
    """
    为面构建局部 2D 坐标系，返回该面所有顶点在局部平面上的坐标列表。
    坐标系原点：第一个顶点；x 轴：第一边方向；z：法向；y = z × x。
    """
    pts = [np.array(vertices[i], dtype=np.float64) for i in face]
    n = len(pts)
    p0 = pts[0]
    # 寻找不共线的第二、第三点
    p1 = pts[1] if n > 1 else p0 + np.array([1.0, 0.0, 0.0])
    # 计算法向：使用前三个点，若共线则尝试后续点
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
# 4. 2D 变换工具
# ============================================================
def rotate_point(p, angle):
    """绕原点旋转点"""
    x, y = p
    c = math.cos(angle)
    s = math.sin(angle)
    return (c * x - s * y, s * x + c * y)


def reflect_point(p, p_start, p_end):
    """点 p 关于直线 (p_start, p_end) 的镜像"""
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
    """
    对局部坐标应用变换：旋转、平移，可选反射。
    """
    result = [rotate_point(pt, angle) for pt in local_coords]
    result = [(x + translation[0], y + translation[1]) for x, y in result]
    if do_reflect:
        result = [reflect_point(pt, ref_start, ref_end) for pt in result]
    return result


# ============================================================
# 5. 通用网格展开
# ============================================================
def unfold_generic(vertices, faces, edges, mask, root_face=0, scale=1.0):
    """
    根据生成树 mask（边索引的位掩码）展开所有面。
    返回: face_global_coords: list of list of (x,y)，按面索引存储，坐标已乘以 scale。
    """
    N = len(faces)          # 面数
    M = len(edges)          # 边总数

    # 1. 构建生成树的邻接表（面-面）
    tree_adj = [[] for _ in range(N)]
    for e_idx in range(M):
        if (mask >> e_idx) & 1:
            u, v = edges[e_idx]    # 两个相邻的面索引
            tree_adj[u].append(v)
            tree_adj[v].append(u)

    # 2. 为每个面预先计算局部 2D 坐标
    local_coords = [face_local_2d(vertices, f) for f in faces]

    # 3. 建立顶点在每个面中的位置索引（便于寻找共享边）
    # 格式：face_vertex_pos[face_idx][vertex_idx] = 在该面顶点列表中的局部索引列表
    face_vertex_pos = []
    for f_idx, face in enumerate(faces):
        pos_dict = defaultdict(list)
        for i, v in enumerate(face):
            pos_dict[v].append(i)
        face_vertex_pos.append(pos_dict)

    # 4. 初始化全局坐标（每个面一个坐标列表）
    global_coords = [None] * N
    # 放置根面：直接使用局部坐标（稍后会统一缩放）
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

            # 寻找共享边 (在顶点索引意义上的两个顶点)
            shared_vertices = None
            # 只需检查两个面的顶点交集即可
            set_f = set(f_face)
            set_g = set(g_face)
            common = set_f & set_g
            # 通常共享边由两个顶点组成，若多边形共享一条完整边
            if len(common) < 2:
                raise RuntimeError(f"面 {f} 和 {g} 没有共享边。")

            # 选择共享边上的两个顶点（取第一个两个）
            common_list = list(common)[:2]
            v_a, v_b = common_list[0], common_list[1]

            # 在 f 中找到这两个点的局部索引，并确保它们构成一条边（相邻）
            indices_f_a = f_pos[v_a]
            indices_f_b = f_pos[v_b]
            # 找到相邻的索引对
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

            # 在 g 中同样找到局部索引
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

            # 父面中共享边的两个全局坐标
            p_f_start = f_global[i_f]
            p_f_end = f_global[j_f]

            # 子面中共享边的两个局部坐标
            p_g_local_start = g_local[i_g]
            p_g_local_end = g_local[j_g]

            # 计算旋转角度，使局部边对齐全局边
            vec_f = (p_f_end[0] - p_f_start[0], p_f_end[1] - p_f_start[1])
            vec_g_local = (p_g_local_end[0] - p_g_local_start[0], p_g_local_end[1] - p_g_local_start[1])

            angle_f = math.atan2(vec_f[1], vec_f[0])
            angle_g = math.atan2(vec_g_local[1], vec_g_local[0])
            angle = angle_f - angle_g

            # 旋转后平移，使局部起点对齐
            trans_x = p_f_start[0] - (math.cos(angle) * p_g_local_start[0] - math.sin(angle) * p_g_local_start[1])
            trans_y = p_f_start[1] - (math.sin(angle) * p_g_local_start[0] + math.cos(angle) * p_g_local_start[1])

            # 先不加反射，计算第三点方向
            g_global_candidate = transform_face(g_local, angle, (trans_x, trans_y), False, (0,0), (0,0))

            # 检查方向：父面第三点与子面第三点应在共享边两侧
            # 取父面中不在边上的一个顶点
            f_other_idx = None
            for idx in range(n_f):
                if idx != i_f and idx != j_f:
                    f_other_idx = idx
                    break
            if f_other_idx is None:
                # 如果面只有两个顶点？不可能
                f_other_idx = (i_f + 2) % n_f

            # 子面同样
            g_other_idx = None
            for idx in range(n_g):
                if idx != i_g and idx != j_g:
                    g_other_idx = idx
                    break
            if g_other_idx is None:
                g_other_idx = (i_g + 2) % n_g

            p_f_other = f_global[f_other_idx]
            p_g_other = g_global_candidate[g_other_idx]

            # 计算相对于有向边 (p_f_start -> p_f_end) 的叉积符号
            def side(p, a, b):
                return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

            side_f = side(p_f_other, p_f_start, p_f_end)
            side_g = side(p_g_other, p_f_start, p_f_end)

            # 如果符号相同（或有一个为零），需要反射
            need_reflect = (side_f * side_g >= 0)

            g_global_final = transform_face(g_local, angle, (trans_x, trans_y),
                                            need_reflect, p_f_start, p_f_end)

            global_coords[g] = g_global_final
            visited[g] = True
            queue.append(g)

    # 应用缩放
    scaled_coords = []
    for face in global_coords:
        if face:
            scaled_coords.append([(x * scale, y * scale) for x, y in face])
        else:
            scaled_coords.append(None)
    return scaled_coords


# ============================================================
# 6. 绘制函数（白色线框 + 白色文字，透明背景）
# ============================================================
def draw_net_png(face_global_coords, N):
    """
    绘制所有面的展开图，返回 (canvas, w, h, face0_center)。
    """
    xs, ys = [], []
    for coords in face_global_coords:
        if coords:
            for x, y in coords:
                xs.append(x)
                ys.append(y)

    if not xs:
        return None, 0, 0, (-1, -1)

    margin = 60
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin

    w = int(math.ceil(max_x - min_x))
    h = int(math.ceil(max_y - min_y))

    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    ox = -min_x
    oy = -min_y

    white_color = (240, 240, 240, 255)

    face0_center = (-1, -1)

    for i, coords in enumerate(face_global_coords):
        if not coords:
            continue

        pts = np.array([[[int(x + ox), int(y + oy)] for x, y in coords]], dtype=np.int32)

        cv2.polylines(canvas, pts, True, white_color, 2, cv2.LINE_AA)

        cx = int(sum(p[0] for p in pts[0]) / len(pts[0]))
        cy = int(sum(p[1] for p in pts[0]) / len(pts[0]))
        put_text_center(canvas, str(i), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, white_color, 2)

        if i == 0:
            face0_center = (cx, cy)

    return canvas, w, h, face0_center


# ============================================================
# 7. 主程序
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
def main(nets_path):
    # 读取 nets.bin
    nd = load_nets(nets_path)
    print(f"网格面数 (N): {nd.N}")
    print(f"邻接边数 (M): {nd.M}")
    print(f"总展开图数量: {len(nd.masks)}")

    # 解析 OFF 得到顶点和面
    vertices, faces = parse_off_full(nd.raw_off)
    print(f"顶点数: {len(vertices)}, 面数: {len(faces)}")

    # ---- 控制展开图整体大小的缩放因子 ----
    scale = 63

    out_dir = Path("nets_png")
    out_dir.mkdir(exist_ok=True)
    for item in out_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)

    for idx, mask in enumerate(tqdm(nd.masks, desc="绘制展开图")):
        try:
            global_coords = unfold_generic(vertices, faces, nd.edges, mask,
                                           root_face=0, scale=scale)
        except Exception as e:
            print(f"展开 mask {idx} 失败: {e}")
            continue

        canvas, img_w, img_h, (f0_cx, f0_cy) = draw_net_png(global_coords, nd.N)
        if canvas is None:
            continue

        filename = f"net_{idx:05d}_mask_{mask}_size_{img_w}x{img_h}_f0_{f0_cx}_{f0_cy}.png"
        subdir = out_dir / f"{(idx // 1000) * 1000 + 1}~{min((idx // 1000 + 1) * 1000, len(nd.masks))}"
        subdir.mkdir(exist_ok=True)

        cv2.imwrite(str(subdir / filename), canvas)

    print("完成。")


if __name__ == "__main__":
    main()