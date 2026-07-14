import re
import math
import shutil
from pathlib import Path
from tqdm import tqdm
from load import load_nets, NetData


def mask_to_edge_tuple(mask, edges):
    selected = []
    for i in range(len(edges)):
        if (mask >> i) & 1:
            selected.append(edges[i])
    return tuple(selected)


def parse_filename(filename):
    """
    解析文件名格式：
    net_XXXXX_mask_MMMMMMMM_size_WWxHH_f0_cx_cy.png
    返回 (mask, width, height, cx, cy) 或 None
    """
    m = re.match(
        r'net_(\d{5})_mask_(\d+)_size_(\d+)x(\d+)_f0_(-?\d+)_(-?\d+)\.png$',
        filename
    )
    if not m:
        return None
    _, mask_str, w_str, h_str, cx_str, cy_str = m.groups()
    return int(mask_str), int(w_str), int(h_str), int(cx_str), int(cy_str)


def parse_reordered_filename(filename):
    """
    解析重排后的文件名格式：
    net_XXXXX_MMMMMMMMMMMMMMM_WWxHH_cx_cy.png
    返回 (new_idx, mask, width, height, cx, cy) 或 None
    """
    m = re.match(
        r'net_(\d{5})_([0-9A-Fa-f]{16})_(\d+)x(\d+)_(-?\d+)_(-?\d+)\.png$',
        filename
    )
    if not m:
        return None
    idx_str, mask_hex, w_str, h_str, cx_str, cy_str = m.groups()
    return int(idx_str), int(mask_hex, 16), int(w_str), int(h_str), int(cx_str), int(cy_str)


def boundary_distances(w, h, cx, cy):
    """
    计算点 (cx, cy) 到图片四条边界的距离。
    返回字典：
    {
        'left': cx,
        'right': w - 1 - cx,
        'top': cy,
        'bottom': h - 1 - cy
    }
    """
    return {
        'left': cx,
        'right': w - 1 - cx,
        'top': cy,
        'bottom': h - 1 - cy
    }


def compute_max_distances_from_dir(directory: Path):
    """
    从已存在的目录中读取所有 PNG 文件，计算纵向和横向的最大边界距离。
    返回 (max_vertical, vertical_info, max_horizontal, horizontal_info)
    其中 *_info 为列表，每个元素是 (序号, 方向) 的元组。
    """
    png_files = []
    for f in directory.rglob("*.png"):
        info = parse_reordered_filename(f.name)
        if info is None:
            continue
        new_idx, mask, w, h, cx, cy = info
        png_files.append((new_idx, mask, w, h, cx, cy, f))

    if not png_files:
        print("输出目录中未找到有效的 PNG 文件")
        return 0.0, [], 0.0, []

    max_vertical = 0.0      # 纵向（上/下）
    max_horizontal = 0.0    # 横向（左/右）
    vertical_info = []      # (new_idx, 方向)
    horizontal_info = []    # (new_idx, 方向)

    for new_idx, mask, w, h, cx, cy, f in png_files:
        dists = boundary_distances(w, h, cx, cy)

        # 纵向：上、下
        vert_dist = max(dists['top'], dists['bottom'])
        if vert_dist > max_vertical:
            max_vertical = vert_dist
            vertical_info = [(new_idx, 'top' if dists['top'] > dists['bottom'] else 'bottom')]
        elif vert_dist == max_vertical:
            direction = 'top' if dists['top'] > dists['bottom'] else 'bottom'
            vertical_info.append((new_idx, direction))

        # 横向：左、右
        horiz_dist = max(dists['left'], dists['right'])
        if horiz_dist > max_horizontal:
            max_horizontal = horiz_dist
            horizontal_info = [(new_idx, 'left' if dists['left'] > dists['right'] else 'right')]
        elif horiz_dist == max_horizontal:
            direction = 'left' if dists['left'] > dists['right'] else 'right'
            horizontal_info.append((new_idx, direction))

    return max_vertical, vertical_info, max_horizontal, horizontal_info


def main():
    # 加载真实网格数据（包含边列表）
    nd = load_nets("nets.bin")
    edges = nd.edges
    print(f"读取到 {len(edges)} 条边，面数 {nd.N}")

    input_dir = Path("nets_png")
    output_dir = Path("nets_png_reordered")

    # 如果输出目录已存在，直接计算并输出最大距离
    if output_dir.exists():
        print(f"输出目录 {output_dir} 已存在，跳过复制，直接计算最大距离...")
        max_vert, vert_info, max_horiz, horiz_info = compute_max_distances_from_dir(output_dir)

        print(f"\n纵向（上/下）最远距离: {max_vert:.2f} 像素")
        print(f"对应图片序号及方向:")
        for idx, direction in vert_info:
            print(f"  序号 {idx}, 方向: {direction}")

        print(f"\n横向（左/右）最远距离: {max_horiz:.2f} 像素")
        print(f"对应图片序号及方向:")
        for idx, direction in horiz_info:
            print(f"  序号 {idx}, 方向: {direction}")
        return

    output_dir.mkdir(exist_ok=True)

    # 收集所有 PNG 文件
    png_files = []
    for f in input_dir.rglob("*.png"):
        info = parse_filename(f.name)
        if info is None:
            print(f"  跳过无法解析的文件: {f}")
            continue
        mask, w, h, cx, cy = info
        edge_tuple = mask_to_edge_tuple(mask, edges)
        png_files.append((edge_tuple, mask, w, h, cx, cy, f))

    if not png_files:
        print("未找到有效的 PNG 文件")
        return

    # 按边元组升序排序（默认元组比较：逐元素字典序）
    png_files.sort(key=lambda x: x[0])

    # 重命名并复制，同时统计0号面中心到边界的最大距离
    max_vertical = 0.0
    max_horizontal = 0.0
    vertical_info = []
    horizontal_info = []

    for new_idx, (edge_tuple, mask, w, h, cx, cy, src_path) in tqdm(
        enumerate(png_files), total=len(png_files)
    ):
        # 计算边界距离
        dists = boundary_distances(w, h, cx, cy)

        # 纵向：上、下
        vert_dist = max(dists['top'], dists['bottom'])
        if vert_dist > max_vertical:
            max_vertical = vert_dist
            vertical_info = [(new_idx, 'top' if dists['top'] > dists['bottom'] else 'bottom')]
        elif vert_dist == max_vertical:
            direction = 'top' if dists['top'] > dists['bottom'] else 'bottom'
            vertical_info.append((new_idx, direction))

        # 横向：左、右
        horiz_dist = max(dists['left'], dists['right'])
        if horiz_dist > max_horizontal:
            max_horizontal = horiz_dist
            horizontal_info = [(new_idx, 'left' if dists['left'] > dists['right'] else 'right')]
        elif horiz_dist == max_horizontal:
            direction = 'left' if dists['left'] > dists['right'] else 'right'
            horizontal_info.append((new_idx, direction))

        # mask 转为 16 位大写十六进制（适配最多 64 条边）
        mask_hex = f"{mask:016X}"
        new_filename = f"net_{new_idx:05d}_{mask_hex}_{w}x{h}_{cx}_{cy}.png"

        # 子目录：每 1000 个一组
        start = (new_idx // 1000) * 1000
        end = start + 999
        subdir = output_dir / f"{start + 1}~{end + 1}"
        subdir.mkdir(exist_ok=True)

        dst_path = subdir / new_filename
        shutil.copy2(src_path, dst_path)

    print(f"\n全部重排完成，输出目录: {output_dir}")
    print(f"\n纵向（上/下）最远距离: {max_vertical:.2f} 像素")
    print(f"对应图片序号及方向:")
    for idx, direction in vertical_info:
        print(f"  序号 {idx}, 方向: {direction}")

    print(f"\n横向（左/右）最远距离: {max_horizontal:.2f} 像素")
    print(f"对应图片序号及方向:")
    for idx, direction in horizontal_info:
        print(f"  序号 {idx}, 方向: {direction}")


if __name__ == "__main__":
    main()