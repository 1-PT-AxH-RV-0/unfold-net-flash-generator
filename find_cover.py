import re
import shutil
from pathlib import Path
import cv2
import numpy as np
from collections import namedtuple


NetInfo = namedtuple("NetInfo", ["idx", "mask", "width", "height", "ratio", "path"])


def parse_filename(filename: str):
    """解析文件名 net_XXXXX_xxxxxxxx_WxH_xxx_xxx.png"""
    m = re.match(r'net_(\d{5})_([0-9A-F]{16})_(\d+)x(\d+)_(\d+)_(\d+)\.png$', filename)
    if not m:
        return None
    idx = int(m.group(1))
    mask = m.group(2)
    width = int(m.group(3))
    height = int(m.group(4))
    ratio = width / height
    return NetInfo(idx, mask, width, height, ratio, None)


def find_all_nets(directory: str | Path = "nets_png_reordered"):
    dir_path = Path(directory)
    nets = []

    for f in dir_path.rglob("*.png"):
        info = parse_filename(f.name)
        if info is None:
            continue
        nets.append(info._replace(path=f))

    return nets


def find_closest_ratio(nets, target_ratio: float):
    closest = None
    min_diff = float("inf")

    for net in nets:
        diff = abs(net.ratio - target_ratio)
        if diff < min_diff:
            min_diff = diff
            closest = net

    return closest, min_diff


def add_bg_color(src_path: Path, dst_path: Path, bg_color: tuple = (0x0f, 0x0f, 0x0f)):
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f"无法读取图片: {src_path}")

    h, w = img.shape[:2]

    bg = np.full((h, w, 3), bg_color, dtype=np.uint8)

    bgr = img[:, :, :3]
    alpha = img[:, :, 3:4].astype(float) / 255.0

    bg_float = bg.astype(float)
    fg_float = bgr.astype(float)

    blended = (fg_float * alpha + bg_float * (1 - alpha)).astype(np.uint8)
    cv2.imwrite(str(dst_path), blended)

    print(f"  已保存: {dst_path}")


def main():
    nets = find_all_nets("nets_png_reordered")
    print(f"共找到 {len(nets)} 个 PNG 文件")

    if not nets:
        return

    target_16_9 = 16 / 9
    target_4_3 = 4 / 3

    closest_16_9, diff_16_9 = find_closest_ratio(nets, target_16_9)
    closest_4_3, diff_4_3 = find_closest_ratio(nets, target_4_3)

    print("\n" + "=" * 50)
    print("最接近 16:9 的展开图:")
    print(f"  序号: {closest_16_9.idx:05d}")
    print(f"  Mask: {closest_16_9.mask}")
    print(f"  尺寸: {closest_16_9.width} x {closest_16_9.height}")
    print(f"  比例: {closest_16_9.ratio:.6f} (偏差 {diff_16_9:.6f})")

    print("\n" + "=" * 50)
    print("最接近 4:3 的展开图:")
    print(f"  序号: {closest_4_3.idx:05d}")
    print(f"  Mask: {closest_4_3.mask}")
    print(f"  尺寸: {closest_4_3.width} x {closest_4_3.height}")
    print(f"  比例: {closest_4_3.ratio:.6f} (偏差 {diff_4_3:.6f})")

    root_dir = Path(".")
    bg_color = (0x0f, 0x0f, 0x0f)

    print("\n" + "=" * 50)
    print("复制并添加背景色...")

    dst_16_9 = root_dir / f"net_16_9_{closest_16_9.mask}.png"
    add_bg_color(closest_16_9.path, dst_16_9, bg_color)

    dst_4_3 = root_dir / f"net_4_3_{closest_4_3.mask}.png"
    add_bg_color(closest_4_3.path, dst_4_3, bg_color)

    print(f"\n完成！文件已保存到项目根目录:")
    print(f"  {dst_16_9.name}")
    print(f"  {dst_4_3.name}")


if __name__ == "__main__":
    main()
