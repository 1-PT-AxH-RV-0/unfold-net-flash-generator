import click
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

PNG_DIR = "nets_png_reordered"
WIDTH = 1920
HEIGHT = 1080


def list_png_paths(directory=PNG_DIR):
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    png_files = []
    for f in dir_path.rglob("*.png"):
        parts = f.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            idx = int(parts[1])
            png_files.append((idx, f))

    png_files.sort(key=lambda x: x[0])
    return [p for _, p in png_files]


def get_text_size(text, font, scale, thickness):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    return tw, th + baseline


def render_frame(args):
    i, png_path, bg_color, text_color, scale = args

    # 1. 创建背景
    frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)

    # 2. 绘制文字
    parts = png_path.stem.split("_")
    mask = int(parts[2], 16)
    cx, cy = int(parts[4]), int(parts[5])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2

    text = f"Index / Mask: {i:05d} / {mask:016X}"
    tw, th = get_text_size(text, font, font_scale, font_thickness)

    margin = 20
    text_x = margin
    text_y = margin + th

    cv2.putText(
        frame, text, (text_x, text_y),
        font, font_scale, text_color, font_thickness, cv2.LINE_AA
    )

    # 3. 读取 PNG 图片
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return i, frame

    # 4. 根据 scale 缩放图片和对齐坐标
    if scale != 1.0:
        img_h, img_w = img.shape[:2]
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        cx = int(cx * scale)
        cy = int(cy * scale)

    # 5. 分离颜色和 alpha 通道
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:4].astype(float) / 255.0
    else:
        bgr = img
        alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=float)

    img_h, img_w = bgr.shape[:2]

    # 6. 计算图片在帧中的放置区域，使缩放后的 (cx, cy) 对齐到画面中心
    frame_center_x = WIDTH // 2
    frame_center_y = HEIGHT // 2

    paste_x = frame_center_x - cx
    paste_y = frame_center_y - cy

    # 裁剪区域（只绘制落在帧内的部分）
    src_x_start = max(0, -paste_x)
    src_y_start = max(0, -paste_y)
    src_x_end = min(img_w, WIDTH - paste_x)
    src_y_end = min(img_h, HEIGHT - paste_y)

    dst_x_start = max(0, paste_x)
    dst_y_start = max(0, paste_y)
    dst_x_end = min(WIDTH, paste_x + img_w)
    dst_y_end = min(HEIGHT, paste_y + img_h)

    copy_w = src_x_end - src_x_start
    copy_h = src_y_end - src_y_start

    if copy_w > 0 and copy_h > 0:
        roi = frame[dst_y_start:dst_y_start + copy_h, dst_x_start:dst_x_start + copy_w].astype(float)
        fg = bgr[src_y_start:src_y_start + copy_h, src_x_start:src_x_start + copy_w].astype(float)
        fg_alpha = alpha[src_y_start:src_y_start + copy_h, src_x_start:src_x_start + copy_w]

        blended = (fg * fg_alpha + roi * (1 - fg_alpha)).astype(np.uint8)
        frame[dst_y_start:dst_y_start + copy_h, dst_x_start:dst_x_start + copy_w] = blended

    return i, frame


def wait_any(futures):
    from concurrent.futures import wait, FIRST_COMPLETED
    if not futures:
        return set(), set()
    done, pending = wait(futures, return_when=FIRST_COMPLETED)
    return done, pending


def parse_hex_color(ctx, param, value):
    if value is None:
        return None
    value = value.strip()
    if value.startswith("0x") or value.startswith("0X"):
        value = value[2:]
    elif value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        raise click.BadParameter("颜色格式必须为 6 位十六进制，如 0f0f0f 或 0x0f0f0f")
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        return (r, g, b)
    except ValueError:
        raise click.BadParameter("颜色值包含非法十六进制字符")


@click.command()
@click.option("--output", "-o", default="video.mp4", show_default=True, help="输出的视频文件路径")
@click.option("--fps", "-f", default=16, show_default=True, help="视频帧率")
@click.option("--bg-color", "-b", default="0f0f0f", show_default=True, callback=parse_hex_color,
              help="背景颜色 (RRGGBB 十六进制)")
@click.option("--text-color", "-t", default="f0f0f0", show_default=True, callback=parse_hex_color,
              help="文字颜色 (RRGGBB 十六进制)")
@click.option("--max-frames", "-M", default=None, type=int, help="最大生成帧数，默认全部")
@click.option("--workers", default=4, show_default=True, help="并行渲染的线程数")
@click.option("--buffer", default=8, show_default=True, help="预渲染缓冲帧数")
@click.option("--scale", "-s", default=1.0, show_default=True, type=float,
              help="图片缩放倍数")
def main(output, fps, bg_color, text_color, max_frames, workers, buffer, scale):
    png_paths = list_png_paths(PNG_DIR)
    total = min(max_frames, len(png_paths)) if max_frames is not None else len(png_paths)
    print(f"共找到 {len(png_paths)} 个 PNG，本次生成前 {total} 帧")
    print(f"视频分辨率: {WIDTH}×{HEIGHT}, 缩放: {scale}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (WIDTH, HEIGHT))

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output}")

    frames_buffer = {}
    next_write_idx = 0
    written_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending_futures = set()
        task_idx = 0

        with tqdm(total=total, desc="渲染帧") as pbar:
            while task_idx < total or pending_futures:
                # 补充任务到上限
                while task_idx < total and len(pending_futures) < buffer:
                    t = (task_idx, png_paths[task_idx], bg_color, text_color, scale)
                    pending_futures.add(executor.submit(render_frame, t))
                    task_idx += 1

                # 等待至少一个完成
                done, pending_futures = wait_any(pending_futures)
                for future in done:
                    i, frame = future.result()
                    frames_buffer[i] = frame

                    # 按顺序写入
                    while next_write_idx in frames_buffer:
                        writer.write(frames_buffer[next_write_idx])
                        del frames_buffer[next_write_idx]
                        next_write_idx += 1
                        written_count += 1

                    pbar.update(1)

    writer.release()
    print(f"视频已保存: {output}（共 {written_count} 帧）")


if __name__ == "__main__":
    main()