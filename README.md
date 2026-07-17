# UnfoldNetFlashGenerator

将多面体展开图（Net）生成快闪视频的工具链。
输入一个多面体的 OFF 格式文件，自动枚举所有非同构的展开图（生成树），绘制为 PNG 序列，并按字典序排序后合成为视频，同时生成封面图。

## 项目背景

多面体展开图（Net）是将多面体表面沿棱剪开后摊平在平面上的图形。
本工具用于：
- 枚举多面体的所有生成树（即所有可能的展开方式）
- 通过自同构群去重，得到非等价展开图
- 将每个展开图渲染为透明 PNG
- 按边集字典序排序（便于视频连续观看）
- 生成高清视频，并自动选出比例最接近 16:9 和 4:3 的展开图作为封面

适用于正十二面体（43380 个展开图）、正二十面体（43380 个）、菱形十二面体（6912 个）等顶点/面数较少的多面体（展开图总数建议控制在 10 万以内）。

---

## 依赖与安装

### 编译环境
- C++17 兼容编译器（GCC/Clang/MSVC）
- Python 3.8+
- 依赖库：
  ```bash
  pip install click opencv-python numpy tqdm
  ```

### 安装
```bash
# 克隆仓库（假设）
git clone <repo-url>
cd UnfoldNetFlashGenerator

# 编译 unfold.cpp
g++ -std=c++17 -O3 -o unfold unfold.cpp
```

---

## 快速开始

1. **获取多面体 OFF 文件**
   从 [polytope.miraheze.org](https://polytope.miraheze.org) 或其他来源下载 `.off` 文件（例如 `Icosahedron.off`）。

2. **枚举展开图**
   ```bash
   ./unfold icosahedron.off nets.bin
   ```
   生成 `nets.bin` 二进制文件，包含 OFF 原文、邻接边信息及所有去重后的掩码。

3. **绘制展开图 PNG**
   ```bash
   python drawer.py -p nets.bin -s 82.7
   ```
   （`-s` 控制绘制大小，默认 50.0，可根据需要调整）
   在 `nets_png/` 下生成大量 PNG 文件（按子目录分组，每 1000 个一组）。

4. **按字典序重排**
   ```bash
   python reorder_pngs.py
   ```
   读取 `nets_png/`，按边元组字典序排序，复制到 `nets_png_reordered/`，并输出最远的边界距离（用于下一步缩放）。
   注意：若 `nets_png_reordered/` 已存在，并且你是在生成新的图（而非没看见上次的输出，需要重新运行），请删掉它。

5. **生成视频**
   根据上一步输出的“纵向最远距离”和“横向最远距离”，选择合适的缩放比例（`--scale`），使展开图在 1920×1080 画面中尽量大且不溢出。
   示例：
   ```bash
   python videor.py -s 2.0 -o output.mp4
   ```
   也可使用默认缩放 1.0。

6. **生成封面图**
   ```bash
   python find_cover.py
   ```
   在项目根目录生成 `net_16_9_<mask>.png` 和 `net_4_3_<mask>.png`（带背景色），可作为视频封面。

---

## 详细用法

### 1. `unfold.cpp` – 枚举去重展开图

编译后得到一个可执行文件（如 `unfold`）。
**输入**：OFF 格式的多面体文件。
**输出**：二进制文件 `nets.bin`，包含：
- OFF 文件的原始文本（用于后续解析）
- 邻接边列表（面与面之间的相邻棱）
- 所有去重后的展开图掩码（每个掩码表示一棵生成树的边索引集合）

**用法**：
```bash
./unfold <input.off> [output.bin]
```
- `input.off`：必需，OFF 文件路径。
- `output.bin`：可选，默认 `nets.bin`。

**注意**：若邻接边数 `M > 60`，程序会报错，因为掩码用 64 位整数存储，无法超过 60 条边。
建议用于边数较少的多面体（通常 < 50）。

---

### 2. `drawer.py` – 渲染展开图为 PNG

将 `nets.bin` 中每个掩码对应的展开图绘制成透明 PNG（白色线框，面编号标注）。
**依赖**：`load.py` 提供数据加载，需要 `bin` 文件存在。

**用法**：
```bash
python drawer.py [--nets_path PATH] [--scale SCALE]
```
- `--nets_path` / `-p`：指定 `nets.bin` 路径，默认 `nets.bin`。
- `--scale` / `-s`：控制展开图整体放大倍数（浮点数），默认 `50.0`。
  该值乘以原始单位坐标，决定最终像素尺寸。数值越大，展开图在画布上越大。
- `--workers` / `-w`：并行工作进程数，默认 `4`。

**输出**：`nets_png/` 目录，内部按子目录分组（例如 `1~1000/`、`1001~2000/`……），每个 PNG 文件名格式：
```
net_XXXXX_mask_<mask>_size_WxH_f0_cx_cy.png
```
其中 `cx, cy` 是第 0 号面中心在画布上的坐标（供后续对齐使用）。

**调参建议**：绘制完成后，可运行 `reorder_pngs.py` 观察边界距离，若距离严重过小或过大，请调整 `--scale` 重新绘制。不严重的话可以在生成视频那一步中直接调参调整以节省时间。

---

### 3. `reorder_pngs.py` – 重排与边界距离统计

将 `nets_png/` 下的 PNG 按边元组（即选中的邻接边集合）的字典序排序，并复制到 `nets_png_reordered/`，同时计算 0 号面中心到画布四边的最大距离（纵向和横向），便于决定视频缩放。

**用法**：
```bash
python reorder_pngs.py
```
无参数。

**行为**：
- 若 `nets_png_reordered/` 已存在，则**跳过复制**，直接计算并输出最大距离。
- 否则创建目录，读取 `nets_png/` 下所有 PNG，解析文件名获取边元组，排序后复制到新目录（同样按每 1000 个分文件夹）。

**输出**：
- 在 `nets_png_reordered/` 中生成重命名的文件，格式：
```
net_<新序号>_<掩码16位16进制>_WxH_cx_cy.png
```
- 控制台打印出纵向（上下）和横向（左右）的最大距离（像素），以及对应的图片序号和方向。
  例如：
```
纵向最远距离: 430.00 像素
对应图片序号及方向:
序号 12345, 方向: top
```

这个距离是后续 `videor.py` 中 `--scale` 取值的重要参考。

---

### 4. `videor.py` – 生成视频

将 `nets_png_reordered/` 中的 PNG 序列按顺序合成为 MP4 视频，支持多线程渲染、自定义背景色、文字颜色、缩放等。

**用法**：
```bash
python videor.py [OPTIONS]
```

**选项**：
| 参数 | 短名 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--output` | `-o` | 字符串 | `video.mp4` | 输出视频路径 |
| `--fps` | `-f` | 整数 | 16 | 视频帧率 |
| `--bg-color` | `-b` | 字符串 | `0f0f0f` | 背景色（RRGGBB 十六进制） |
| `--text-color` | `-t` | 字符串 | `f0f0f0` | 文字颜色（RRGGBB 十六进制） |
| `--max-frames` | `-M` | 整数 | `None` | 最大帧数（默认全部） |
| `--workers` | (无) | 整数 | 4 | 并行渲染线程数 |
| `--buffer` | (无) | 整数 | 8 | 预渲染缓冲帧数 |
| `--scale` | `-s` | 浮点 | 1.0 | 图片缩放倍数 |

**背景与文字颜色**：
颜色值支持 `0x0f0f0f`、`#0f0f0f` 或直接 `0f0f0f`（不区分大小写）。

**缩放确定**：
通常利用 `reorder_pngs.py` 给出的最大距离，计算能容纳的最大缩放：

若横向最大距离为 Hmax，纵向最大距离为 Vmax，则缩放上限约为 `min(1920/(2*Hmax), 1080/(2*Vmax))`，可适当留边距。

**输出**：视频文件（默认 `video.mp4`）。

---

### 5. `find_cover.py` – 生成封面图

在重排后的目录中查找比例最接近 16:9 和 4:3 的展开图，并合成带背景色的 PNG 文件（背景色固定为 `#0f0f0f`）。

**用法**：
```bash
python find_cover.py
```
无参数。

**输出**：项目根目录下生成：
- `net_16_9_<mask>.png`
- `net_4_3_<mask>.png`

这两个文件可作为视频封面或缩略图。

---

## 工作流程总结

1. 获取 OFF → `unfold` → `nets.bin`
2. `drawer.py --scale <绘制大小>` → `nets_png/`（原始 PNG）
3. `reorder_pngs.py` → `nets_png_reordered/`（排序后，并输出最大距离）
4. 根据最大距离计算合适的 `--scale` 值（用于 `videor.py`）
5. `videor.py -s <视频缩放> -o my_video.mp4`
6. `find_cover.py`（生成封面）

---

## 常见问题

**Q：`unfold.cpp` 运行过慢？**
A：若 `M` 较大，枚举可能会非常慢，请耐心等待。也可先查询展开图数量，超过一百万种就非常不建议了。

**Q：视频中展开图被裁剪？**
A：调整 `videor.py` 的 `-s` 为更小值。可参考 `reorder_pngs.py` 输出的最远距离来估算。

**Q：OFF 文件从何获取？**
A：推荐 [polytope.miraheze.org](https://polytope.miraheze.org)，搜索多面体名称，下载其 OFF 文件。
