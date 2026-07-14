import struct
from pathlib import Path


class NetData:
    def __init__(self, raw_off, edges, masks):
        self.raw_off = raw_off
        self.edges = edges
        self.masks = masks
        self.N = len(self.parse_off_faces())
        self.M = len(edges)

    def parse_off_faces(self):
        lines = self.raw_off.strip().splitlines()
        idx = 0
        # 跳过空行，查找OFF头部
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        if idx >= len(lines) or lines[idx].strip() != "OFF":
            raise ValueError("无效的OFF文件：缺少OFF头部")
        idx += 1
        # 跳过注释行和空行
        while idx < len(lines) and (lines[idx].strip() == "" or lines[idx].strip().startswith("#")):
            idx += 1
        V, F, E = map(int, lines[idx].split())
        idx += 1
        # 跳过顶点数据
        for _ in range(V):
            idx += 1
        # 读取面数据
        faces = []
        for _ in range(F):
            parts = list(map(float, lines[idx].split()))
            n = int(parts[0])
            faces.append(parts[1:1 + n])
            idx += 1
        return faces

    def mask_to_edges(self, mask):
        return [self.edges[i] for i in range(self.M) if (mask >> i) & 1]

    def mask_to_edge_set(self, mask):
        return frozenset(self.edges[i] for i in range(self.M) if (mask >> i) & 1)

    def mask_to_edge_indices(self, mask):
        return [i for i in range(self.M) if (mask >> i) & 1]


def load_nets(bin_path="nets.bin"):
    data = Path(bin_path).read_bytes()
    offset = 0

    # 1. OFF字符串长度
    off_len = struct.unpack_from("<Q", data, offset)[0]
    offset += 8

    # 2. OFF字符串
    raw_off = data[offset:offset + off_len].decode("utf-8")
    offset += off_len

    # 3. 边数量
    M = struct.unpack_from("<i", data, offset)[0]
    offset += 4

    # 4. 边数组
    edges = []
    for _ in range(M):
        a, b = struct.unpack_from("<ii", data, offset)
        edges.append((a, b))
        offset += 8

    # 5. 掩码数量
    count = struct.unpack_from("<Q", data, offset)[0]
    offset += 8

    # 6. 掩码列表
    masks = []
    for _ in range(count):
        masks.append(struct.unpack_from("<Q", data, offset)[0])
        offset += 8

    return NetData(raw_off, edges, masks)


if __name__ == "__main__":
    nd = load_nets("nets.bin")
    print(f"OFF头部: {nd.raw_off[:20]}...")
    print(f"面数量: {nd.N}")
    print(f"邻接边数量: {nd.M}")
    print(f"前5条边: {nd.edges[:5]}")
    print(f"总网络数: {len(nd.masks)}")
    print(f"第一个掩码: {nd.masks[0]:016X}")

    mask = nd.masks[0]
    print(f"\n掩码 {mask:016X}:")
    print(f"  选中的边: {nd.mask_to_edges(mask)}")
    print(f"  边索引: {nd.mask_to_edge_indices(mask)}")
    print(f"  边数量: {len(nd.mask_to_edge_indices(mask))}")
    print(f"  期望值: {nd.N - 1} (在{nd.N}个面上的生成树)")
