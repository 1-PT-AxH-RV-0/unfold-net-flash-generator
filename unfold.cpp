#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdint>
#include <thread>
#include <atomic>
#include <chrono>

using u64 = unsigned long long;

const int DEFAULT_WORKERS = 8;

// ---------- DSU ----------
struct DSU {
    std::vector<int> p;
    DSU(int n = 0) { init(n); }
    DSU() = default;
    void init(int n) { p.resize(n); std::iota(p.begin(), p.end(), 0); }
    int find(int u) const {
        while (p[u] != u) u = p[u];
        return u;
    }
    int find(int u) {
        return p[u] == u ? u : p[u] = find(p[u]);
    }
    bool unite(int u, int v) {
        u = find(u); v = find(v);
        if (u == v) return false;
        p[v] = u;
        return true;
    }
};

// ---------- OFF Parser ----------
struct Polyhedron {
    int V = 0, F = 0, E_off = 0;
    std::vector<std::array<double, 3>> verts;
    std::vector<std::vector<int>> faces;
    std::vector<std::array<int, 2>> adjEdges;
    int N = 0;
    int M = 0;
};

Polyhedron readOFF(const std::string& path, std::string& rawContent) {
    Polyhedron P;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: cannot open " << path << std::endl;
        std::exit(1);
    }

    std::ostringstream oss;
    oss << ifs.rdbuf();
    rawContent = oss.str();

    std::istringstream iss(rawContent);
    std::string magic;
    iss >> magic;
    if (magic != "OFF") {
        std::cerr << "Error: expected OFF, got: " << magic << std::endl;
        std::exit(1);
    }

    iss >> P.V >> P.F >> P.E_off;

    P.verts.resize(P.V);
    for (int i = 0; i < P.V; ++i) {
        iss >> P.verts[i][0] >> P.verts[i][1] >> P.verts[i][2];
    }

    P.faces.resize(P.F);
    for (int i = 0; i < P.F; ++i) {
        int n;
        iss >> n;
        P.faces[i].resize(n);
        for (int j = 0; j < n; ++j) {
            iss >> P.faces[i][j];
        }
    }

    std::map<std::pair<int, int>, std::vector<int>> edgeToFaces;
    for (int i = 0; i < P.F; ++i) {
        const auto& f = P.faces[i];
        int n = f.size();
        for (int j = 0; j < n; ++j) {
            int a = f[j], b = f[(j + 1) % n];
            if (a > b) std::swap(a, b);
            edgeToFaces[{a, b}].push_back(i);
        }
    }

    for (const auto& kv : edgeToFaces) {
        const auto& fl = kv.second;
        if (fl.size() == 2) {
            P.adjEdges.push_back({fl[0], fl[1]});
        } else if (fl.size() > 2) {
            std::cerr << "Warning: edge shared by " << fl.size() << " faces" << std::endl;
        }
    }

    P.N = P.F;
    P.M = (int)P.adjEdges.size();
    return P;
}

// ---------- 阶段1：并行枚举生成树 ----------
struct STTask {
    int depth;
    int n;
    u64 mask;
    DSU dsu;
};

void spanningTreeDFSWorker(int depth, int n, u64 mask, const DSU& dsu,
                           int g_N, int g_M,
                           const std::vector<std::array<int, 2>>& g_edges,
                           std::vector<u64>& out) {
    if (n == g_N - 1) {
        out.push_back(mask);
        return;
    }
    if (depth == g_M) return;

    spanningTreeDFSWorker(depth + 1, n, mask, dsu, g_N, g_M, g_edges, out);

    int u = dsu.find(g_edges[depth][0]);
    int v = dsu.find(g_edges[depth][1]);
    if (u == v) return;
    DSU newDsu = dsu;
    newDsu.unite(u, v);
    spanningTreeDFSWorker(depth + 1, n + 1, mask | (u64(1) << depth), newDsu,
                          g_N, g_M, g_edges, out);
}

void generateTasks(int depth, int n, u64 mask, const DSU& dsu,
                   int g_N, int g_M,
                   const std::vector<std::array<int, 2>>& g_edges,
                   std::vector<STTask>& tasks,
                   int splitDepth, size_t maxTasks) {
    if (n == g_N - 1) {
        tasks.push_back({-1, n, mask, dsu});
        return;
    }
    if (depth == g_M) return;

    if (depth >= splitDepth || tasks.size() >= maxTasks) {
        tasks.push_back({depth, n, mask, dsu});
        return;
    }

    generateTasks(depth + 1, n, mask, dsu, g_N, g_M, g_edges,
                  tasks, splitDepth, maxTasks);

    int u = dsu.find(g_edges[depth][0]);
    int v = dsu.find(g_edges[depth][1]);
    if (u != v) {
        DSU newDsu = dsu;
        newDsu.unite(u, v);
        generateTasks(depth + 1, n + 1, mask | (u64(1) << depth), newDsu,
                      g_N, g_M, g_edges, tasks, splitDepth, maxTasks);
    }
}

// ---------- 阶段2：自同构群 ----------
void automorphismDFS(int depth, u64 mask, std::vector<int>& vertexMapping,
                     int g_N,
                     const std::vector<std::vector<char>>& g_mat,
                     std::vector<std::vector<int>>& g_edgeMappings,
                     int g_M,
                     const std::vector<std::array<int, 2>>& g_edges) {
    if (depth == g_N) {
        std::map<std::pair<int, int>, int> mp;
        for (int i = 0; i < g_M; ++i) {
            int u = vertexMapping[g_edges[i][0]];
            int v = vertexMapping[g_edges[i][1]];
            if (u > v) std::swap(u, v);
            mp[{u, v}] = i;
        }
        std::vector<int> edgeMapping(g_M);
        for (int i = 0; i < g_M; ++i) {
            int u = g_edges[i][0], v = g_edges[i][1];
            if (u > v) std::swap(u, v);
            edgeMapping[i] = mp[{u, v}];
        }
        g_edgeMappings.push_back(edgeMapping);
        return;
    }

    for (int i = 0; i < g_N; ++i) {
        if (mask & (u64(1) << i)) continue;
        bool ok = true;
        for (int j = 0; j < depth; ++j) {
            if (g_mat[depth][j] != g_mat[i][vertexMapping[j]]) {
                ok = false;
                break;
            }
        }
        if (ok) {
            vertexMapping[depth] = i;
            automorphismDFS(depth + 1, mask | (u64(1) << i), vertexMapping,
                            g_N, g_mat, g_edgeMappings, g_M, g_edges);
        }
    }
}

// ---------- 阶段3：查表法 + 排序去重 ----------
constexpr int CHUNK_BITS = 6;
constexpr int CHUNK_TABLE_SIZE = 1 << CHUNK_BITS; // 64

// 扁平化查表: [automorphism * numChunks * 64 + chunk * 64 + value]
std::vector<u64> buildPermTables(int g_M, int autCount,
                                 const std::vector<std::vector<int>>& g_edgeMappings) {
    int numChunks = (g_M + CHUNK_BITS - 1) / CHUNK_BITS;
    std::vector<u64> tables(autCount * numChunks * CHUNK_TABLE_SIZE, 0);

    for (int p = 0; p < autCount; ++p) {
        size_t base = static_cast<size_t>(p) * numChunks * CHUNK_TABLE_SIZE;
        for (int c = 0; c < numChunks; ++c) {
            for (int val = 0; val < CHUNK_TABLE_SIZE; ++val) {
                u64 newMask = 0;
                for (int b = 0; b < CHUNK_BITS; ++b) {
                    if (val & (1 << b)) {
                        int srcBit = c * CHUNK_BITS + b;
                        if (srcBit >= g_M) break;
                        int dstBit = g_edgeMappings[p][srcBit];
                        newMask |= (1ULL << dstBit);
                    }
                }
                tables[base + c * CHUNK_TABLE_SIZE + val] = newMask;
            }
        }
    }
    return tables;
}

inline u64 canonicalFormFast(u64 mask, int autCount, int numChunks,
                             const std::vector<u64>& permTables) {
    u64 best = mask;
    for (int p = 0; p < autCount; ++p) {
        u64 newMask = 0;
        size_t base = static_cast<size_t>(p) * numChunks * CHUNK_TABLE_SIZE;
        for (int c = 0; c < numChunks; ++c) {
            int chunk = (mask >> (c * CHUNK_BITS)) & (CHUNK_TABLE_SIZE - 1);
            newMask |= permTables[base + c * CHUNK_TABLE_SIZE + chunk];
        }
        if (newMask < best) best = newMask;
    }
    return best;
}

// ---------- Main ----------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.off> [output.bin] [workers]" << std::endl;
        return 1;
    }

    std::string offPath = argv[1];
    std::string outPath = (argc >= 3) ? argv[2] : "nets.bin";
    int workers = DEFAULT_WORKERS;
    if (argc >= 4) workers = std::stoi(argv[3]);
    workers = std::max(1, workers);

    std::string rawOFF;
    auto P = readOFF(offPath, rawOFF);

    int g_N = P.N;
    int g_M = P.M;
    auto g_edges = P.adjEdges;

    std::cout << "Vertices: " << P.V << ", Faces: " << P.F << std::endl;
    std::cout << "Face adjacency graph: N=" << g_N << ", M=" << g_M << std::endl;
    std::cout << "Workers: " << workers << std::endl;

    if (g_M > 60) {
        std::cerr << "Error: M=" << g_M << " > 60. u64 mask overflow risk." << std::endl;
        return 1;
    }

    // ============================================================
    // 1. 并行枚举生成树
    // ============================================================
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<STTask> tasks;
    int splitDepth = g_M / 2;
    if (splitDepth > 20) splitDepth = 20;
    if (splitDepth < 0) splitDepth = 0;
    size_t maxTasks = static_cast<size_t>(workers) * 10000;

    DSU dsu(g_N);
    generateTasks(0, 0, 0, dsu, g_N, g_M, g_edges,
                  tasks, splitDepth, maxTasks);

    std::cout << "Generated " << tasks.size()
              << " tasks (splitDepth=" << splitDepth << ")" << std::endl;

    std::vector<std::vector<u64>> threadResults(workers);
    std::atomic<size_t> taskIdx{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < workers; ++t) {
        threads.emplace_back([&, t]() {
            while (true) {
                size_t idx = taskIdx.fetch_add(1);
                if (idx >= tasks.size()) break;
                const auto& task = tasks[idx];
                if (task.depth == -1) {
                    threadResults[t].push_back(task.mask);
                } else {
                    spanningTreeDFSWorker(task.depth, task.n, task.mask,
                                          task.dsu, g_N, g_M, g_edges,
                                          threadResults[t]);
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    std::vector<u64> allTrees;
    size_t totalTrees = 0;
    for (auto& res : threadResults) totalTrees += res.size();
    allTrees.reserve(totalTrees);
    for (auto& res : threadResults) {
        allTrees.insert(allTrees.end(), res.begin(), res.end());
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Spanning trees: " << allTrees.size() << " ("
              << std::chrono::duration<double>(t2 - t1).count() << "s)"
              << std::endl;

    // ============================================================
    // 2. 计算自同构群
    // ============================================================
    auto t3 = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<char>> g_mat(g_N, std::vector<char>(g_N, 0));
    for (int i = 0; i < g_M; ++i) {
        int u = g_edges[i][0], v = g_edges[i][1];
        g_mat[u][v] = g_mat[v][u] = 1;
    }
    std::vector<std::vector<int>> g_edgeMappings;
    std::vector<int> vm(g_N);
    automorphismDFS(0, 0, vm, g_N, g_mat, g_edgeMappings, g_M, g_edges);

    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "Automorphisms: " << g_edgeMappings.size() << " ("
              << std::chrono::duration<double>(t4 - t3).count() << "s)"
              << std::endl;

    // ============================================================
    // 3. 查表法并行计算 canonical form
    // ============================================================
    auto t5 = std::chrono::high_resolution_clock::now();

    int numChunks = (g_M + CHUNK_BITS - 1) / CHUNK_BITS;
    auto permTables = buildPermTables(g_M, (int)g_edgeMappings.size(), g_edgeMappings);

    std::vector<u64> canonicals(allTrees.size());
    size_t blockSize = (allTrees.size() + workers - 1) / workers;

    threads.clear();
    for (int t = 0; t < workers; ++t) {
        size_t start = t * blockSize;
        size_t end = std::min(start + blockSize, allTrees.size());
        if (start >= end) continue;
        threads.emplace_back([&, start, end]() {
            for (size_t i = start; i < end; ++i) {
                canonicals[i] = canonicalFormFast(allTrees[i],
                                                  (int)g_edgeMappings.size(),
                                                  numChunks, permTables);
            }
        });
    }
    for (auto& th : threads) th.join();

    auto t6 = std::chrono::high_resolution_clock::now();
    std::cout << "Canonical forms computed: "
              << std::chrono::duration<double>(t6 - t5).count() << "s"
              << std::endl;

    // ============================================================
    // 4. 排序去重
    // ============================================================
    std::vector<size_t> idx(allTrees.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
        [&](size_t a, size_t b) { return canonicals[a] < canonicals[b]; });

    std::vector<u64> representatives;
    representatives.reserve(allTrees.size() / 100);
    for (size_t i = 0; i < idx.size(); ++i) {
        if (i == 0 || canonicals[idx[i]] != canonicals[idx[i - 1]]) {
            representatives.push_back(allTrees[idx[i]]);
        }
    }

    auto t7 = std::chrono::high_resolution_clock::now();
    std::cout << "Non-isomorphic nets: " << representatives.size()
              << " (total dedup: "
              << std::chrono::duration<double>(t7 - t5).count() << "s)"
              << std::endl;

    // ============================================================
    // 5. 写二进制文件
    // ============================================================
    std::ofstream ofs(outPath, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: cannot create " << outPath << std::endl;
        return 1;
    }

    u64 offLen = rawOFF.size();
    ofs.write(reinterpret_cast<const char*>(&offLen), sizeof(offLen));
    ofs.write(rawOFF.data(), offLen);

    int32_t m32 = g_M;
    ofs.write(reinterpret_cast<const char*>(&m32), sizeof(m32));

    for (int i = 0; i < g_M; ++i) {
        int32_t a = g_edges[i][0];
        int32_t b = g_edges[i][1];
        ofs.write(reinterpret_cast<const char*>(&a), sizeof(a));
        ofs.write(reinterpret_cast<const char*>(&b), sizeof(b));
    }

    u64 count = representatives.size();
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
    ofs.write(reinterpret_cast<const char*>(representatives.data()),
              count * sizeof(u64));

    ofs.close();

    size_t totalSize = sizeof(offLen) + offLen + sizeof(m32) +
                       static_cast<size_t>(g_M) * 8 + sizeof(count) + count * sizeof(u64);
    std::cout << "Wrote " << count << " nets to " << outPath << std::endl;
    std::cout << "File size: " << totalSize << " bytes" << std::endl;

    return 0;
}
