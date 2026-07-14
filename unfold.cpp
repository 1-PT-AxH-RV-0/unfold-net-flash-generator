#include <iostream>
#include <vector>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdint>

using u64 = unsigned long long;

// ---------- DSU ----------
struct DSU {
    std::vector<int> p;
    DSU(int n = 0) { init(n); }
    void init(int n) { p.resize(n); std::iota(p.begin(), p.end(), 0); }
    int find(int u) { return p[u] == u ? u : p[u] = find(p[u]); }
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
    int N = 0; // number of faces
    int M = 0; // number of adjacency edges
};

Polyhedron readOFF(const std::string& path, std::string& rawContent) {
    Polyhedron P;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: cannot open " << path << std::endl;
        std::exit(1);
    }

    // Read entire file content
    std::ostringstream oss;
    oss << ifs.rdbuf();
    rawContent = oss.str();

    // Parse from string
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

    // Build face adjacency graph
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

// ---------- Global state ----------
std::vector<std::array<int, 2>> g_edges;
int g_N = 0, g_M = 0;
std::unordered_set<u64> g_spanningTrees;

void spanningTreeDFS(int depth, int n, u64 mask, DSU dsu) {
    if (n == g_N - 1) {
        g_spanningTrees.insert(mask);
        return;
    }
    if (depth == g_M) return;

    spanningTreeDFS(depth + 1, n, mask, dsu);

    int u = dsu.find(g_edges[depth][0]);
    int v = dsu.find(g_edges[depth][1]);
    if (u == v) return;
    DSU newDsu = dsu;
    newDsu.unite(u, v);
    spanningTreeDFS(depth + 1, n + 1, mask | (u64(1) << depth), newDsu);
}

// ---------- Automorphism group ----------
std::vector<std::vector<char>> g_mat;
std::vector<std::vector<int>> g_edgeMappings;

void automorphismDFS(int depth, u64 mask, std::vector<int>& vertexMapping) {
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
            automorphismDFS(depth + 1, mask | (u64(1) << i), vertexMapping);
        }
    }
}

// ---------- Deduplication ----------
std::unordered_set<u64> g_seen;
std::vector<u64> g_representatives;

bool calcTreeHash(u64 mask) {
    std::unordered_set<u64> orbit;
    for (const auto& mapping : g_edgeMappings) {
        u64 newMask = 0;
        for (int i = 0; i < g_M; ++i) {
            if (mask & (u64(1) << i)) newMask |= (u64(1) << mapping[i]);
        }
        if (g_seen.count(newMask)) return false;
        orbit.insert(newMask);
    }
    g_seen.insert(orbit.begin(), orbit.end());
    g_representatives.push_back(mask);
    return true;
}

// ---------- Main ----------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.off> [output.bin]" << std::endl;
        return 1;
    }

    std::string offPath = argv[1];
    std::string outPath = (argc >= 3) ? argv[2] : "nets.bin";

    std::string rawOFF;
    auto P = readOFF(offPath, rawOFF);

    g_N = P.N;
    g_M = P.M;
    g_edges = P.adjEdges;

    std::cout << "Vertices: " << P.V << ", Faces: " << P.F << std::endl;
    std::cout << "Face adjacency graph: N=" << g_N << ", M=" << g_M << std::endl;

    if (g_M > 60) {
        std::cerr << "Error: M=" << g_M << " > 60. u64 mask overflow risk." << std::endl;
        return 1;
    }

    // 1. Enumerate spanning trees
    DSU dsu(g_N);
    spanningTreeDFS(0, 0, 0, dsu);
    std::cout << "Spanning trees: " << g_spanningTrees.size() << std::endl;

    // 2. Automorphism group
    g_mat.assign(g_N, std::vector<char>(g_N, 0));
    for (int i = 0; i < g_M; ++i) {
        int u = g_edges[i][0], v = g_edges[i][1];
        g_mat[u][v] = g_mat[v][u] = 1;
    }
    std::vector<int> vm(g_N);
    automorphismDFS(0, 0, vm);
    std::cout << "Automorphisms: " << g_edgeMappings.size() << std::endl;

    // 3. Deduplicate
    int result = 0;
    for (u64 mask : g_spanningTrees) {
        result += calcTreeHash(mask);
    }
    std::cout << "Non-isomorphic nets: " << result << std::endl;
    std::cout << "Representatives: " << g_representatives.size() << std::endl;

    // 4. Write binary with embedded OFF and EDGES
    std::ofstream ofs(outPath, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: cannot create " << outPath << std::endl;
        return 1;
    }

    // Format:
    // 8 bytes: rawOFF length (uint64_t)
    // L bytes: rawOFF string
    // 4 bytes: M (int32)
    // M * 8 bytes: EDGES array (2 int32 each)
    // 8 bytes: count (uint64_t)
    // count * 8 bytes: masks (uint64_t)

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

    u64 count = g_representatives.size();
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
    ofs.write(reinterpret_cast<const char*>(g_representatives.data()), count * sizeof(u64));

    ofs.close();

    size_t totalSize = sizeof(offLen) + offLen + sizeof(m32) + g_M * 8 + sizeof(count) + count * sizeof(u64);
    std::cout << "Wrote " << count << " nets to " << outPath << std::endl;
    std::cout << "File size: " << totalSize << " bytes" << std::endl;

    return 0;
}
