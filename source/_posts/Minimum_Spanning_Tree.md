---
title: Minimum Spanning Tree
mathjax: true
date: 2020-11-27 15:09:39
categories: Algorithm
description: 最小生成树的Prim算法和Kruskal算法
---

## 最小生成树

有`n`个城市，任意两个城市间修路要花钱，用`n-1`条路在最小开销的情况下连通所有城市，就是最小生成树

[模板题：洛谷P3366 【模板】最小生成树](https://www.luogu.com.cn/problem/P3366)

## Prim算法

### 描述

此算法又称作加点法

1. 输入：一个加权连通图，其中顶点集合为$V$，边集合为$E$
2. 初始化：$V_{new} = ${$x$}，其中$x$为$V$的任一节点（起始点），$E_{new} = ${}
3. 重复下列操作，直到$V_{new} = V$
   1. 在$E$中选取权值最小的边$(u, v)$，其中$u∈V_{new}$且$v∉V_{new}$
   2. 将$v$加入$V_{new}$，将$(u, v)加入E_{new}$
4. 输出：使用$V_{new}和E_{new}来描述所得到的最小生成树$

### 具体实现

- 维护一个数组`vis[]`，令所有$u∈V_{new}$的`vis[u] = true`
- 维护一个数组`dist[]`，使得`dist[v]`是以访问过的点为出点，v为入点的最小边长
- 这样，在执行Prim算法3.1时，我们只需要寻找满足`vis[v] = false`和`dist[v]`最小的`v`
- 每次找到`v`后，执行Prim算法3.2，令`vis[v] = 1`，并且`res += dist[v]`
- 为了维护`dist[]`，每次找到`v`后，`v`变为访问过的出点，则更新以`v`为出点未访问过的为入点的最小边长

- 空间复杂度：$O(V^2)$
- 时间复杂度：$O(V^2)$

### Prim模板

```cpp
template<class T>
int32_t Prim(const T &graph, const size_t &n) {
  int32_t res = 0;
  bool *vis = new bool[n + 1];
  int32_t *dist = new int32_t[n + 1];

  for (int i = 0; i <= n; ++i) {
    vis[i] = 0;
    dist[i] = kInf;
  }

  dist[1] = 0;
  for (int i = 1; i <= n; ++i) {
    int32_t x = 0;
    for (int j = 1; j <= n; ++j)
      if (!vis[j] && dist[j] < dist[x]) x = j;

    vis[x] = true;
    res += dist[x];

    for (int j = 1; j <= n; ++j)
      if (!vis[j] && dist[j] > graph[x][j]) dist[j] = graph[x][j];
  }

  delete[] dist;
  delete[] vis;
  return res;
}
```

### 堆优化

- 执行Prim算法3.1时，我们只需要寻找满足`vis[v] = false`和`dist[v]`最小的`v`
- 根据`dist[]`的性质，可对其使用堆排序
- 这样，在执行Prim算法3.1时，从堆中取出记录有入点和权值的边
- 若入点已访问，则舍弃；否则，执行Prim算法3.2，令`vis[v] = 1`，并且`res += 权值`
- 每次找到`v`后，`v`变为访问过的出点，则更新以`v`为出点未访问过的为入点的最小边长，将其放入堆中

- 空间复杂度：$O(V^2)$
- 时间复杂度：$O(ElogV)$（通常要使用邻接矩阵）

### 堆优化Prim模板

```cpp
struct Node {
  int val, p;
  inline bool operator<(const Node &other) const { return val > other.val; }
};

template <class T>
int32_t HeapPrim(const T &graph, const size_t &n) {
  int32_t res = 0;
  bool *vis = new bool[n + 1];
  std::priority_queue<Node> heap;

  for (int i = 1; i <= n; ++i) vis[i] = 0;

  for (heap.push(Node{0, 1}); !heap.empty();) {
    Node node = heap.top();
    heap.pop();
    if (vis[node.p]) continue;

    vis[node.p] = true;
    res += node.val;

    for (int i = 1; i <= n; ++i) {
      if (vis[i]) continue;
      if (graph[node.p][i] != kInf) heap.push(Node{graph[node.p][i], i});
    }
  }

  delete[] vis;
  return res;
}
```

## Kruskal算法

### 描述

此算法又称作加边法

1. 输入：一个加权连通图，其中顶点集合为$V$，边集合为$E$
2. 初始化：新建图G，G中拥有原图中相同的节点，但没有边
3. 重复以下步骤，知道G中所有节点都在同一个连同分量中
   1. 取出权值最小的边
   2. 如果这条边连接的两个节点于图G中不在同一个连通分量中，则添加这条边到图G中

### 实现

- 使用并查集来维护连通分量

- 空间复杂度：$O(E+V)$
- 时间复杂度：$O(ElogE)$

### Kruskal模板

```cpp
class DisjointSet {
  int32_t *_;

 public:
  DisjointSet(const int &n) : _(new int32_t[n + 1]) {
    for (int i = 1; i <= n; ++i) _[i] = i;
  }

  ~DisjointSet() { delete[] _; }

  int32_t Find(const int &x) { return x == _[x] ? x : _[x] = Find(_[x]); }
  void Merge(const int &x, const int &y) { _[Find(x)] = _[Find(y)]; }
};

struct Edge {
  int src, dst, val;
  inline bool operator<(const Edge &other) const { return val < other.val; }
};

int32_t Kruskal(Edge *edges, const size_t &n, const size_t &m) {
  int32_t res = 0;
  DisjointSet disjoint_set(n);

  std::sort(edges + 1, edges + 1 + m);

  for (int i = 1; i <= m; ++i) {
    int32_t src = disjoint_set.Find(edges[i].src),
            dst = disjoint_set.Find(edges[i].dst);
    if (src == dst) continue;
    disjoint_set.Merge(src, dst);
    res += edges[i].val;
  }

  return res;
}
```

## 总结

- Kruskal易于理解，且容易实现
- Prim算法在边密集的图中效率更高
- Kruskal在边稀疏的图中中效率更高
