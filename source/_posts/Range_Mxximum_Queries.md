---
title: Range Mxximum Queries
mathjax: true
date: 2020-12-16 01:12:23
categories: Algorithm
description: 给定一个大小为$n$的数组，及$m$组询问，每次询问求区间$[l,r]$的最大（最小）值
---

## 暴力算法

以区间最大值查询为例：

```cpp
// a[n + 1]：给定数组
// max[n + 1][n + 1]：max[l][r]记录a中区间[l,r]的最小值
for (int i = 1; i <= n; ++i) {
  max[i][i] = a[i];
  for (int j = i + 1; j <= n; ++j)
    max[i][j] = std::min(max[i][j - 1], a[j]);
  }
```

暴力算法的时间和空间复杂度都为：$O(n^2)$

## 优化思路：分治法，记忆化搜索

先考虑求单次查询时显然有：$Min(a[i]|l\leq i\leq r)=Min(Min(a[i]|l\leq i\leq ⌊\frac{l+r}{2}⌋), Min(a[i]|⌈\frac{l+r}{2}⌉\leq i\leq r))$

```cpp
int RMQMax(const int &l, const int &r) {
  if (l == r) return a[l];
  int &&m = (l + r) >> 1;
  return std::max(RMQMax(l, m), RMQMax(m + 1, r));
}
```

单次查询的时间复杂度是：$O(logn)$

但如何把分治法用于多次查询呢？考虑**记忆化搜索**：在分治过程中打表
- `RMQMax`的时间复杂度是$O(logn)$，故考虑$O(nlogn)$的表
- 每次递归都把区间对半分割，故可考虑省去右端点，而改为记录区间长度$2^loglen$，这样每次递归只需把$loglen-=1$
- 当区间长度不是$2$的整数次幂时，只需要把原区间划分成两个长为$2$的整数次幂的区间即可

### 模板

[模板题：HRBUST1189 区间最大值II](https://vjudge.net/problem/HRBUST-1189)

```cpp
// Init
std::memset(dp, 0xff, sizeof(dp));

int RMQDAC(const int &l, const int &loglen) {
  if (dp[l][loglen] != -1) return dp[l][loglen];
  if (loglen == 0) return dp[l][0] = a[l];
  return dp[l][loglen] = std::max(RMQMax(l, loglen - 1),
                                  RMQMax(l + Pow2(loglen - 1), loglen - 1));
}

inline int RMQMax(const int &l, const int &r) {
  int loglen = std::log2(r - l + 1);
  return std::max(RMQDAC(l, loglen), RMQDAC(r - Pow2(loglen) + 1, loglen));
}
```

## 动态规划的Sparse Table（ST）算法

分析一下记忆化搜索，就可以把它化作多态规划的ST算法

### 模板

```cpp
inline int Pow2(const int &x) { return 1 << x; }
void RMQST() {
  for (int i = 1; i <= n; ++i) max[i][0] = min[i][0] = a[i];
  for (int j = 1; Pow2(j) <= n; ++j)
    for (int i = 1; i + Pow2(j - 1) <= n; ++i) {
      max[i][j] = std::max(max[i][j - 1], max[i + Pow2(j - 1)][j - 1]);
      min[i][j] = std::min(min[i][j - 1], min[i + Pow2(j - 1)][j - 1]);
    }
}
inline int RMQMax(const int &l, const int &r) {
  int loglen = log2(r - l + 1);
  return std::max(max[l][loglen], max[r - Pow2(loglen) + 1][loglen]);
}
inline int RMQMin(const int &l, const int &r) {
  int loglen = log2(r - l + 1);
  return std::min(min[l][loglen], min[r - Pow2(loglen) + 1][k]);
}
```# 区间最大（最小）值查询

给定一个大小为$n$的数组，及$m$组询问，每次询问求区间$[l,r]$的最大（最小）值

### 暴力算法

以区间最大值查询为例：

```cpp
// a[n + 1]：给定数组
// max[n + 1][n + 1]：max[l][r]记录a中区间[l,r]的最小值
for (int i = 1; i <= n; ++i) {
  max[i][i] = a[i];
  for (int j = i + 1; j <= n; ++j)
    max[i][j] = std::min(max[i][j - 1], a[j]);
  }
```

暴力算法的时间和空间复杂度都为：$O(n^2)$

## 优化思路：分治法，记忆化搜索

先考虑求单次查询时显然有：$Min(a[i]|l\leq i\leq r)=Min(Min(a[i]|l\leq i\leq ⌊\frac{l+r}{2}⌋), Min(a[i]|⌈\frac{l+r}{2}⌉\leq i\leq r))$

```cpp
int RMQMax(const int &l, const int &r) {
  if (l == r) return a[l];
  int &&m = (l + r) >> 1;
  return std::max(RMQMax(l, m), RMQMax(m + 1, r));
}
```

单次查询的时间复杂度是：$O(logn)$

但如何把分治法用于多次查询呢？考虑**记忆化搜索**：在分治过程中打表
- `RMQMax`的时间复杂度是$O(logn)$，故考虑$O(nlogn)$的表
- 每次递归都把区间对半分割，故可考虑省去右端点，而改为记录区间长度$2^loglen$，这样每次递归只需把$loglen-=1$
- 当区间长度不是$2$的整数次幂时，只需要把原区间划分成两个长为$2$的整数次幂的区间即可

### 模板

[模板题：HRBUST1189 区间最大值II](https://vjudge.net/problem/HRBUST-1189)

```cpp
// Init
std::memset(dp, 0xff, sizeof(dp));

int RMQDAC(const int &l, const int &loglen) {
  if (dp[l][loglen] != -1) return dp[l][loglen];
  if (loglen == 0) return dp[l][0] = a[l];
  return dp[l][loglen] = std::max(RMQMax(l, loglen - 1),
                                  RMQMax(l + Pow2(loglen - 1), loglen - 1));
}

inline int RMQMax(const int &l, const int &r) {
  int loglen = std::log2(r - l + 1);
  return std::max(RMQDAC(l, loglen), RMQDAC(r - Pow2(loglen) + 1, loglen));
}
```

## 动态规划的Sparse Table（ST）算法

分析一下记忆化搜索，就可以把它化作多态规划的ST算法

### 模板

```cpp
inline int Pow2(const int &x) { return 1 << x; }
void RMQST() {
  for (int i = 1; i <= n; ++i) max[i][0] = min[i][0] = a[i];
  for (int j = 1; Pow2(j) <= n; ++j)
    for (int i = 1; i + Pow2(j - 1) <= n; ++i) {
      max[i][j] = std::max(max[i][j - 1], max[i + Pow2(j - 1)][j - 1]);
      min[i][j] = std::min(min[i][j - 1], min[i + Pow2(j - 1)][j - 1]);
    }
}
inline int RMQMax(const int &l, const int &r) {
  int loglen = log2(r - l + 1);
  return std::max(max[l][loglen], max[r - Pow2(loglen) + 1][loglen]);
}
inline int RMQMin(const int &l, const int &r) {
  int loglen = log2(r - l + 1);
  return std::min(min[l][loglen], min[r - Pow2(loglen) + 1][k]);
}
```
