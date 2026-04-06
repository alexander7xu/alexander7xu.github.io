---
title: Prefix Sum and Finite Difference
mathjax: true
date: 2020-12-08 23:25:38
categories: Algorithm
description: 前缀和与差分
---

## 前缀和

### 定义

$$
sum[i]=
\begin{cases}
0&                   {i=0}\\\\
\sum_{j=1}^{i}a[j]&  {i\geq 1}
\end{cases}
$$

写成代码是：
`sum[0] = 0;`
`sum[i] = sum[i - 1] + a[i]`

### 应用

- 快速求区间和

根据定义，有 $\sum_{i=l}^{r}a[i] = sum[r] - sum[l - 1]$

### 二维前缀和

![$A\cup B = A + B - A\cap B$](./Prefix_Sum_and_Finite_Difference/Inclusion–exclusion_principle.png)

$$
sum[y][x]=
\begin{cases}
0&                   {y=0\ or\ x=0}\\\\
\sum_{i=1}^{y}\sum_{j=1}^{x}a[i][j]&  {y\geq 1\ and\ x\geq 1}
\end{cases}
$$

根据容斥原理，有：
`sum[y][0] = sum[0][x] = 0`
`sum[y][x] = sum[y - 1][x] + sum[y][x - 1] - sum[y - 1][x - 1] + a[y][x]`

- 快速求二维区间和

根据定义，有 $\sum_{y=y_0}^{y1}\sum_{x=x_0}^{x1}a[y][x] = sum[y1][x1] - sum[y0 - 1][x1] - sum[y1][x0 - 1] + sum[y0 - 1][x0 - 1]$

### 模板

```cpp
void GeneratePrefixSum(int sum[], const int a[], const int &size) {
  sum[0] = 0;
  for (int i = 1; i <= size; ++i) sum[i] = sum[i - 1] + a[i];
}

inline int GetSum(const int sum[], const int &l, const int &r) {
  return sum[r] - sum[l - 1];
}

void GeneratePrefixSum2D(int *sum[], const int *a[], const int &size_y,
                         const int &size_x) {
  for (int x = 0; x <= size_x; ++x) sum[0][x] = 0;

  for (int y = 1; y <= size_y; ++y) {
    sum[y][0] = 0;
    for (int x = 1; x <= size_x; ++x)
      sum[y][x] = sum[y - 1][x] + sum[y][x - 1] - sum[y - 1][x - 1] + a[y][x];
  }
}

inline int GetSum2D(const int *sum[], const int &y0, const int &x0,
                    const int &y1, const int &x1) {
  return sum[y1][x1] - sum[y0 - 1][x1] - sum[y1][x0 - 1] + sum[y0][x0];
}
```

## 差分

### 定义

$diff[i]=a[i] - a[i-1] (i\geq 1且a[0] = 0)$

差分是前缀和的逆运算，$a[]$是$diff[]$的前缀和

### 应用

- 区间染色转端点染色

若有操作 $a[i] += t x\leq i\leq y$，可以令 $diff[x] += t, diff[y + 1] -= t$，对$diff$求前缀和，$sum[i]$即得所有操作对$a[i]$的影响

### 二维差分

二维前缀和的逆运算

`diff[y][x] = a[y][x] - a[y - 1][x] - a[y][x - 1] + a[y - 1][x - 1]`

### 模板

```cpp
void GenerateFiniteDiff(int diff[], const int a[], const int &size) {
  for (int i = 1; i <= size; ++i) diff[i] = a[i] - a[i - 1];
}

inline void Modify(int diff[], const int &l, const int &r, const int &val) {
  diff[l] += val, diff[r + 1] -= val;
}

void GenerateFiniteDiff2D(int *diff[], const int *a[], const int &size_y,
                          const int &size_x) {
  for (int y = 1; y <= size_y; ++y) {
    for (int x = 1; x <= size_x; ++x)
      diff[y][x] = a[y][x] - a[y - 1][x] - a[y][x - 1] + a[y - 1][x - 1];
  }
}

inline void Modify2D(int *diff[], const int &y0, const int &x0, const int &y1,
                     const int &x1, const int &val) {
  diff[y0][x0] += val;
  diff[y1 + 1][x0] -= val;
  diff[y0][x1 + 1] -= val;
  diff[y1 + 1][x1 + 1] += val;
}
```

## 解题

[CF1262E. Arson In Berland Forest](https://codeforces.com/problemset/problem/1262/E)

题意：给一张有`x`和`.`两个标记的图，已知每单位时间`X`会扩散到周围8格，求最长的扩散时间及此时初始的图

### 思路

- 数据规模是$10^6$，考虑用**二分答案**求可能的时间，下界为0，上界为$max(m, n)$。
- 确定`Judge(t)`函数：
  - 对于答案的`X`点，在时间为$t$时，以其为左上角，$2t+1$为边长的正方形在输入图中都必须是`X`
  - 那么可以枚举所有的`X`点，符合条件（输入图中以其为左上角的`X`必须有$(2t+1)^2$个）时对以其为左上角，边长为$2t+1$的正方形进行染色
  - 最终若有输入图的`X`未被染色（染色数量小于输入图`X`的数量）则`Judge(t)`失败，反之成功
- 考虑对其进行优化：
  - 枚举符合条件的点时，如何快速计算输入图中以其为中心的`X`的数量？使用**二维前缀和**处理输入图
  - 如何计算染色数量？使用**二维差分**进行端点染色，对差分计算**二维前缀和**后，将非0转为1，再计算一次**二维前缀和**
- 时间复杂度：$O(nmlogn)$ 大约是$10^7$，本题 time limit per test: 2s

### AC代码

```cpp
#include <iostream>
#include <vector>

typedef std::vector<std::vector<int64_t> > V;
int n, m;
char str[int(1e6 + 5)];
V sum, diff, ds;

inline int d(const int &x) { return x << 1; }

inline int64_t GetSum2D(const V &sum, const int &y0, const int &x0,
                        const int &y1, const int &x1) {
  return sum[y1][x1] - sum[y0 - 1][x1] - sum[y1][x0 - 1] + sum[y0 - 1][x0 - 1];
}

inline void Modify2D(V &diff, const int &y0, const int &x0, const int &y1,
                     const int &x1, const int64_t &val) {
  diff[y0][x0] += val;
  diff[y1 + 1][x1 + 1] += val;
  diff[y1 + 1][x0] -= val;
  diff[y0][x1 + 1] -= val;
}

bool Judge(const int64_t &t) {
  for (int i = 1; i <= n + 1; ++i) diff[i].assign(m + 2, 0);

  for (int i = 1, sz_i = n - d(t); i <= sz_i; ++i)
    for (int j = 1, sz_j = m - d(t); j <= sz_j; ++j)
      if (GetSum2D(sum, i, j, i + d(t), j + d(t)) ==
          int64_t(d(t) + 1) * (d(t) + 1))
        Modify2D(diff, i, j, i + d(t), j + d(t), 1);

  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= m; ++j) {
      diff[i][j] += diff[i - 1][j] + diff[i][j - 1] - diff[i - 1][j - 1];
      ds[i][j] =
          ds[i - 1][j] + ds[i][j - 1] - ds[i - 1][j - 1] + (diff[i][j] > 0);
    }

  return GetSum2D(sum, 1, 1, n, m) == GetSum2D(ds, 1, 1, n, m);
}

int main() {
  std::ios::sync_with_stdio(0);
  std::cin.tie(0), std::cout.tie(0);

  std::cin >> n >> m;

  // 计算输入图中'X'数量的前缀和
  sum.resize(n + 1);
  sum[0].resize(m + 1, 0);
  for (int i = 1; i <= n; ++i) {
    std::cin >> str + 1;

    sum[i].resize(m + 1);
    sum[i][0] = 0;
    for (int j = 1; j <= m; ++j)
      sum[i][j] =
          sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + (str[j] == 'X');
  }

  // 初始化差分及其前缀和
  diff.resize(n + 2);
  ds.resize(n + 1);
  diff[0].resize(m + 2, 0);
  for (int i = 0; i <= n; ++i) ds[i].resize(m + 1, 0);

  int r = (n > m ? n : m);
  for (int l = 0, mid; l < r;) {
    mid = (l + r) >> 1;
    if (Judge(mid)) {
      l = mid + 1;
    } else {
      r = mid;
    }
  }

  std::cout << --r << "\n";
  for (int i = 1; i <= n; ++i) diff[i].assign(m + 1, 0);

  for (int i = 1, sz_i = n - d(r); i <= sz_i; ++i)
    for (int j = 1, sz_j = m - d(r); j <= sz_j; ++j)
      if (GetSum2D(sum, i, j, i + d(r), j + d(r)) == (d(r) + 1) * (d(r) + 1))
        diff[i + r][j + r] = 1;

  for (int i = 1; i <= n; ++i) {
    for (int j = 1; j <= m; ++j) std::cout << (diff[i][j] ? "X" : ".");
    std::cout << "\n";
  }

  return 0;
}
```
