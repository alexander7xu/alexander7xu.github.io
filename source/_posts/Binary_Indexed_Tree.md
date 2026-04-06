---
title: Binary Indexed Tree
mathjax: true
date: 2020-12-09 06:46:23
categories: Algorithm
description: 树状数组
---

## 树状数组

![树状数组](./Binary_Indexed_Tree/binary_indexed_tree.png)

> 按照Peter M. Fenwick的说法，正如所有的整数都可以表示成2的幂和，我们也可以把一串序列表示成一系列子序列的和。采用这个想法，我们可将一个前缀和划分成多个子序列的和，而划分的方法与数的2的幂和具有极其相似的方式。一方面，子序列的个数是其二进制表示中1的个数，另一方面，子序列代表的f[i]的个数也是2的幂。

二进制最低位的1所处的位置，表示这个节点处在树的倒数第几层，同时表示其管理的区间长度

### 应用

- 按照定义可以实现：
  - 单点修改，区间查询

- 加上{% post_link Prefix_Sum_and_Finite_Difference 前缀和与差分 %}可以实现：
  - 区间修改，单点查询
  - 区间修改，区间查询

### 单点修改，区间查询

修改一个子节点时，会影响其父节点的值；而如何从子节点索引到父节点的？

**二进制最低位的1所处的位置，表示这个节点处在树的倒数第几层，同时表示其管理的区间长度**

- 故，欲索引到父节点，即索引到上层，只需将其最低位1加1进位
- 而反之将最低位1置0,则索引到当前节点上层的，非当前节点之父的节点
- 可用LowBit函数取出最低位1


#### 模板

[模板题：树状数组1：单点修改，区间查询](https://vjudge.net/problem/LibreOJ-130)

```cpp
inline size_t LowBit(const size_t &x) { return x & -x; }

template <class DataT>
class BinaryIndexedTree final {
  DataT *data_;
  size_t size_;

 public:
  BinaryIndexedTree(const size_t &size)
      : data_(new DataT[size + 1]), size_(size) {
    for (size_t i = 0; i <= size_; ++i) data_[i] = 0;
  }

  template <class ArrT>
  BinaryIndexedTree(const ArrT &arr, const size_t &size)
      : BinaryIndexedTree(size) {
    for (size_t i = 1; i <= size_; ++i) {
      data_[i] += arr[i];
      size_t &&p = i + LowBit(i);
      if (p <= size_) data_[p] += data_[i];
    }
  }

  ~BinaryIndexedTree() { delete[] data_; }

  void Add(size_t p, const DataT &val) {
    for (; p <= size_; p += LowBit(p)) data_[p] += val;
  }

  DataT GetSum(size_t p) const {
    DataT res = 0;
    for (; p; p -= LowBit(p)) res += data_[p];
    return res;
  }

  DataT GetSum(const size_t &l, const size_t &r) const {
    return GetSum(r) - GetSum(l - 1);
  }
};
```

### 区间修改，单点查询

对差分数组建立树状数组即可

#### 模板

[模板题：树状数组2：区间修改，单点查询](https://vjudge.net/problem/LibreOJ-131)

```cpp
// auto diff = BinaryIndexedTree<int>(n + 1);

template <class DataT>
inline void Modify(BinaryIndexedTree<DataT> *diff, const size_t &l,
                   const size_t &r, const DataT &val) {
  diff->Add(l, val);
  diff->Add(r + 1, -val);
}
```

### 区间修改，区间查询

> 考虑$a[]$的前缀和与差分数组$diff[]$的关系：
> 
> $\sum_{i=1}^{n}a[i]$
> 
> $=\sum_{i=1}^{n}\sum_{j=1}^{i}diff[j]$
> 
> $=\sum_{i=1}^{n}diff[i]*(n-i+1)$
> 
> $=(n+1)\sum_{i=1}^{n}diff[i] - \sum_{i=1}^{n}diff[i]*i$

所以，可以对$diff[i]$和$diff[i]*i$分别建立树状数组

#### 模板

[模板题：树状数组3：区间修改，区间查询](https://vjudge.net/problem/LibreOJ-132)

```cpp
// GenerateBinaryIndexedTree(diff_tree, nullptr, size);
// GenerateBinaryIndexedTree(diff_muti, nullptr, size);

template <class DataT>
void Modify(BinaryIndexedTree<DataT> *diff_tree,
            BinaryIndexedTree<DataT> *diff_muti, const size_t &l,
            const size_t &r, const DataT &val) {
  diff_tree->Add(l, val);
  diff_tree->Add(r + 1, -val);
  diff_muti->Add(l, val * l);
  diff_muti->Add(r + 1, -val * (r + 1));
}

template <class DataT>
inline DataT Query(const BinaryIndexedTree<DataT> &diff_tree,
                   const BinaryIndexedTree<DataT> &diff_muti, const size_t &p) {
  return (p + 1) * diff_tree.GetSum(p) - diff_muti.GetSum(p);
}

template <class DataT>
inline DataT Query(const BinaryIndexedTree<DataT> &diff_tree,
                   const BinaryIndexedTree<DataT> &diff_muti, const size_t &l,
                   const size_t &r) {
  return Query(diff_tree, diff_tree, r) - Query(diff_tree, diff_muti, l - 1);
}
```

## 二维树状数组

**二进制最低位的1所处的位置，表示这个节点处在树的倒数第几层，同时表示其管理的区间长度**

一维树状数组$tree[i]$记录了右端点为$i$，长度为$LowBit(i)$的区间和

二维树状数组$tree[y][x]$记录了右下角为$(y, x)$，高为$LowBit(y)$，宽为$LowBit(x)$的二维区间和

### 单点修改，区间查询

只需要按照定义，使用容斥原理，对一维树状数组修改

[模板题：二维树状数组1：单点修改，区间查询](https://vjudge.net/problem/LibreOJ-133)

#### 模板

```cpp
inline size_t LowBit(const size_t &x) { return x & -x; }

template <class DataT>
class BinaryIndexedTree2D final {
  DataT **data_;
  size_t size_y_, size_x_;

 public:
  BinaryIndexedTree2D(const size_t &size_y, const size_t &size_x)
      : data_(new DataT *[size_y + 1]), size_y_(size_y), size_x_(size_x) {
    for (size_t y = 0; y <= size_y_; ++y) {
      data_[y] = new DataT [size_x_ + 1];
      for (size_t x = 0; x <= size_x_; ++x) data_[y][x] = 0;
    }
  }

  template <class Arr2DT>
  BinaryIndexedTree2D(const Arr2DT &arr2d, const size_t &size_y,
                      const size_t &size_x)
      : BinaryIndexedTree2D(size_y, size_x) {
    for (size_t y = 1; y <= size_y_; ++y)
      for (size_t x = 1; x <= size_x_; ++x) {
        data_[y][x] += arr2d[y][x];
        size_t &&xx = x + LowBit(x);
        size_t &&yy = y + LowBit(y);

        if (xx <= size_x_) data_[y][xx] += data_[y][x];
        if (yy <= size_y_) data_[yy][x] += data_[y][x];
        if (xx <= size_x_ && yy <= size_y_) data_[yy][xx] += data_[y][x];
      }
  }

  ~BinaryIndexedTree2D() {
    for (size_t y = 0; y <= size_y_; ++y) delete[] data_[y];
    delete[] data_;
  }

  void Add(const size_t &y, const size_t &x, const DataT &val) {
    for (size_t yy = y; yy <= size_y_; yy += LowBit(yy))
      for (size_t xx = x; xx <= size_x_; xx += LowBit(xx)) data_[yy][xx] += val;
  }

  DataT GetSum(const size_t &y, const size_t &x) const {
    DataT res = 0;
    for (size_t yy = y; yy; yy -= LowBit(yy))
      for (size_t xx = x; xx; xx -= LowBit(xx)) res += data_[yy][xx];
    return res;
  }

  DataT GetSum(const size_t &y0, const size_t &x0, const size_t &y1,
               const size_t &x1) const {
    return GetSum(y1, x1) - GetSum(y0 - 1, x1) - GetSum(y1, x0 - 1) +
           GetSum(y0 - 1, x0 - 1);
  }
};
```

### 区间修改，单点查询

对二维差分数组建立二维树状数组即可

#### 模板

[模板题：二维树状数组2：区间修改，单点查询](https://vjudge.net/problem/LibreOJ-134)

```cpp
// auto diff = BinaryIndexedTree2D<int64_t>(n, m);

template <class DataT>
inline void Modify2D(BinaryIndexedTree2D<DataT> *diff, const size_t &y0,
                   const size_t &x0, const size_t &y1, const size_t &x1,
                   const DataT &val) {
  diff->Add(y0, x0, val);
  diff->Add(y1 + 1, x0, -val);
  diff->Add(y0, x1 + 1, -val);
  diff->Add(y1 + 1, x1 + 1, val);
}
```

### 区间修改，区间查询

**这个厉害了！**

依然按照一维树状数组-区间修改，区间查询的思路；先看看数学

> 考虑$a[][]$的前缀和与差分数组$diff[][]$的关系：
> 
> $\sum_{y=1}^{Y}\sum_{x=1}^{X}a[y][x]$
> 
> $=\sum_{y=1}^{Y}\sum_{x=1}^{X}\sum_{yy=1}^{y}\sum_{xx=1}^{x}diff[yy][xx]$
> 
> $=\sum_{y=1}^{Y}\sum_{x=1}^{X}diff[y][x]*(Y-y+1)(X-x+1)$
> 
> 因为$(Y-y+1)(X-x+1)=(Y+1)(X+1)-(Y+1)x-(X+1)y+xy$，所以：
>
> $\sum_{y=1}^{Y}\sum_{x=1}^{X}diff[y][x]*(Y-y+1)(X-x+1)$
> 
> $=(Y+1)(X+1)\sum_{y=1}^{Y}\sum_{x=1}^{X}diff[y][x]$
> 
> $-(Y+1)\sum_{y=1}^{Y}\sum_{x=1}^{X}diff[y][x]*x$
> 
> $-(X+1)\sum_{y=1}^{Y}\sum_{x=1}^{X}diff[y][x]*y$
> 
> $+\sum_{y=1}^{Y}\sum_{x=1}^{X}diff[y][x]*yx$

所以，建立四个二维树状数组：

- $diff[y][x]$

- $diff[y][x]*x$

- $diff[y][x]*y$

- $diff[y][x]*yx$

#### 模板

[模板题：二维树状数组3：区间修改，区间查询](https://vjudge.net/problem/LibreOJ-135)

```cpp
template <class DataT>
inline void Modify2D(BinaryIndexedTree2D<DataT> diff[], const size_t &y0,
                     const size_t &x0, const size_t &y1, const size_t &x1,
                     const DataT &val) {
  diff[0].Add(y0, x0, val);
  diff[0].Add(y1 + 1, x0, -val);
  diff[0].Add(y0, x1 + 1, -val);
  diff[0].Add(y1 + 1, x1 + 1, val);

  diff[1].Add(y0, x0, val * x0);
  diff[1].Add(y1 + 1, x0, -val * x0);
  diff[1].Add(y0, x1 + 1, -val * (x1 + 1));
  diff[1].Add(y1 + 1, x1 + 1, val * (x1 + 1));

  diff[2].Add(y0, x0, val * y0);
  diff[2].Add(y1 + 1, x0, -val * (y1 + 1));
  diff[2].Add(y0, x1 + 1, -val * y0);
  diff[2].Add(y1 + 1, x1 + 1, val * (y1 + 1));

  diff[3].Add(y0, x0, val * y0 * x0);
  diff[3].Add(y1 + 1, x0, -val * (y1 + 1) * x0);
  diff[3].Add(y0, x1 + 1, -val * y0 * (x1 + 1));
  diff[3].Add(y1 + 1, x1 + 1, val * (y1 + 1) * (x1 + 1));
}

template <class DataT>
inline DataT Query(const BinaryIndexedTree2D<DataT> diff_tree[],
                   const size_t &y, const size_t &x) {
  return (y + 1) * (x + 1) * diff_tree[0].GetSum(y, x) -
         (y + 1) * diff_tree[1].GetSum(y, x) -
         (x + 1) * diff_tree[2].GetSum(y, x) + diff_tree[3].GetSum(y, x);
}

template <class DataT>
inline DataT Query(const BinaryIndexedTree2D<DataT> diff_tree[],
                   const size_t &y0, const size_t &x0, const size_t &y1,
                   const size_t &x1) {
  return Query(diff_tree, y1, x1) - Query(diff_tree, y0 - 1, x1) -
         Query(diff_tree, y1, x0 - 1) + Query(diff_tree, y0 - 1, x0 - 1);
}
```

完整代码

```cpp
#include <iostream>

inline size_t LowBit(const size_t &x) { return x & -x; }

template <class DataT>
class BinaryIndexedTree2D final {
  DataT **data_;
  size_t size_y_, size_x_;

 public:
  BinaryIndexedTree2D(const size_t &size_y, const size_t &size_x)
      : data_(new DataT *[size_y + 1]), size_y_(size_y), size_x_(size_x) {
    for (size_t y = 0; y <= size_y_; ++y) {
      data_[y] = new DataT[size_x_ + 1];
      for (size_t x = 0; x <= size_x_; ++x) data_[y][x] = 0;
    }
  }

  template <class Arr2DT>
  BinaryIndexedTree2D(const Arr2DT &arr2d, const size_t &size_y,
                      const size_t &size_x)
      : BinaryIndexedTree2D(size_y, size_x) {
    for (size_t y = 1; y <= size_y_; ++y)
      for (size_t x = 1; x <= size_x_; ++x) {
        data_[y][x] += arr2d[y][x];
        size_t &&xx = x + LowBit(x);
        size_t &&yy = y + LowBit(y);

        if (xx <= size_x_) data_[y][xx] += data_[y][x];
        if (yy <= size_y_) data_[yy][x] += data_[y][x];
        if (xx <= size_x_ && yy <= size_y_) data_[yy][xx] += data_[y][x];
      }
  }

  ~BinaryIndexedTree2D() {
    for (size_t y = 0; y <= size_y_; ++y) delete[] data_[y];
    delete[] data_;
  }

  void Add(const size_t &y, const size_t &x, const DataT &val) {
    for (size_t yy = y; yy <= size_y_; yy += LowBit(yy))
      for (size_t xx = x; xx <= size_x_; xx += LowBit(xx)) data_[yy][xx] += val;
  }

  DataT GetSum(const size_t &y, const size_t &x) const {
    DataT res = 0;
    for (size_t yy = y; yy; yy -= LowBit(yy))
      for (size_t xx = x; xx; xx -= LowBit(xx)) res += data_[yy][xx];
    return res;
  }

  DataT GetSum(const size_t &y0, const size_t &x0, const size_t &y1,
               const size_t &x1) const {
    return GetSum(y1, x1) - GetSum(y0 - 1, x1) - GetSum(y1, x0 - 1) +
           GetSum(y0 - 1, x0 - 1);
  }
};

template <class DataT>
inline void Modify2D(BinaryIndexedTree2D<DataT> diff[], const size_t &y0,
                     const size_t &x0, const size_t &y1, const size_t &x1,
                     const DataT &val) {
  diff[0].Add(y0, x0, val);
  diff[0].Add(y1 + 1, x0, -val);
  diff[0].Add(y0, x1 + 1, -val);
  diff[0].Add(y1 + 1, x1 + 1, val);

  diff[1].Add(y0, x0, val * x0);
  diff[1].Add(y1 + 1, x0, -val * x0);
  diff[1].Add(y0, x1 + 1, -val * (x1 + 1));
  diff[1].Add(y1 + 1, x1 + 1, val * (x1 + 1));

  diff[2].Add(y0, x0, val * y0);
  diff[2].Add(y1 + 1, x0, -val * (y1 + 1));
  diff[2].Add(y0, x1 + 1, -val * y0);
  diff[2].Add(y1 + 1, x1 + 1, val * (y1 + 1));

  diff[3].Add(y0, x0, val * y0 * x0);
  diff[3].Add(y1 + 1, x0, -val * (y1 + 1) * x0);
  diff[3].Add(y0, x1 + 1, -val * y0 * (x1 + 1));
  diff[3].Add(y1 + 1, x1 + 1, val * (y1 + 1) * (x1 + 1));
}

template <class DataT>
inline DataT Query(const BinaryIndexedTree2D<DataT> diff_tree[],
                   const size_t &y, const size_t &x) {
  return (y + 1) * (x + 1) * diff_tree[0].GetSum(y, x) -
         (y + 1) * diff_tree[1].GetSum(y, x) -
         (x + 1) * diff_tree[2].GetSum(y, x) + diff_tree[3].GetSum(y, x);
}

template <class DataT>
inline DataT Query(const BinaryIndexedTree2D<DataT> diff_tree[],
                   const size_t &y0, const size_t &x0, const size_t &y1,
                   const size_t &x1) {
  return Query(diff_tree, y1, x1) - Query(diff_tree, y0 - 1, x1) -
         Query(diff_tree, y1, x0 - 1) + Query(diff_tree, y0 - 1, x0 - 1);
}

int n, m;

int main() {
  std::ios::sync_with_stdio(0);
  std::cin.tie(0), std::cout.tie(0);

  std::cin >> n >> m;

  BinaryIndexedTree2D<int64_t> diff[4] = {
      BinaryIndexedTree2D<int64_t>(n, m), BinaryIndexedTree2D<int64_t>(n, m),
      BinaryIndexedTree2D<int64_t>(n, m), BinaryIndexedTree2D<int64_t>(n, m)};

  for (int t, x, y, u, v, w; std::cin >> t >> x >> y >> u >> v;) {
    if (t == 1) {
      std::cin >> w;
      Modify2D<int64_t>(diff, x, y, u, v, w);
    } else {
      std::cout << Query(diff, x, y, u, v) << "\n";
    }
  }

  return 0;
}
```
