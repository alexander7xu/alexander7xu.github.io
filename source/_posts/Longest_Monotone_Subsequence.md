---
title: Longest Monotone Subsequence
mathjax: false
date: 2020-11-27 15:09:39
categories: Algorithm
description: 最长单调子序列的动态规划/贪心+二分查找算法
---

## 严格单调递增子序列

求`a[n]`的最长严格单调递增子序列长度，简称LMS

## 思路1：动态规划

考虑`dp[i]=以a[i]结尾的最长LMS`

- 首先，对于`a[1]`，最长严单增子序显然是它本身，也就是长度为`dp[1]=1`
- 对于i>1，考虑`a[i]`与`a[j], 1 <= j < i`
  - 如果`a[j] < a[i]`，那么显然`a[j]`结尾的最长严单增子序，尾部再加上一个`a[i]`，仍然严单增
  - 如果`a[j] >= a[i]`，那么反之，`a[i]`不能与`a[j]`结尾的子序列构成最长严单增子序
- 则有dp方程：`dp[i] = Max{dp[i], dp[j] + 1 if a[j] < a[i] else 1}`

时间复杂度：`O(n^2)`

空间复杂度：`O(n)`

### 模板

```cpp
// cmp： < 时为最长严单增子序列； <= 时为最长单增子序列； 反之为相应的递减
template <class Compare>
int32_t LongestMonotoneSubsequence(const int32_t *a, const size_t &n,
                                   const Compare &cmp) {
  if (n == 0 || a == nullptr) return 0;

  int32_t res = 1, *dp = new int32_t[n];

  for (int i = 0; i < n; ++i) {
    dp[i] = 1;
    for (int j = 0; j < i; ++j)
      if (cmp(a[j], a[i]) && dp[i] <= dp[j]) dp[i] = dp[j] + 1;
    if (res < dp[i]) res = dp[i];
  }

  delete[] dp;
  return res;
}
```

## 思路2：贪心+二分查找

- 考虑计算`dp[i]`时：
- 有`x, y < i`，且`a[x] < a[y] < a[i]`，且`dp[x] = dp[y]`，那么选择`x`显然比选择`y`更好
- 所以可以使用贪心：维护一个数组`b[]`，使得`b[j], 1<=j<=t`为`a[i], 1<=i<=n`中长度为`j`的最长单调子序列中尾元素的最小值
- 那么，当`a[i] > b[t]`时，显然`dp[i] = b[t]+1`，同时要令`b[++t] = a[i]`
- 反之，从`b[j], 1<=j<=t`中找出第一个满足`b[j]>=a[i]`的，令`b[j] = a[i]`
- 可以发现最终`dp[n] = t`，故可省去`dp[i]`
- 可以发现`b[]`是满足严单增的，故`a[i] <= b[t]`时可用二分法查找

时间复杂度：`O(nlogn)`

空间复杂度：`O(n)`

### 模板

```cpp
// cmp： < 时为第一个大于key的； <= 时为最后一个等于或第一个大于； 反之为相应的
template <class Compare>
int32_t BinarySearch(const int32_t *a, const size_t &n, const int32_t &key,
              const Compare &cmp) {
  size_t lp = 0, rp = n - 1, mp;
  if (cmp(a[rp], key)) return n;

  while (lp != rp) {
    mp = (lp + rp) >> 1;
    if (cmp(a[mp], key)) {
      lp = mp + 1;
    } else {
      rp = mp;
    }
  }
  return lp;
}

// cmp： < 时为最长严单增子序列； <= 时为最长单增子序列； 反之为相应的递减
template <class Compare>
int32_t LongestMonotoneSubsequence(const int32_t *a, const size_t &n,
                                   const Compare &cmp) {
  if (n == 0) return 0;

  int32_t res = 0, *b = new int32_t[n];
  b[0] = a[0];

  for (int i = 1; i < n; ++i) {
    if (cmp(b[res], a[i])) {
      b[++res] = a[i];
    } else {
      b[BinarySearch(b, res, a[i], cmp)] = a[i];
    }
  }

  delete[] b;
  return res + 1;
}
```
