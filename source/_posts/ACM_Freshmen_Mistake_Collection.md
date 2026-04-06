---
title: ACM Freshmen Mistake Collection
mathjax: false
date: 2020-11-27 15:30:37
categories: Algorithm
description: ACM新生赛错题本
---

<div align="middle"><iframe frameborder="no" border="0" marginwidth="0" marginheight="0" width=330 height=86 src="//music.163.com/outchain/player?type=2&id=551339691&auto=1&height=66"></iframe></div>

## 做梦

### 题目

#### Problem Description

忙碌了一天的lzh疲惫地躺在床上，但还念念不忘白天的习题，渐渐地，他的视野逐渐模糊，随之而来的是另一幅场景......

在一片祥和的森林里，有一群头顶为黑色菌盖(菌盖就是像顶帽子的那部分)或者白色菌盖的蘑菇人，它们是lzh梦中的生物，其中黑色的蘑菇人有3个、白色的蘑菇人有5个，它们除了头顶的蘑菇颜色不同外，没有什么不同，甚至连思维方式都是一样的！它们通过视觉可以看到除自己以外的其它所有蘑菇人的颜色，现在它们想按颜色分类，白色的蘑菇人为一堆，黑色的蘑菇人为另一堆，但是它们不会说话，也看不到自己的颜色，所以现在它们散乱的坐在一块草坪上，仿佛在思考着什么。

不知过了多久，处于上帝视角的lzh随口说了句“至少有1个黑色的”，突然蘑菇人们都站了起来，第1分钟过去了，所有蘑菇人没有什么行动......第2分钟过去了，蘑菇人还是没有什么行动......第3分钟后，3个黑色的蘑菇人突然想懂了什么，一蹦一跳地跳出了草坪，坐在了一棵树桩下。作为梦的主人的lzh，也突然明白了这些蘑菇人的行为，因为刚刚谈论的是黑色的蘑菇人，所以当黑色的蘑菇人意识到自己是黑色的时候，就会离开草坪，而蘑菇人思考得很慢，1分钟才能想懂一个问题，但是它们的换位思考能力和逻辑推理能力很不错，并且lzh相信蘑菇人十分聪明，不会相信毫无逻辑的话。

又不知过了多久，清晨7点的闹钟像往常一样将lzh唤醒，lzh对这个梦还意犹未尽，觉得这个梦真的是太有趣了，所以他将这个梦分享给你，并笑着问你如果总共有a个蘑菇人，其中有b个白色蘑菇人，蘑菇人们听到的是“至少有c个黑色蘑菇人”，那么至少在多少分钟后蘑菇人能分成黑白两部分（草坪内的为一部分，草坪外的为另一部分，黑白蘑菇人不在同一部分时则认为是分成黑白两部分了）。

#### Input

输入只有一行，输入3个整数，依次为a，b，c。(1≤a≤100),(0≤b≤a),(1≤c≤1000)。
表示有a个蘑菇人，其中有b个白色蘑菇人，蘑菇人们听到的是“至少有c个黑色蘑菇人”。

#### Output

如果你认为在lzh说话之前，蘑菇人早已经分成两部分，那么输出整数0;
如果你认为在lzh说完话后，蘑菇人仍然不能分成两部分，那么输出lzh的一小段记忆片段“0x3f3f3f3f......”（不包括引号，但是包括省略号）;
如果你认为在lzh说完话后的ans分钟后，蘑菇人才能分成两部分，输出ans。

#### Sample Input

1 1 1
2 1 1000
8 5 1
3 1 1

#### Sample Output

0
0x3f3f3f3f......
3
2

#### Hint

对于a=3,b=1,num=1的情况，也就是只有1个白色蘑菇人，2个黑色蘑菇人的情况，它们听到“至少有1个黑色蘑菇人”。
它们的心理活动如下：
在听到lzh说话之前，
A(黑)：abab......
B(黑)：abab......
C(白)：abab......
在听到lzh说话之后，
A(黑)：我要走吗？
B(黑)：我要走吗？
C(白)：我要走吗？
第一分钟后，
A(黑)：有1个黑色的，它应该走的啊，但是它怎么不走啊？
B(黑)：有1个黑色的，它应该走的啊，但是它怎么不走啊？
C(白)：有2个黑色的，它们应该会在下一分钟后走。
第二分钟后，
A(黑)：哦，我也是黑色的，溜了溜了。
B(黑)：哦，我也是黑色的，溜了溜了。
C(白)：(~O~)，继续睡觉吧！

### 错点

- 5小时实在是累得，没看到**黑蘑菇是从草坪上离开，结果判了`if (a-b == b == 1) ans = 0`**
- 蘑菇们都长着眼睛，那“至少有c个蘑菇”这个信息其实是没有用的；因为**蘑菇们看到的是a-b或a-b-1个黑蘑菇**
- 所以在解题时，没有用到c，猜数据直接输出了a-b。。导致AK失败

### 题解

#### 首先处理两个特殊情况：

- 当蘑菇人只有一种颜色时，`ans=0`（**注意，题意是黑蘑菇离开草坪；当`黑=白=1`时，他们都在草坪上而没有分开**）
- 当`c`大于`黑色的数量`时，输出`0x3f3f3f3f......`

####  然后是错点

- **蘑菇们看到的是a-b或a-b-1个黑蘑菇**，***但其实，谁也无法确定究竟是多少，也就是薛定谔的猫***
- 所有蘑菇都很懒，它们都会优先认为自己是白的
- 一个白/黑的蘑菇看到了n个黑蘑菇，那么它会认为这些黑蘑菇只看到了n-1个黑蘑菇
- 同时，它将这个想法套用到它看到的黑蘑菇身上，也就是说，它认为这些黑蘑菇认为它的同伴们看到了n-2个
- 这就是一个递归，而在没有c时，结束条件是0：最后有一个蘑菇**被认为**看到0个黑蘑菇，这样大家都认为自己是白的。
- 而当至少有c个黑蘑菇时，递归结束条件是c：有蘑菇**被认为看到了c-1个黑蘑菇**，它一定会发现自己是黑的。

#### c=1的情况

- 只有1个时，那么第一分钟会有1个唯一的黑蘑菇反应，因为这个黑蘑菇只能看到白的
- 而如果它没有反应，只看到1个黑蘑菇的就会有反应：这个黑的没反应，是因为它觉得我应该有反应，那么我是黑的
- 而如果2个黑蘑菇都没反应：是因为看到2个黑蘑菇的蘑菇希望他们在第2分钟反应。说明在他们以外也看到了2个黑的，那么我是黑的
- 而3个、4个、N个同理。

最后是c>1，其实就是加速了进程：从有c个蘑菇开始上面的递推。

**最后得出结果：`ans=a-b-c+1`**

### AC代码

```cpp
#include <iostream>

int main() {
  std::ios::sync_with_stdio(0);
  std::cin.tie(0), std::cout.tie(0);

  for (int a, b, c; std::cin >> a >> b >> c; std::cout << "\n") {
    if (b == 0 || a - b == 0) {
      std::cout << "0";
    } else if (a - b < c) {
      std::cout << "0x3f3f3f3f......";
    } else {
      std::cout << a - b - c + 1;
    }
  }

  return 0;
}
```

## 合唱队形

### 题目

#### Description

N位同学站成一排，音乐老师要请其中的(N-K)位同学出列，使得剩下的K位同学不交换位置就能排成合唱队形。
合唱队形是指这样的一种队形：设K位同学从左到右依次编号为1, 2, …, K，他们的身高分别为T1, T2, …, TK，则他们的身高满足T1 < T2 < … < Ti , Ti > Ti+1 > … > TK (1 <= i <= K)。
你的任务是，已知所有N位同学的身高，计算最少需要几位同学出列，可以使得剩下的同学排成合唱队形。

#### Input

输入的第一行是一个整数N（2 <= N <= 100），表示同学的总数。
第一行有n个整数，用空格分隔，第i个整数Ti（130 <= Ti <= 230）是第i位同学的身高（厘米）。

#### Output

可能包括多组测试数据，对于每组数据，
输出包括一行，这一行只包含一个整数，就是最少需要几位同学出列。

#### Sample Input 1 

3
174 208 219 
6
145 206 193 171 187 167 
0

#### Sample Output 1

0
1

### 错点

好好人想什么区间DP呢。。。还是做过的原题。。

### 题解

{% post_link Longest_Monotone_Subsequence 最长单调子序列 %}

- 队形呈现一个峰
- 考虑身高严单增时：那么就是求最长严单增子序列
- 考虑身高严单减时：将其反序求最长严单增子序列即可
- 最后合并两者，取出最大值

### AC代码

```cpp
#include <iostream>

int32_t n, a[2][100];
int32_t dp[2][100];

void LongestIncreasingSubsequence(const int32_t *a, const size_t &n, int32_t *dp) {
  for (int i = 0; i < n; ++i) {
    dp[i] = 1;
    for (int j = 0; j < i; ++j)
      if (a[j] < a[i] && dp[i] <= dp[j]) dp[i] = dp[j] + 1;
  }
}

int main() {
  std::ios::sync_with_stdio(0);
  std::cin.tie(0), std::cout.tie(0);

  std::cin >> n;
  for (int i = 0; i < n; ++i) { 
    std::cin >> a[0][i];
    a[1][n - 1 - i] = a[0][i];
  }

  LongestIncreasingSubsequence(a[0], n, dp[0]);
  LongestIncreasingSubsequence(a[1], n, dp[1]);

  int32_t res = 0;
  for (int i = 0; i < n; ++i) res = std::max(res, dp[0][i] + dp[1][n - 1 - i]);

  std::cout << n - res + 1 << "\n";

  return 0;
}
```

## 建设电力系统

### 题目

#### Description

小明所在的城市由于下暴雪的原因，电力系统严重受损。许多电力线路被破坏，因此许多村庄与主电网失去了联系。政府想尽快重建电力系统，所以，身为程序员的你被赋予了一项任务，就是编程计算重建电力系统的最少花费，重建的电力系统必须保证任意两个村庄之间至少存在一条通路。

#### Input

输入的第一行为一个整数T（1<=T<=50），表示有T组测试数据。

每组输入第一行是两个正整数N，E（2<=N<=500，N<=E<=N*(N-1)/2），分别表示村庄的个数和原有电力线路的个数。

接下来的E行，每行包含三个整数A，B，K（0<=A,B<N，0<=K<1000）。A和B分别表示电力线路的起始村庄代号。如果K=0，表示这条线路仍然正常。如果K是一个正整数，表示这条线路需要花费K的代价来重建。

题目保证输入中没有重边，也没有起始村庄相同的边。

#### Output

对于每组输入，输出重建电力系统所需的最小花费，以此来保证任意两个村庄之间至少存在一条通路。

#### Sample Input 1 

1
3 3
0 1 5
0 2 0
1 2 9

#### Sample Output 1

5

### 错点

看出来是最小生成树了，但是忘了怎么写

### 题解

最小生成树的模板题

{% post_link Minimum_Spanning_Tree 最小生成树 %}

### AC代码

```cpp
#include <algorithm>
#include <iostream>

const int kN = 500, kM = kN * (kN - 1) / 2;
int T, N, E;
int fa[kN];

struct Edge {
  int src, dst, val;
  bool operator<(const Edge &other) const { return val < other.val; }
} edges[kM];

int Find(const int &x) { return x == fa[x] ? x : fa[x] = Find(fa[x]); }
void Merge(const int &x, const int &y) { fa[Find(x)] = fa[Find(y)]; }

int Kruskal() {
  int res = 0;
  std::sort(edges, edges + E);

  for (int i = 0; i < E; ++i) {
    int32_t src = Find(edges[i].src), dst = Find(edges[i].dst);
    if (src == dst) continue;
    Merge(src, dst);
    res += edges[i].val;
  }
  return res;
}

int main() {
  std::ios::sync_with_stdio(0);
  std::cin.tie(0), std::cout.tie(0);

  for (std::cin >> T; T--;) {
    std::cin >> N >> E;
    for (int i = 0; i < E; ++i)
      std::cin >> edges[i].src >> edges[i].dst >> edges[i].val;

    for (int i = 0; i < N; ++i) fa[i] = i;
    std::cout << Kruskal() << "\n";
  }

  return 0;
}
```
