---
title: From Floyd Algorithm to CKY Parsing
mathjax: true
date: 2025-12-05 23:38:39
categories: Algorithm
description: Something popped into my brain in the CoLi lecture. Analysis of these two similar algorithms.
---

## Floyd Algorithm

### Transitive Closure

Wikipedia: [Transitive closure](https://en.wikipedia.org/wiki/Transitive_closure)

> If $X$ is a set of airports and $x R y$ means "there is a direct flight from airport $x$ to airport $y$" (for $x$ and $y$ in $X$), then the transitive closure of $R$ on $X$ is the relation $R^+$ such that $x R^+ y$ means "it is possible to fly from $x$ to $y$ in one or more flights".
>
> The problem can also be solved by the Floyd–Warshall algorithm in $O(n^3)$.

Consider a transit airport $z$, if there is a flight from $x$ to $z$ and another one from $z$ to $y$, then $y$ is reachable from $x$.

Define the state $T[k][i][j]$, which represents whether there is a path from $i$ to $j$ restricted to intermediate nodes from the set $\{0...k\}$. When computing $T[k][][]$, there are only two possible scenarios:
- $j$ is reachable from $i$ without using $k$ as an intermediate node.
- $j$ is reachable from $i$ using $k$ as an intermediate node.

Formula: $T[k][i][j] = T[k-1][i][j] \lor (T[k-1][i][k] \land T[k-1][k][j])$

```python
adj_mat: Bool[T, "i=n j=n"] # input
transit: Bool[T, "i=n j=n"] # zero initialized

transit[:, :] = adj_mat[:, :]
for i in range(transit.shape[-1]):
    transit[i, i] = True

for k in range(transit.shape[0]):
    for i in range(transit.shape[0]):
        for j in range(transit.shape[0]):
            transit[i, j] |= transit[i, k] & transit[k, j]

return transit
```

Example question: [LeetCode 1462. Course Schedule IV](https://leetcode.com/problems/course-schedule-iv/)

### Multiple-Source Shortest Paths

Now, we not only want to ask whether it is possible to fly from $x$ to $y$, but we also want to find the shortest path between them.

Consider an intermediate airport $z$, if the sum of distances from $x$ to $z$ and $z$ to $y$ is strictly less than the currently known shortest path from $x$ to $y$, then the optimal path must route through $z$.

Formula: $T[k][i][j] = \text{min}(T[k-1][i][j], (T[k-1][i][k] + T[k-1][k][j]))$

```python
adj_mat: Float[T, "i=n j=n"]    # input, all positive, unreachable paths are weighted by INF
transit: Float[T, "i=n j=n"]    # INF initialized

transit[:, :] = adj_mat[:, :]
for i in range(transit.shape[-1]):
    transit[i, i] = 0.0

for k in range(transit.shape[0]):
    for i in range(transit.shape[0]):
        for j in range(transit.shape[0]):
            transit[i, j] = min(transit[i, j], transit[i, k] + transit[k, j])

return transit
```

Example question: [LeetCode 1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/)

## CKY Parsing

Wikipedia: [CYK algorithm](https://en.wikipedia.org/wiki/CYK_algorithm)

Given a string $w$ and context-free grammar $G$, the CKY algorithm is used to determine whether $w$ is grammatically valid and to construct its parse tree.

The context-free grammar $G$ (assumed to be in Chomsky Normal Form) contains two kinds of rules $A \rightarrow BC$ and $A \rightarrow t$. To generate $BC$ from $A$, there must exist a **split point $k$** such that $B$ yields the substring $w_{\{i...k\}}$ and $C$ yields the substring $w_{\{k...j\}}$. This logic is highly analogous to the intermediate node concept in the Floyd algorithm.

$T[i][j]=\bigcup_{k} \\{ A \mid A \rightarrow BC,\ B \in T[i][k],\ C \in T[k][j] \\}$

Note that $T[i][j]$ here represents the set of nonterminal symbols capable of generating the substring spanning from index $i$ to $j$. Consequently, all parsing results for strictly smaller constituent substrings must be computed beforehand. This dependency directly dictates the distinct loop execution order in the CKY algorithm.

### Coding

There are several variants of CKY algorithm tailored to specific tasks, such as recognition, tree counting, CFG parsing, and PCFG parsing. Since all these variants share the same core logic, they can be elegantly implemented by simply swapping out the underlying `Chart` class.

```python
def _cky_one_sentence(self, sentence: list[str], chart: ChartBase) -> None:
    """
    Main logic of CKY algorithm.
    """
    # Leaf records are initialized by the Chart class.
    # See ChartBase.__init__

    # for each width b from 2 to n:
    for length in range(1, len(sentence)):
        # for each start position i from 1 to n-b+1:
        for left in range(0, len(sentence) - length):
            right = left + length
            # for each left width k from 1 to b-1:
            for mid in range(left, right):
                self._reduce(chart, left, right, mid)

def _reduce(self, chart: ChartBase, left: int, right: int, mid: int) -> None:
    """
    Try to reduce (left, mid) (mid+1, right) -> (left, right) with all possible rules
    """
    # for each key B in Chart(i,i+k) and key C in Chart(i+k,i+b):
    for left_nt, right_nt in product(
        chart.get(left, mid).keys(), chart.get(mid + 1, right).keys()
    ):
        # for each production rule A -> B C:
        for pa_nt in self._inv_nonterminal_production.get((left_nt, right_nt), ()):
            # Chart(i,i+b).reduce(key=A, left=(i,i+k,B), right=(i+k,i+b,C))
            chart.reduce(left, right, mid, left_nt, right_nt, pa_nt)
```

All of the above algorithms have the same time complexity of $O(n^3)$.

The moment I first saw the CKY algorithm in the CoLi lecture, the Floyd algorithm popped into my brain! It's really an interesting callback!

[Assignment: CKY parsing](https://github.com/coli-saar/cl/wiki/Assignment:-CKY-parsing)
