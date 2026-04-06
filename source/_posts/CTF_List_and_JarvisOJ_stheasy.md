---
title: CTF List and JarvisOJ:stheasy
mathjax: false
date: 2020-11-26 20:55:22
categories: Course Notes
description: 2020-11-27下午CTF课上的小实验
---

## 实验内容

- 数据结构：将两个升序链表合并为一个新的升序链表
- 逆向：JarvisOJ:stheasy

## 数据结构：将两个升序链表合并为一个新的升序链表

### 什么是链表

{% post_link CTF_Tictactoe_Game_CE_Hacking 参见我的上一篇文章 %}

#### 链表结构

- 一个节点有且仅有两个属性：它存储的值，它的下一个节点（还可以有它的上一个节点，但在本题没必要）

#### 链表的功能

结构决定功能
- 提供高效（O(1)）的任意位置插入与删除（显然，执行这两个操作时只需要简单的改变上/下一个结点指向的位置）
- 遍历访问而不可随机访问（对于任意的n，你无法直接确定第n个节点到底是哪一个，除非从头开始遍历）

### 源代码

[Leetcode题目](https://leetcode.com/problems/merge-two-sorted-lists/)

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct ListNode {
  int32_t val;            // 当前节点的值
  struct ListNode *next;  // 当前节点的下一个节点
};

// 创建一个有且仅有一个节点的链表
struct ListNode *NewListNode(int32_t val) {
  struct ListNode *node = calloc(1, sizeof(struct ListNode));
  node->val = val;
  node->next = NULL;
  return node;
}

// 在链表中插入一个新节点，返回新节点的地址
struct ListNode *InsetrtListNode(int32_t val, struct ListNode *after) {
  if (after == NULL) return NULL;  // Unexpected
  struct ListNode *node = NewListNode(val);
  node->next = after->next;
  after->next = node;
  return node;
}

// 释放链表占用的内存
void FreeList(struct ListNode *head) {
  for (struct ListNode *next; head != NULL; head = next) {
    next = head->next;
    free(head);
  }
}

// 从stdin读取一个链表
struct ListNode *ReadList() {
  struct ListNode *head = NULL, *tail = head;
  int32_t val;

  do {
    scanf("%d", &val);                    // 输入一个整数
    if (tail == NULL) {                   // 还没有链表
      tail = head = NewListNode(val);     // 创建一个链表
    } else {                              // 有链表了
      tail = InsetrtListNode(val, tail);  // 插入到尾部之后
    }
  } while (getchar() != '\n' && getchar() != '\n');
  // 读取两个字符来忽略掉"->"
  // 当读取到换行符（"\n"(AKA: LF)或"\r\n"(AKA: LRLF)）时，一个链表结束
  return head;
}

void PrintList(struct ListNode *head) {
  if (head == NULL) return;
  printf("%d", head->val);
  for (head = head->next; head != NULL; head = head->next) printf("->%d", head->val);
  printf("\n");
}

// 合并两个已排序的链表
struct ListNode *Merge(struct ListNode *a, struct ListNode *b) {
  // 直接创建一个链表方便操作，此时链表头是无用的0
  struct ListNode *head = NewListNode(0), *tail = head;

  while (a != NULL || b != NULL) {
    // 当b为空时，从a取值；当a为空时，从b取值；两个都非空时，取值较小者
    if (b == NULL || a != NULL && a->val < b->val) {
      tail = InsetrtListNode(a->val, tail);
      a = a->next;
    } else {
      tail = InsetrtListNode(b->val, tail);
      b = b->next;
    }
  }

  tail = head->next;  // 删掉刚刚无用的链表头
  free(head);
  return tail;
}

int main() {
  struct ListNode *a = ReadList();
  struct ListNode *b = ReadList();
  struct ListNode *res = Merge(a, b);

  PrintList(res);

  FreeList(a);
  FreeList(b);
  FreeList(res);

  return 0;
}
```

### Python对比

```python
# 抽象内存，有效位置从1开始
Memory = ["Invalid"]


# C中的 node->val，node->next，就表示为 Memory[node][0]，Memory[node][1]
# 使用一个三元素的列表来表示链表节点
# 列表[0]：节点的值
# 列表[1]：为下一个节点在Memory的位置
def NewListNode(val):           # 创建一个链表节点
    address = len(Memory)       # 内存中的位置，即C中的指针值
    Memory.append([val, 0])     # 内存中加入一个新的列表节点
    return address


# val：节点的值
# after：另外一个节点在Memory的位置（0为无效位置）（即指针）
def InsertListNode(val, after):
    if after == 0:      # 无效位置
        return 0
    new_node_address = NewListNode(val)                 # struct ListNode *node = NewListNode(val);

    Memory[new_node_address][1] = Memory[after][1]      # node->next = after->next;
    Memory[after][1] = new_node_address                 # after->next = node;
    return new_node_address


def FreeList(head):
    while head != 0:
        next = Memory[head][1]
        Memory[head] = []
        head = next


def ReadList():
    head = tail = None
    while True:
        val = input()           # 输入一个整数
        if (val == "End"):      # 输入End结束输入
            break
        if (tail == None):
            tail = head = NewListNode(int(val))
        else:
            tail = InsertListNode(int(val), tail)
    return head


def PrintList(head):
    print(Memory[head][0], end="")
    head = Memory[head][1]

    while head != 0:
        print("->", end=str(Memory[head][0]))
        head = Memory[head][1]
    print("")


def Merge(a, b):
    head = tail = NewListNode(0)    # 一个无用的0节点，简化代码

    while (a != 0 or b != 0):       # a = 0 代表着 a列表的结尾
        if b == 0 or a != 0 and Memory[a][0] < Memory[b][0]:
            tail = InsertListNode(Memory[a][0], tail)
            a = Memory[a][1]
        else:
            tail = InsertListNode(Memory[b][0], tail)
            b = Memory[b][1]

    tail = Memory[head][1]          # 删掉无用的0节点
    Memory[head] = []               # free(head)
    return tail


def main():
    a = ReadList()
    b = ReadList()
    res = Merge(a, b)

    PrintList(res)

    FreeList(a)
    FreeList(b)
    FreeList(res)
    return 0


if __name__ == "__main__":
    main()
```
