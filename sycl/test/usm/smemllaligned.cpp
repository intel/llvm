// RUN: %clangxx -fsycl %s -o %t1.out -lOpenCL
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows

//==---- smemllaligned.cpp - Aligned Shared Memory Linked List test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl::sycl;

int numNodes = 4;

struct Node {
  Node() : pNext(nullptr), Num(0xDEADBEEF) {}

  Node *pNext;
  uint32_t Num;
};

class foo;
int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();
  Node *s_head = nullptr;
  Node *s_cur = nullptr;

  s_head = (Node *)aligned_alloc_shared(alignof(Node), sizeof(Node), dev, ctxt);
  if (s_head == nullptr) {
    return -1;
  }
  s_cur = s_head;

  for (int i = 0; i < numNodes; i++) {
    s_cur->Num = i * 2;

    if (i != (numNodes - 1)) {
      s_cur->pNext =
          (Node *)aligned_alloc_shared(alignof(Node), sizeof(Node), dev, ctxt);
      if (s_cur->pNext == nullptr) {
        return -1;
      }
    } else {
      s_cur->pNext = nullptr;
    }

    s_cur = s_cur->pNext;
  }

  auto e1 = q.submit([=](handler &cgh) {
    cgh.single_task<class foo>([=]() {
      Node *pHead = s_head;
      while (pHead) {
        pHead->Num = pHead->Num * 2 + 1;
        pHead = pHead->pNext;
      }
    });
  });

  e1.wait();

  s_cur = s_head;
  int mismatches = 0;
  for (int i = 0; i < numNodes; i++) {
    const int want = i * 4 + 1;
    if (s_cur->Num != want) {
      return -1;
    }
    Node *old = s_cur;
    s_cur = s_cur->pNext;
    free(old, ctxt);
  }

  return 0;
}
