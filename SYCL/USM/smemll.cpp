// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

//==------------------- smemll.cpp - Shared Memory Linked List test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

using namespace sycl;

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

  if (!dev.get_info<info::device::usm_shared_allocations>())
    return 0;

  Node *s_head = (Node *)malloc_shared(sizeof(Node), dev, ctxt);
  if (s_head == nullptr) {
    return -1;
  }
  Node *s_cur = s_head;

  for (int i = 0; i < numNodes; i++) {
    s_cur->Num = i * 2;

    if (i != (numNodes - 1)) {
      s_cur->pNext = (Node *)malloc_shared(sizeof(Node), dev, ctxt);
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
  for (int i = 0; i < numNodes; i++) {
    const int want = i * 4 + 1;
    if (s_cur->Num != want) {
      return -2;
    }
    Node *old = s_cur;
    s_cur = s_cur->pNext;
    free(old, ctxt);
  }

  return 0;
}
