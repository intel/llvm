// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

//==------------------- hmemll.cpp - Host Memory Linked List test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

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
  auto ctxt = q.get_context();
  auto dev = q.get_device();

  if (!dev.get_info<info::device::usm_host_allocations>())
    return 0;

  Node *h_head = (Node *)malloc_host(sizeof(Node), ctxt);
  if (h_head == nullptr) {
    return -1;
  }
  Node *h_cur = h_head;

  for (int i = 0; i < numNodes; i++) {
    h_cur->Num = i * 2;

    if (i != (numNodes - 1)) {
      h_cur->pNext = (Node *)malloc_host(sizeof(Node), ctxt);
      if (h_cur->pNext == nullptr) {
        return -1;
      }
    } else {
      h_cur->pNext = nullptr;
    }

    h_cur = h_cur->pNext;
  }

  auto e1 = q.submit([=](handler &cgh) {
    cgh.single_task<class foo>([=]() {
      Node *pHead = h_head;
      while (pHead) {
        pHead->Num = pHead->Num * 2 + 1;
        pHead = pHead->pNext;
      }
    });
  });

  e1.wait();

  h_cur = h_head;
  for (int i = 0; i < numNodes; i++) {
    const int want = i * 4 + 1;
    if (h_cur->Num != want) {
      return -2;
    }
    Node *old = h_cur;
    h_cur = h_cur->pNext;
    free(old, ctxt);
  }

  return 0;
}
