// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
//
//==----------- ordered_dmemll.cpp - Device Memory Linked List test --------==//
// It uses an ordered queue where explicit waiting is not necessary between
// kernels
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr int numNodes = 4;

struct Node {
  Node() : pNext(nullptr), Num(0xDEADBEEF) {}

  Node *pNext;
  uint32_t Num;
};

class foo;
int main() {
  queue q{property::queue::in_order()};
  auto dev = q.get_device();
  auto ctxt = q.get_context();
  Node *d_head = nullptr;
  Node *d_cur = nullptr;
  Node h_cur;

  d_head = (Node *)malloc_device(sizeof(Node), dev, ctxt);
  if (d_head == nullptr) {
    return -1;
  }
  d_cur = d_head;

  for (int i = 0; i < numNodes; i++) {
    h_cur.Num = i * 2;

    if (i != (numNodes - 1)) {
      h_cur.pNext = (Node *)malloc_device(sizeof(Node), dev, ctxt);
      if (h_cur.pNext == nullptr) {
        return -1;
      }
    } else {
      h_cur.pNext = nullptr;
    }

    event e0 = q.memcpy(d_cur, &h_cur, sizeof(Node));
    e0.wait();

    d_cur = h_cur.pNext;
  }

  q.submit([=](handler &cgh) {
    cgh.single_task<class foo>([=]() {
      Node *pHead = d_head;
      while (pHead) {
        pHead->Num = pHead->Num * 2 + 1;
        pHead = pHead->pNext;
      }
    });
  });

  q.submit([=](handler &cgh) {
    cgh.single_task<class bar>([=]() {
      Node *pHead = d_head;
      while (pHead) {
        pHead->Num = pHead->Num + 42;
        pHead = pHead->pNext;
      }
    });
  });

  d_cur = d_head;
  for (int i = 0; i < numNodes; i++) {
    event c = q.memcpy(&h_cur, d_cur, sizeof(Node));
    c.wait();
    free(d_cur, ctxt);

    const int want = i * 4 + 43;
    if (h_cur.Num != want) {
      std::cout << "Result mismatches " << h_cur.Num << " vs expected "
                << i * 4 + 43 << " for index " << i << std::endl;
      return -1;
    }
    d_cur = h_cur.pNext;
  }

  return 0;
}
