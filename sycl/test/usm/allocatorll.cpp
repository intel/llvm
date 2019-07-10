// RUN: %clang -std=c++11 -fsycl %s -o %t1.out -lstdc++ -lOpenCL -lsycl
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
//==---- allocatorll.cpp - Device Memory Linked List Allocator test --------==//
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
  bool failed = false;

  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  usm_allocator<Node, usm::alloc::device> alloc(&ctxt, &dev);

  Node *d_head = nullptr;
  Node *d_cur = nullptr;
  Node h_cur;

  for (int i = 0; i < numNodes; i++) {
    if (i == 0) {
      d_head = alloc.allocate(1);
      if (d_head == nullptr) {
        failed = true;
        break;
      }
      d_cur = d_head;
    }

    h_cur.Num = i * 2;

    if (i != (numNodes - 1)) {
      h_cur.pNext = alloc.allocate(1);
      if (h_cur.pNext == nullptr) {
        failed = true;
        break;
      }
    } else {
      h_cur.pNext = nullptr;
    }

    event e0 = q.memcpy(d_cur, &h_cur, sizeof(Node));
    e0.wait();

    d_cur = h_cur.pNext;
  }

  if (!failed) {
    auto e1 = q.submit([=](handler &cgh) {
      cgh.single_task<class foo>([=]() {
        Node *pHead = d_head;
        while (pHead) {
          pHead->Num = pHead->Num * 2 + 1;
          pHead = pHead->pNext;
        }
      });
    });

    e1.wait();

    d_cur = d_head;
    for (int i = 0; i < numNodes; i++) {
      event c = q.memcpy(&h_cur, d_cur, sizeof(Node));
      c.wait();
      alloc.deallocate(d_cur, 1);

      const int want = i * 4 + 1;
      if (h_cur.Num != want) {
        failed = true;
      }
      d_cur = h_cur.pNext;
    }
  }

  return failed;
}
