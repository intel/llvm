// XFAIL: cuda
// piextUSM*Alloc functions for CUDA are not behaving as described in
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/cl_intel_unified_shared_memory.asciidoc
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

//==------------------- dmemll.cpp - Device Memory Linked List test --------==//
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

  if (!dev.get_info<info::device::usm_device_allocations>())
    return 0;

  Node h_cur;

  Node *d_head = (Node *)malloc_device(sizeof(Node), dev, ctxt);
  if (d_head == nullptr) {
    return -1;
  }
  Node *d_cur = d_head;

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
    free(d_cur, ctxt);

    const int want = i * 4 + 1;
    if (h_cur.Num != want) {
      return -2;
    }
    d_cur = h_cur.pNext;
  }

  return 0;
}
