// RUN: env ENABLE_INFER_AS=1 %clang -std=c++11 -fsycl %s -o %t1.out -lstdc++ -lOpenCL -lsycl -DINTEL_USM
// RUN: env ENABLE_INFER_AS=1 %CPU_RUN_PLACEHOLDER %t1.out
//==---- hmemllaligned.cpp - Aligned Host Memory Linked List test ----------==//
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
    Node() :
        pNext( nullptr ),
        Num( 0xDEADBEEF ) {}

    Node*   pNext;
    uint32_t Num;
};

class foo;
int main() {
  bool failed = false;
  
  queue q;
  auto ctxt = q.get_context();
  Node* d_head = nullptr;
  Node* d_cur = nullptr;
  
  for (int i = 0; i < numNodes; i++) {
    if (i == 0) {
      d_head = (Node *) aligned_alloc_host(
        alignof(Node),
        sizeof(Node),
        ctxt);
      if (d_head == nullptr) {
        failed = true;
        break;
      }
      d_cur = d_head;
    }

    d_cur->Num = i * 2;

    if (i != (numNodes - 1)) {
      d_cur->pNext = (Node *) aligned_alloc_host(
        alignof(Node),
        sizeof(Node),
        ctxt);
      if (d_cur->pNext == nullptr) {
        failed = true;
        break;
      }
    }
    else {
      d_cur->pNext = nullptr;
    }

    d_cur = d_cur->pNext;
  }
  
  if (!failed) {
    auto e1 = q.submit([=](handler& cgh) {
        cgh.single_task<class foo>([=]() {
            Node* pHead = d_head;
            while (pHead) {
              pHead->Num = pHead->Num * 2 + 1;
              pHead = pHead->pNext;
            }
          });
      });
    
    e1.wait();
    
    d_cur = d_head;
    for (int i = 0; i < numNodes; i++) {
      const int want = i*4 + 1;
      if (d_cur->Num != want) {
        failed = true;
      }
      Node* old = d_cur;
      d_cur = d_cur->pNext;
      free(old, ctxt);
    }
  } 

  return failed;
}
