// RUN: %clang -std=c++11 -fsycl %s -o %t1.out -lstdc++ -lOpenCL -lsycl -DINTEL_USM
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
//==------------------- mixed.cpp - Mixed Memory test ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include <CL/sycl.hpp>

using namespace cl::sycl;

class foo;
int main() {
  bool failed = false;
  int* darray = nullptr;
  int* sarray = nullptr;
  int* harray = nullptr;
  const int N = 4;
  const int MAGIC_NUM = 42;
  
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  darray = (int *) malloc_device(N*sizeof(int), dev, ctxt);
  if (darray == nullptr) {
    failed = true;
  }
  else {
    sarray = (int *) malloc_shared(N*sizeof(int), dev, ctxt);

    if (sarray == nullptr) {
      failed = true;
    }
    else {
      harray = (int *) malloc_host(N*sizeof(int), ctxt);
      if (harray == nullptr) {
        failed = true;
      }
      else {
        for (int i = 0; i < N; i++) {
          sarray[i] = MAGIC_NUM-1;
          harray[i] = 1;
        }

        auto e0 = q.memset(darray, 0, N * sizeof(int));
        e0.wait();
        
        if (!failed) {
          auto e1 = q.submit([=](handler& cgh) {
              cgh.single_task<class foo>([=]() {
                  for (int i = 0; i < N; i++) {
                    sarray[i] += darray[i] + harray[i];;
                  }
                });
            });
          
          e1.wait();
          
          
          for (int i = 0; i < N; i++) {
            if (sarray[i] != MAGIC_NUM) {
              failed = true;
            }
          }
          free(darray, ctxt);
          free(sarray, ctxt);
          free(harray, ctxt);
        }
      }
    }
  }
  return failed;
}
