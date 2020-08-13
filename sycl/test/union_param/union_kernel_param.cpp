// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// TODO: Uncomment once test is fixed on GPU
// RUNx: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==-union_kernel_param.cpp-Checks passing unionss as kernel params--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <iostream>
#include <stdlib.h>

using namespace cl::sycl;

union TestUnion {
public:
  int myint;
  char mychar;
  double mydouble;

  TestUnion() { mydouble = 0.0; };
};

int main(int argc, char **argv) {
  TestUnion x;

  auto q = queue(gpu_selector{});
  q.submit([&](handler &cgh) {
    cgh.parallel_for(range<1>(10), [=](id<1> i) { x.mydouble = 5.0; });
  });

  if (x.mydouble != 5.0) {
    isError = true;
    if (isError)
      std::cout << " Error !!!"
                << "\n";
    else
      std::cout << " Results match !!!"
                << "\n";
  }
  return 0;
}
