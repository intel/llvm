//==--------------- multisource.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Separate kernel sources and host code sources
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -c -o %t.kernel.o %s -DINIT_KERNEL -DCALC_KERNEL
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -c -o %t.main.o %s -DMAIN_APP
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %t.kernel.o %t.main.o -o %t.fat
// RUN: env SYCL_DEVICE_TYPE=HOST %t.fat
// RUN: %CPU_RUN_PLACEHOLDER %t.fat
// RUN: %GPU_RUN_PLACEHOLDER %t.fat
// RUN: %ACC_RUN_PLACEHOLDER %t.fat

// Multiple sources with kernel code
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -c -o %t.init.o %s -DINIT_KERNEL
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -c -o %t.calc.o %s -DCALC_KERNEL
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -c -o %t.main.o %s -DMAIN_APP
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %t.init.o %t.calc.o %t.main.o -o %t.fat
// RUN: env SYCL_DEVICE_TYPE=HOST %t.fat
// RUN: %CPU_RUN_PLACEHOLDER %t.fat
// RUN: %GPU_RUN_PLACEHOLDER %t.fat
// RUN: %ACC_RUN_PLACEHOLDER %t.fat


#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

#ifdef MAIN_APP
void init_buf(queue &q, buffer<int, 1> &b, range<1> &r, int i) ;
#elif INIT_KERNEL
void init_buf(queue &q, buffer<int, 1> &b, range<1> &r, int i){
  q.submit([&](handler &cgh) {
    auto B = b.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class init>(r, [=](id<1> index) { B[index] = i; });
  });
}
#endif

#ifdef MAIN_APP
void calc_buf(queue &q, buffer<int, 1> &a, buffer<int, 1> &b,
              buffer<int, 1> &c, range<1> &r);
#elif CALC_KERNEL
void calc_buf(queue &q, buffer<int, 1> &a, buffer<int, 1> &b,
              buffer<int, 1> &c, range<1> &r){
  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc>(
        r, [=](id<1> index) { C[index] = A[index] - B[index]; });
  });
}
#endif

#ifdef MAIN_APP
const size_t N = 100;
int main() {
  {
    queue q;

    range<1> r(N);
    buffer<int, 1> a(r);
    buffer<int, 1> b(r);
    buffer<int, 1> c(r);

    init_buf(q, a, r, 2);
    init_buf(q, b, r, 1);

    calc_buf(q, a, b, c, r);

    auto C = c.get_access<access::mode::read>();
    for (size_t i = 0; i < N; i++) {
      if (C[i] != 1) {
        std::cout << "Wrong value " << C[i] << " for element " << i
                  << std::endl;
        return -1;
      }
    }
  }

  std::cout << "Done!" << std::endl;
  return 0;
}
#endif

