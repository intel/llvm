//==--------------- multisource.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// XFAIL: cuda
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/16413
// Separate kernel sources and host code sources
// Test with `--offload-new-driver`
// RUN: %{build} --offload-new-driver -c -o %t.kernel.o -DINIT_KERNEL -DCALC_KERNEL
// RUN: %{build} --offload-new-driver -c -o %t.main.o -DMAIN_APP
// RUN: %clangxx -Wno-error=unused-command-line-argument -fsycl -fsycl-targets=%{sycl_triple} --offload-new-driver %t.kernel.o %t.main.o -o %t1.fat
// RUN: %{run} %t1.fat

// Multiple sources with kernel code
// Test with `--offload-new-driver`
// RUN: %{build} --offload-new-driver -c -o %t.init.o -DINIT_KERNEL
// RUN: %{build} --offload-new-driver -c -o %t.calc.o -DCALC_KERNEL
// RUN: %{build} --offload-new-driver -c -o %t.main.o -DMAIN_APP
// RUN: %clangxx -Wno-error=unused-command-line-argument -fsycl -fsycl-targets=%{sycl_triple} --offload-new-driver %t.init.o %t.calc.o %t.main.o -o %t2.fat
// RUN: %{run} %t2.fat

// Multiple sources with kernel code with old-style objects
// Test with `--offload-new-driver`
// RUN: %{build} --no-offload-new-driver -c -o %t.init.o -DINIT_KERNEL
// RUN: %{build} --no-offload-new-driver -c -o %t.calc.o -DCALC_KERNEL
// RUN: %{build} --no-offload-new-driver -c -o %t.main.o -DMAIN_APP
// RUN: %clangxx -Wno-error=unused-command-line-argument -fsycl -fsycl-targets=%{sycl_triple} --offload-new-driver %t.init.o %t.calc.o %t.main.o -o %t3.fat
// RUN: %{run} %t3.fat

// Multiple sources with kernel code with old-style objects in a static archive
// Test with `--offload-new-driver`
// RUN: %{build} --no-offload-new-driver -c -o %t.init.o -DINIT_KERNEL
// RUN: %{build} --no-offload-new-driver -c -o %t.calc.o -DCALC_KERNEL
// RUN: %{build} --no-offload-new-driver -c -o %t.main.o -DMAIN_APP
// RUN: llvm-ar r %t.a %t.init.o %t.calc.o
// RUN: %clangxx -Wno-error=unused-command-line-argument -fsycl -fsycl-targets=%{sycl_triple} --offload-new-driver %t.main.o %t.a -o %t4.fat
// RUN: %{run} %t4.fat

#include <sycl/detail/core.hpp>

#include <iostream>

using namespace sycl;

#ifdef MAIN_APP
void init_buf(queue &q, buffer<int, 1> &b, range<1> &r, int i);
#elif INIT_KERNEL
void init_buf(queue &q, buffer<int, 1> &b, range<1> &r, int i) {
  q.submit([&](handler &cgh) {
    auto B = b.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class init>(r, [=](id<1> index) { B[index] = i; });
  });
}
#endif

#ifdef MAIN_APP
void calc_buf(queue &q, buffer<int, 1> &a, buffer<int, 1> &b, buffer<int, 1> &c,
              range<1> &r);
#elif CALC_KERNEL
void calc_buf(queue &q, buffer<int, 1> &a, buffer<int, 1> &b, buffer<int, 1> &c,
              range<1> &r) {
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

    auto C = c.get_host_access();
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
