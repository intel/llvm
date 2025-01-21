// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==----- same_unnamed_kernels.cpp - SYCL kernel naming variants test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>

template <typename F, typename B>
void run(sycl::queue &q, B &buf, const F &func) {
  auto e = q.submit([&](sycl::handler &cgh) {
    auto acc = buf.template get_access<sycl::access::mode::write>(cgh);
    cgh.single_task([=]() { func(acc); });
  });
  e.wait();
}

int main() {
  sycl::queue q;

  int A[1] = {1};
  int B[1] = {1};
  sycl::buffer<int, 1> bufA(A, 1);
  sycl::buffer<int, 1> bufB(B, 1);

  run(q, bufA,
      [&](const sycl::accessor<int, 1, sycl::access::mode::write> &acc) {
        acc[0] = 0;
      });
  run(q, bufB,
      [&](const sycl::accessor<int, 1, sycl::access::mode::write> &acc) {
        acc[0] *= 2;
      });

  sycl::host_accessor hostA{bufA, sycl::read_only};
  sycl::host_accessor hostB{bufB, sycl::read_only};
  if (hostA[0] != 0 || hostB[0] != 2)
    return -1;

  return 0;
}
