// RUN: %clangxx -fsycl %s -o %t.out -fsycl-unnamed-lambda
// RUN: %RUN_ON_HOST %t.out

//==----- same_unnamed_kernels.cpp - SYCL kernel naming variants test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

template <typename F, typename B>
void run(cl::sycl::queue &q, B &buf, const F &func) {
  auto e = q.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.template get_access<cl::sycl::access::mode::write>(cgh);
    cgh.single_task([=]() { func(acc); });
  });
  e.wait();
}

int main() {
  cl::sycl::queue q;

  int A[1] = {1};
  int B[1] = {1};
  cl::sycl::buffer<int, 1> bufA(A, 1);
  cl::sycl::buffer<int, 1> bufB(B, 1);

  run(q, bufA,
      [&](const cl::sycl::accessor<int, 1, cl::sycl::access::mode::write>
              &acc) { acc[0] = 0; });
  run(q, bufB,
      [&](const cl::sycl::accessor<int, 1, cl::sycl::access::mode::write>
              &acc) { acc[0] *= 2; });

  if (A[0] != 0 || B[0] != 2)
    return -1;

  return 0;
}
