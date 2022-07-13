// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out %HOST_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER

//==------- scalar_vec_access.cpp - SYCL scalar access to vec test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// CHECK-NOT: Error: unexpected behavior because of accessing element of the
// vector by reference

#include <iostream>
#include <sycl/sycl.hpp>

typedef float float4_t __attribute__((ext_vector_type(4)));

int main() {

  cl::sycl::queue Q;

  Q.submit([=](cl::sycl::handler &cgh) {
    cl::sycl::stream out(1024, 100, cgh);
    cgh.single_task<class K>([=]() {
      // Test that it is possible to get a reference to single element of the
      // vector type. This behavior could possibly change in the future, this
      // test is necessary to track that.
      float4_t my_float4 = {0.0, 1.0, 2.0, 3.0};
      float f[4];
      for (int i = 0; i < 4; ++i) {
        f[i] = reinterpret_cast<float *>(&my_float4)[i];
        if (f[i] != i)
          out << "Error: unexpected behavior because of accessing element of "
                 "the vector by reference"
              << cl::sycl::endl;
      }

      // Test that there is no template resolution error
      cl::sycl::float4 a = {1.0, 2.0, 3.0, 4.0};
      out << cl::sycl::native::recip(a.x()) << cl::sycl::endl;
    });
  });
  Q.wait();

  // Test that there is no ambiguity in overload resolution.
  cl::sycl::float4 a = {1.0, 2.0, 3.0, 4.0};
  std::cout << a.x() << std::endl;

  return 0;
}
