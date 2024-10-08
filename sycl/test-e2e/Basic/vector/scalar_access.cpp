// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s

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

#include <sycl/detail/core.hpp>

#include <sycl/stream.hpp>
#include <sycl/types.hpp>

typedef float float4_t __attribute__((ext_vector_type(4)));

int main() {

  sycl::queue Q;

  Q.submit([=](sycl::handler &cgh) {
    sycl::stream out(1024, 100, cgh);
    cgh.single_task<class K>([=]() {
      // Test that it is possible to get a reference to single element of the
      // vector type. This behavior could possibly change in the future, this
      // test is necessary to track that.
      float4_t my_float4 = {0.0f, 1.0f, 2.0f, 3.0f};
      float f[4];
      for (int i = 0; i < 4; ++i) {
        f[i] = reinterpret_cast<float *>(&my_float4)[i];
        if (f[i] != i)
          out << "Error: unexpected behavior because of accessing element of "
                 "the vector by reference"
              << sycl::endl;
      }

      // Test that there is no template resolution error
      sycl::float4 a = {1.0f, 2.0f, 3.0f, 4.0f};
      out << sycl::native::recip(a.x()) << sycl::endl;
    });
  });
  Q.wait();

  // Test that there is no ambiguity in overload resolution.
  sycl::float4 a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::cout << a.x() << std::endl;

  return 0;
}
