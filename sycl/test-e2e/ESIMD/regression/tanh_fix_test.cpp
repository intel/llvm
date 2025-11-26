// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==- tanh_fix_test.cpp - Test for tanh -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <vector>

constexpr auto sycl_write = sycl::access::mode::write;
#define SIMD 16

int test_tanh(float x) {
  std::vector<float> out(SIMD);
  std::vector<float> out1(SIMD);

  float ha = x;
  float scalar_result = 0;
  float scalar_result1 = 0;

  {
    sycl::queue queue;

    sycl::buffer<float, 1> vector_buffer(out.data(), out.size());
    sycl::buffer<float, 1> scalar_buffer(&scalar_result, sycl::range<1>(1));
    sycl::buffer<float, 1> vector_buffer1(out1.data(), out1.size());
    sycl::buffer<float, 1> scalar_buffer1(&scalar_result1, sycl::range<1>(1));

    auto e = queue.submit([&](sycl::handler &cgh) {
      sycl::accessor<float, 1, sycl_write> vector_out =
          vector_buffer.get_access<sycl_write>(cgh);
      sycl::accessor<float, 1, sycl_write> scalar_out =
          scalar_buffer.get_access<sycl_write>(cgh);
      sycl::accessor<float, 1, sycl_write> vector_out1 =
          vector_buffer1.get_access<sycl_write>(cgh);
      sycl::accessor<float, 1, sycl_write> scalar_out1 =
          scalar_buffer1.get_access<sycl_write>(cgh);

      auto kernel = ([=]() [[intel::sycl_explicit_simd]] {
        using namespace sycl::ext::intel::esimd;

        simd<float, SIMD> a = ha;

        simd<float, SIMD> vector_result =
            sycl::ext::intel::experimental::esimd::tanh(a);
        simd<float, 1> scalar_result =
            sycl::ext::intel::experimental::esimd::tanh(ha);
        simd<float, SIMD> vector_result1 =
            sycl::ext::intel::experimental::esimd::tanh_cody_waite(a);
        simd<float, 1> scalar_result1 =
            sycl::ext::intel::experimental::esimd::tanh_cody_waite(ha);

        vector_result.copy_to(vector_out, 0);
        scalar_result.copy_to(scalar_out, 0);
        vector_result1.copy_to(vector_out1, 0);
        scalar_result1.copy_to(scalar_out1, 0);
      });

      cgh.single_task<class Reduction>(kernel);
    });
    queue.wait();
  }

  float std_result = std::tanh(ha);

  if (std::fabs(std_result - scalar_result) > 0.000001f) {
    std::cout << "Scalar test failed for " << x << "." << std::endl;
    return 1;
  }

  if (std::fabs(std_result - scalar_result1) > 0.000001f) {
    std::cout << "Scalar test failed for cody waite implementation for " << x
              << "." << std::endl;
    return 1;
  }

  for (int i = 0; i < SIMD; ++i) {
    if (scalar_result != out[i] &&
        !(std::isnan(scalar_result) && std::isnan(out[i]))) {
      std::cout << "Vector test failed for " << x << "." << std::endl;
      return 1;
    }

    if (scalar_result1 != out1[i] &&
        !(std::isnan(scalar_result1) && std::isnan(out1[i]))) {
      std::cout << "Vector test failed for cody waite implementation for " << x
                << "." << std::endl;
      return 1;
    }
  }

  return 0;
}

int main() {

  int test_result = 0;

  for (float x = -100.f; x < 100.f; x += 0.1f) {
    test_result |= test_tanh(x);
  }

  if (!test_result) {
    std::cout << "Pass" << std::endl;
  }
  return test_result;
}
