// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==- fmod_compatibility_test.cpp - Test for compatibility with std::fmod -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/detail/core.hpp>
#include <vector>

constexpr auto sycl_write = sycl::access::mode::write;
#define SIMD 16

int test_fmod(float x, float y) {
  std::vector<float> out(SIMD);

  float ha = x;
  float hb = y;
  float scalar_result = 0;

  {
    sycl::queue queue;

    sycl::buffer<float, 1> vector_buffer(out.data(), out.size());
    sycl::buffer<float, 1> scalar_buffer(&scalar_result, sycl::range<1>(1));

    auto e = queue.submit([&](sycl::handler &cgh) {
      sycl::accessor<float, 1, sycl_write> vector_out =
          vector_buffer.get_access<sycl_write>(cgh);
      sycl::accessor<float, 1, sycl_write> scalar_out =
          scalar_buffer.get_access<sycl_write>(cgh);

      auto kernel = ([=]() [[intel::sycl_explicit_simd]] {
        using namespace sycl::ext::intel::esimd;

        simd<float, SIMD> a = ha;
        simd<float, SIMD> b = hb;

        simd<float, SIMD> vector_result =
            sycl::ext::intel::experimental::esimd::fmod(a, b);
        simd<float, 1> scalar_result =
            sycl::ext::intel::experimental::esimd::fmod(ha, hb);

        vector_result.copy_to(vector_out, 0);
        scalar_result.copy_to(scalar_out, 0);
      });

      cgh.single_task<class Reduction>(kernel);
    });
    queue.wait();
  }

  float mod = std::fmod(ha, hb);

  if (mod != out[0] || std::signbit(mod) != std::signbit(out[0])) {
    std::cout << "Vector test failed for " << x << " and " << y << "."
              << std::endl;
    return 1;
  }

  if (mod != scalar_result ||
      std::signbit(mod) != std::signbit(scalar_result)) {
    std::cout << "Scalar test failed for " << x << " and " << y << "."
              << std::endl;
    return 1;
  }

  return 0;
}

int main() {

  int test_result = 0;
  test_result |= test_fmod(49152 * 1364.0f + 626.0f, 509);
  test_result |= test_fmod(+5.1f, +3.0f);
  test_result |= test_fmod(-5.1f, +3.0f);
  test_result |= test_fmod(+5.1f, -3.0f);
  test_result |= test_fmod(-5.1f, -3.0f);
  test_result |= test_fmod(+0.0f, 1.0f);
  test_result |= test_fmod(-0.0f, 1.0f);

  if (!test_result) {
    std::cout << "Pass" << std::endl;
  }
  return test_result;
}
