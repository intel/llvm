// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//==- math_const_fix_test.cpp - Test to verify math functions correctness-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This is basic test to test some functional equivalency between ESIMD and STD
// implementations of some math functions (sin, cos, atan, atan2).

#include <cmath>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/usm.hpp>
#include <sycl/usm/usm_allocator.hpp>
#include <vector>

int ErrCnt = 0;
template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;
template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

int test_sin_cos_atan(float xmin, float xmax, float step) {
  int size = (xmax - xmin) / step;

  sycl::queue queue;
  shared_allocator<float> allocator(queue);

  shared_vector<float> vector_input(size, allocator);
  shared_vector<float> vector_output_sin(size, allocator);
  shared_vector<float> vector_output_cos(size, allocator);
  shared_vector<float> vector_output_atan(size, allocator);

  shared_vector<float> scalar_output_sin(size, allocator);
  shared_vector<float> scalar_output_cos(size, allocator);
  shared_vector<float> scalar_output_atan(size, allocator);

  int idx = 0;
  for (float inputValue = xmin; inputValue < xmax; inputValue += step) {
    vector_input[idx++] = inputValue;
  }

  auto GlobalRange = sycl::range<1>(size);
  sycl::range<1> LocalRange{1};
  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  {
    auto e = queue.submit([&](sycl::handler &cgh) {
      float *vector_input_ptr = vector_input.data();
      float *vector_output_sin_ptr = vector_output_sin.data();
      float *scalar_output_sin_ptr = scalar_output_sin.data();
      float *vector_output_cos_ptr = vector_output_cos.data();
      float *scalar_output_cos_ptr = scalar_output_cos.data();
      float *vector_output_atan_ptr = vector_output_atan.data();
      float *scalar_output_atan_ptr = scalar_output_atan.data();
      auto kernel = ([=](sycl::nd_item<1> ndi) [[intel::sycl_explicit_simd]] {
        using namespace sycl::ext::intel::esimd;
        auto idx = ndi.get_global_id(0);

        simd<float, 1> a;
        simd<float, 1> scalar_result_sin;
        simd<float, 1> scalar_result_cos;
        simd<float, 1> scalar_result_atan;

        a.copy_from(vector_input_ptr + idx);

        simd<float, 1> vector_result_sin =
            sycl::ext::intel::experimental::esimd::sin_emu(a);
        simd<float, 1> vector_result_cos =
            sycl::ext::intel::experimental::esimd::cos_emu(a);
        simd<float, 1> vector_result_atan =
            sycl::ext::intel::experimental::esimd::atan(a);

        float input = a[0];
        scalar_result_sin[idx] =
            sycl::ext::intel::experimental::esimd::sin_emu(input);
        scalar_result_cos[idx] =
            sycl::ext::intel::experimental::esimd::cos_emu(input);
        scalar_result_atan[idx] =
            sycl::ext::intel::experimental::esimd::atan(input);

        vector_result_sin.copy_to(vector_output_sin_ptr + idx);
        scalar_result_sin.copy_to(scalar_output_sin_ptr + idx);
        vector_result_cos.copy_to(vector_output_cos_ptr + idx);
        scalar_result_cos.copy_to(scalar_output_cos_ptr + idx);
        vector_result_atan.copy_to(vector_output_atan_ptr + idx);
        scalar_result_atan.copy_to(scalar_output_atan_ptr + idx);
      });

      cgh.parallel_for<class TestSinCos>(Range, kernel);
    });
    queue.wait();
  }
  for (int i = 0; i < size; ++i) {
    float input = vector_input[i];
    float std_result_sin = std::sin(input);
    float std_result_cos = std::cos(input);
    float std_result_atan = std::atan(input);
    if (std::fabs(std_result_atan - vector_output_atan[i]) > 0.00001f) {
      if (ErrCnt++ < 100)
        std::cout << "Vector test failed for atan for " << input << "."
                  << std::endl;
      return 1;
    }

    if (std::fabs(std_result_sin - vector_output_sin[i]) > 0.00001f) {
      if (ErrCnt++ < 100)
        std::cout << "Vector test failed for sin for " << input << "."
                  << std::endl;
      return 1;
    }

    if (std::fabs(std_result_cos - vector_output_cos[i]) > 0.00001f) {
      if (ErrCnt++ < 100)
        std::cout << "Vector test failed for cos for " << input << "."
                  << std::endl;
      return 1;
    }
    if (scalar_output_sin[i] != vector_output_sin[i]) {
      if (ErrCnt++ < 100)
        std::cout << "Scalar test failed for sin " << input << "." << std::endl;
      return 1;
    }

    if (scalar_output_cos[i] != vector_output_cos[i]) {
      if (ErrCnt++ < 100)
        std::cout << "Scalar test failed for cos for " << input << "."
                  << std::endl;
      return 1;
    }

    if (scalar_output_atan[i] != vector_output_atan[i]) {
      if (ErrCnt++ < 100)
        std::cout << "Scalar test failed for atan for " << input << "."
                  << std::endl;
      return 1;
    }
  }

  return 0;
}

int test_atan2(float min, float max, float step) {
  int size = (max - min) / step;

  sycl::queue queue;
  shared_allocator<float> allocator(queue);

  shared_vector<float> vector_input(size, allocator);
  shared_vector<float> vector_output_atan2_fast(size * size, allocator);
  shared_vector<float> vector_output_atan2(size * size, allocator);
  shared_vector<float> scalar_output_atan2_fast(size * size, allocator);
  shared_vector<float> scalar_output_atan2(size * size, allocator);

  int idx = 0;
  for (float inputValue = min; inputValue < max; inputValue += step) {
    vector_input[idx++] = inputValue;
  }

  auto GlobalRange = sycl::range<2>(size, size);
  sycl::range<2> LocalRange{1, 1};
  sycl::nd_range<2> Range(GlobalRange, LocalRange);

  {
    auto e = queue.submit([&](sycl::handler &cgh) {
      float *vector_input_ptr = vector_input.data();
      float *vector_output_atan2_fast_ptr = vector_output_atan2_fast.data();
      float *scalar_output_atan2_fast_ptr = scalar_output_atan2_fast.data();
      float *vector_output_atan2_ptr = vector_output_atan2.data();
      float *scalar_output_atan2_ptr = scalar_output_atan2.data();

      auto kernel = ([=](sycl::nd_item<2> ndi) [[intel::sycl_explicit_simd]] {
        using namespace sycl::ext::intel::esimd;
        auto idx_x = ndi.get_global_id(0);
        auto idx_y = ndi.get_global_id(1);

        simd<float, 1> a;
        simd<float, 1> b;
        a.copy_from(vector_input_ptr + idx_x);
        b.copy_from(vector_input_ptr + idx_y);

        float x = a[0];
        float y = b[0];
        simd<float, 1> vector_result =
            sycl::ext::intel::experimental::esimd::atan2_fast(a, b);
        simd<float, 1> scalar_result =
            sycl::ext::intel::experimental::esimd::atan2_fast(x, y);
        simd<float, 1> vector_result1 =
            sycl::ext::intel::experimental::esimd::atan2(a, b);
        simd<float, 1> scalar_result1 =
            sycl::ext::intel::experimental::esimd::atan2(x, y);

        vector_result.copy_to(vector_output_atan2_fast_ptr + idx_y * size +
                              idx_x);
        scalar_result.copy_to(scalar_output_atan2_fast_ptr + idx_y * size +
                              idx_x);
        vector_result1.copy_to(vector_output_atan2_ptr + idx_y * size + idx_x);
        scalar_result1.copy_to(scalar_output_atan2_ptr + idx_y * size + idx_x);
      });

      cgh.parallel_for<class TestAtan>(Range, kernel);
    });
    queue.wait();
  }

  for (int ix = 0; ix < size; ++ix) {
    for (int iy = 0; iy < size; ++iy) {
      float input_x = vector_input[ix];
      float input_y = vector_input[iy];
      if (!(std::abs(input_x) > 0.00001f && std::abs(input_y) > 0.00001f)) {
        continue;
      }

      float std_result_atan = std::atan2(input_x, input_y);

      if (std::fabs(std_result_atan - vector_output_atan2[iy * size + ix]) >
          0.0001f) {
        if (ErrCnt++ < 100)
          std::cout << "Vector test failed for atan2 for " << input_x << ","
                    << input_y << "." << std::endl;
        return 1;
      }

      if (std::fabs(std_result_atan -
                    vector_output_atan2_fast[iy * size + ix]) > 0.1f) {
        if (ErrCnt++ < 100)
          std::cout << "Vector test failed for atan2_fast for " << input_x
                    << "," << input_y << "." << std::endl;
        return 1;
      }

      if (scalar_output_atan2[iy * size + ix] !=
          vector_output_atan2[iy * size + ix]) {
        if (ErrCnt++ < 100)
          std::cout << "Scalar test failed for atan2 for " << input_x << ","
                    << input_y << "." << std::endl;
        return 1;
      }

      if (scalar_output_atan2_fast[iy * size + ix] !=
          vector_output_atan2_fast[iy * size + ix]) {
        if (ErrCnt++ < 100)
          std::cout << "Scalar test failed for atan2_fast for " << input_x
                    << "," << input_y << "." << std::endl;
        return 1;
      }
    }
  }
  return 0;
}

int main() {

  int test_result = 0;

  test_result |= test_sin_cos_atan(-10.f, 10.f, 0.1f);
  test_result |= test_atan2(-10.f, 10.f, 0.1f);

  if (!test_result) {
    std::cout << "Pass" << std::endl;
  }
  return test_result;
}
