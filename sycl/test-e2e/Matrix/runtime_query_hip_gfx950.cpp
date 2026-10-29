//===---runtime_query_hip_gfx950.cpp - DPC++ joint_matrix-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target-amd
// REQUIRES: arch-amd_gpu_gfx950
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx950 %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

using namespace sycl::ext::oneapi::experimental::matrix;

bool find_combination(const combination &comb,
                      const std::vector<combination> &expected_combinations) {
  return std::find_if(expected_combinations.begin(),
                      expected_combinations.end(),
                      [&comb](const auto &expected_comb) {
                        return (comb.max_msize == expected_comb.max_msize &&
                                comb.max_nsize == expected_comb.max_nsize &&
                                comb.max_ksize == expected_comb.max_ksize &&
                                comb.msize == expected_comb.msize &&
                                comb.nsize == expected_comb.nsize &&
                                comb.ksize == expected_comb.ksize &&
                                comb.atype == expected_comb.atype &&
                                comb.btype == expected_comb.btype &&
                                comb.ctype == expected_comb.ctype &&
                                comb.dtype == expected_comb.dtype);
                      }) != expected_combinations.end();
}

int main() {
  std::vector<combination> expected_combinations = {
      {0, 0, 0, 32, 32, 8, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 32, 32, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 16, 16, 32, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 32, 32, 2, matrix_type::fp32, matrix_type::fp32,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 16, 16, 4, matrix_type::fp32, matrix_type::fp32,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 32, 32, 16, matrix_type::sint8, matrix_type::sint8,
       matrix_type::sint32, matrix_type::sint32},
      {0, 0, 0, 16, 16, 32, matrix_type::sint8, matrix_type::sint8,
       matrix_type::sint32, matrix_type::sint32},
      {0, 0, 0, 32, 32, 32, matrix_type::sint8, matrix_type::sint8,
       matrix_type::sint32, matrix_type::sint32},
      {0, 0, 0, 16, 16, 64, matrix_type::sint8, matrix_type::sint8,
       matrix_type::sint32, matrix_type::sint32},
      {0, 0, 0, 32, 32, 8, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 16, 16, 16, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 32, 32, 16, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 16, 16, 32, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 16, 16, 4, matrix_type::fp64, matrix_type::fp64,
       matrix_type::fp64, matrix_type::fp64}};

  const matrix_type fp8_types[] = {matrix_type::fp8_e4m3,
                                   matrix_type::fp8_e5m2};
  const matrix_type f8f4_types[] = {
      matrix_type::fp8_e4m3, matrix_type::fp8_e5m2, matrix_type::fp4_e2m1};
  for (matrix_type a : fp8_types)
    for (matrix_type b : fp8_types) {
      expected_combinations.push_back(
          {0, 0, 0, 32, 32, 16, a, b, matrix_type::fp32, matrix_type::fp32});
      expected_combinations.push_back(
          {0, 0, 0, 16, 16, 32, a, b, matrix_type::fp32, matrix_type::fp32});
    }
  for (matrix_type a : f8f4_types)
    for (matrix_type b : f8f4_types) {
      expected_combinations.push_back(
          {0, 0, 0, 32, 32, 64, a, b, matrix_type::fp32, matrix_type::fp32});
      expected_combinations.push_back(
          {0, 0, 0, 16, 16, 128, a, b, matrix_type::fp32, matrix_type::fp32});
    }

  sycl::queue q;
  std::vector<combination> actual_combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  assert(actual_combinations.size() == expected_combinations.size() &&
         "Number of combinations is not equal.");

  for (auto &comb : actual_combinations) {
    assert(find_combination(comb, expected_combinations) &&
           "Some values in matrix runtime query for gfx950 are not expected.");
  }
  return 0;
}
