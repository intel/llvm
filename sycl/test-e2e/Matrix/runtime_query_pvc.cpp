//===---runtime_query_pvc.cpp - DPC++ joint_matrix-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

using namespace sycl::ext::oneapi::experimental::matrix;

bool compare(const combination &lhs, const combination &rhs) {
  return (lhs.max_msize == rhs.max_msize && lhs.max_nsize == rhs.max_nsize &&
          lhs.max_ksize == rhs.max_ksize && lhs.msize == rhs.msize &&
          lhs.nsize == rhs.nsize && lhs.ksize == rhs.ksize &&
          lhs.atype == rhs.atype && lhs.btype == rhs.btype &&
          lhs.ctype == rhs.ctype && lhs.dtype == rhs.dtype);
}

int main() {
  std::vector<combination> expected_combinations = {
      {8, 0, 0, 0, 16, 32, matrix_type::uint8, matrix_type::uint8,
       matrix_type::sint32, matrix_type::sint32},
      {8, 0, 0, 0, 16, 32, matrix_type::uint8, matrix_type::sint8,
       matrix_type::sint32, matrix_type::sint32},
      {8, 0, 0, 0, 16, 32, matrix_type::sint8, matrix_type::uint8,
       matrix_type::sint32, matrix_type::sint32},
      {8, 0, 0, 0, 16, 32, matrix_type::sint8, matrix_type::sint8,
       matrix_type::sint32, matrix_type::sint32},
      {8, 0, 0, 0, 16, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp32, matrix_type::fp32},
      {8, 0, 0, 0, 16, 16, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 16, 16, 16, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 1, 64, 16, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 32, 64, 16, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
      {8, 0, 0, 0, 16, 8, matrix_type::tf32, matrix_type::tf32,
       matrix_type::fp32, matrix_type::fp32},
  };

  sycl::queue q;
  std::vector<combination> actual_combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  assert(actual_combinations.size() == expected_combinations.size() &&
         "Number of combinations is not equal.");

  for (int i = 0; i < actual_combinations.size(); i++) {
    assert(compare(actual_combinations[i], expected_combinations[i]) &&
           "Some values in matrix runtime query for PVC are not expected.");
  }

  return 0;
}
