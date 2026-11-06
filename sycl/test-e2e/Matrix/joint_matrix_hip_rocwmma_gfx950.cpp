//===---joint_matrix_hip_rocwmma_gfx950.cpp - DPC++ joint_matrix-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx950 %s -o %t.out
// RUN: %{run} %t.out

// REQUIRES: target-amd
// REQUIRES: arch-amd_gpu_gfx950
// REQUIRES: aspect-fp16

// rocWMMA-style GEMM and load/store contamination tests of the int8, half,
// bfloat16, float and double joint_matrix shapes of gfx950.

#include "Inputs/joint_matrix_hip_rocwmma.hpp"

template <typename TA, typename TC, size_t M, size_t N, size_t K> void test() {
  hip_rocwmma_gemm<TA, TA, TC, M, N, K>();
  hip_rocwmma_contamination<TA, TA, TC, M, N, K>();
}

int main() {
  test<int8_t, int32_t, 32, 32, 16>();
  test<int8_t, int32_t, 16, 16, 32>();
  test<int8_t, int32_t, 32, 32, 32>();
  test<int8_t, int32_t, 16, 16, 64>();
  test<half, float, 32, 32, 8>();
  test<half, float, 16, 16, 16>();
  test<half, float, 32, 32, 16>();
  test<half, float, 16, 16, 32>();
  test<bfloat16, float, 32, 32, 8>();
  test<bfloat16, float, 16, 16, 16>();
  test<bfloat16, float, 32, 32, 16>();
  test<bfloat16, float, 16, 16, 32>();
  test<float, float, 32, 32, 2>();
  test<float, float, 16, 16, 4>();
  test<double, double, 16, 16, 4>();
  return 0;
}
