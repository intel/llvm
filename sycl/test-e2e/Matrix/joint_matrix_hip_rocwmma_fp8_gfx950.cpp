//===---joint_matrix_hip_rocwmma_fp8_gfx950.cpp - DPC++ joint_matrix-------===//
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

// rocWMMA-style GEMM and load/store contamination tests of the OCP FP8
// joint_matrix shapes of gfx950, including mixed E4M3/E5M2 operands.

#include "Inputs/joint_matrix_hip_rocwmma.hpp"

template <typename TA, typename TB, size_t M, size_t N, size_t K> void test() {
  hip_rocwmma_gemm<TA, TB, float, M, N, K>();
  hip_rocwmma_contamination<TA, TB, float, M, N, K>();
}

template <typename TA, typename TB> void test_shapes() {
  test<TA, TB, 16, 16, 32>();
  test<TA, TB, 32, 32, 16>();
  test<TA, TB, 16, 16, 128>();
  test<TA, TB, 32, 32, 64>();
}

int main() {
  test_shapes<fp8_e4m3, fp8_e4m3>();
  test_shapes<fp8_e5m2, fp8_e5m2>();
  test_shapes<fp8_e4m3, fp8_e5m2>();
  test_shapes<fp8_e5m2, fp8_e4m3>();
  return 0;
}
