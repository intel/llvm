//===---joint_matrix_hip_rocwmma_fp4_gfx950.cpp - DPC++ joint_matrix-------===//
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

// rocWMMA-style GEMM and load/store contamination tests of the OCP FP4
// joint_matrix shapes of gfx950, alone and mixed with the FP8 formats.

#include "Inputs/joint_matrix_hip_rocwmma.hpp"

template <typename TA, typename TB> void test() {
  hip_rocwmma_gemm<TA, TB, float, 16, 16, 128>();
  hip_rocwmma_gemm<TA, TB, float, 32, 32, 64>();
  hip_rocwmma_contamination<TA, TB, float, 16, 16, 128>();
  hip_rocwmma_contamination<TA, TB, float, 32, 32, 64>();
}

int main() {
  test<fp4_e2m1, fp4_e2m1>();
  test<fp4_e2m1, fp8_e4m3>();
  test<fp8_e4m3, fp4_e2m1>();
  test<fp4_e2m1, fp8_e5m2>();
  test<fp8_e5m2, fp4_e2m1>();
  return 0;
}
