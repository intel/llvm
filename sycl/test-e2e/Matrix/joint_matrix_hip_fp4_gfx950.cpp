//===---joint_matrix_hip_fp4_gfx950.cpp - DPC++ joint_matrix---------------===//
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

// OCP FP4 (fp4_e2m1) joint_matrix on gfx950, alone and mixed with the FP8
// formats, for the F8F6F4 MFMA shapes (16x16x128, 32x32x64).

#include "Inputs/joint_matrix_hip_small_float.hpp"

template <typename TA, typename TB> void f8f6f4_pair() {
  hip_small_float_mfma_suite<TA, TB, 16, 16, 128>();
  hip_small_float_mfma_suite<TA, TB, 32, 32, 64>();
}

template <typename T, typename TOther> void f8f6f4_format() {
  hip_small_float_format<T, TOther, true, 16, 16, 128>();
  hip_small_float_format<T, TOther, false, 16, 16, 128>();
  hip_small_float_format<T, TOther, true, 32, 32, 64>();
  hip_small_float_format<T, TOther, false, 32, 32, 64>();
}

int main() {
  check_host_decoding<fp4_e2m1>();

  f8f6f4_format<fp4_e2m1, fp4_e2m1>();
  f8f6f4_format<fp4_e2m1, fp8_e4m3>();
  f8f6f4_format<fp8_e5m2, fp4_e2m1>();

  f8f6f4_pair<fp4_e2m1, fp4_e2m1>();
  f8f6f4_pair<fp4_e2m1, fp8_e4m3>();
  f8f6f4_pair<fp8_e4m3, fp4_e2m1>();
  f8f6f4_pair<fp4_e2m1, fp8_e5m2>();
  f8f6f4_pair<fp8_e5m2, fp4_e2m1>();
  return 0;
}
