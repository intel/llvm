//===---joint_matrix_hip_fp8_gfx950.cpp - DPC++ joint_matrix---------------===//
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

// OCP FP8 (fp8_e4m3, fp8_e5m2) joint_matrix on gfx950, including mixed A/B
// formats, for the FP8 MFMA shapes (16x16x32, 32x32x16) and the F8F6F4 MFMA
// shapes (16x16x128, 32x32x64).

#include "Inputs/joint_matrix_hip_small_float.hpp"

template <typename TA, typename TB> void fp8_pair() {
  hip_small_float_mfma_suite<TA, TB, 16, 16, 32>();
  hip_small_float_mfma_suite<TA, TB, 32, 32, 16>();
  hip_small_float_mfma_suite<TA, TB, 16, 16, 128>();
  hip_small_float_mfma_suite<TA, TB, 32, 32, 64>();
}

template <typename T, typename TOther> void fp8_format() {
  hip_small_float_format<T, TOther, true, 16, 16, 32>();
  hip_small_float_format<T, TOther, false, 16, 16, 32>();
  hip_small_float_format<T, TOther, true, 32, 32, 16>();
  hip_small_float_format<T, TOther, false, 32, 32, 16>();
  hip_small_float_format<T, TOther, true, 16, 16, 128>();
  hip_small_float_format<T, TOther, false, 16, 16, 128>();
  hip_small_float_format<T, TOther, true, 32, 32, 64>();
  hip_small_float_format<T, TOther, false, 32, 32, 64>();
}

int main() {
  check_host_decoding<fp8_e4m3>();
  check_host_decoding<fp8_e5m2>();

  fp8_format<fp8_e4m3, fp8_e4m3>();
  fp8_format<fp8_e5m2, fp8_e5m2>();
  fp8_format<fp8_e4m3, fp8_e5m2>();
  fp8_format<fp8_e5m2, fp8_e4m3>();

  fp8_pair<fp8_e4m3, fp8_e4m3>();
  fp8_pair<fp8_e5m2, fp8_e5m2>();
  fp8_pair<fp8_e4m3, fp8_e5m2>();
  fp8_pair<fp8_e5m2, fp8_e4m3>();
  return 0;
}
