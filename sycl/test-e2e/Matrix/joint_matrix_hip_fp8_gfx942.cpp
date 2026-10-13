//===---joint_matrix_hip_fp8_gfx942.cpp - DPC++ joint_matrix--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx942 %s -o %t.out
// RUN: %{run} %t.out

// REQUIRES: target-amd
// REQUIRES: arch-amd_gpu_gfx942

#include "joint_matrix_hip_fp8_mfma.hpp"

// gfx942 (CDNA3) fp8 (E4M3) / bf8 (E5M2) MFMA supports 16x16x32 and 32x32x16
// with an fp32 accumulator. The A and B operand formats are independent, so all
// four (E4M3, E5M2) format pairs are exercised for both shapes: 8 combinations.
// Each combination is run with a row_major and a col_major accumulator store.
template <size_t KX, layout OutLayout> void matrix_fp8_mfma_layout() {
  hip_matrix_fp8_mfma<fp8_e4m3, fp8_e4m3, 32, 32, 16, KX, OutLayout>();
  hip_matrix_fp8_mfma<fp8_e4m3, fp8_e4m3, 16, 16, 32, KX, OutLayout>();
  hip_matrix_fp8_mfma<fp8_e5m2, fp8_e5m2, 32, 32, 16, KX, OutLayout>();
  hip_matrix_fp8_mfma<fp8_e5m2, fp8_e5m2, 16, 16, 32, KX, OutLayout>();
  hip_matrix_fp8_mfma<fp8_e4m3, fp8_e5m2, 32, 32, 16, KX, OutLayout>();
  hip_matrix_fp8_mfma<fp8_e4m3, fp8_e5m2, 16, 16, 32, KX, OutLayout>();
  hip_matrix_fp8_mfma<fp8_e5m2, fp8_e4m3, 32, 32, 16, KX, OutLayout>();
  hip_matrix_fp8_mfma<fp8_e5m2, fp8_e4m3, 16, 16, 32, KX, OutLayout>();
}

template <size_t KX> void matrix_fp8_mfma() {
  matrix_fp8_mfma_layout<KX, layout::row_major>();
  matrix_fp8_mfma_layout<KX, layout::col_major>();
}

int main() {
  matrix_fp8_mfma<1>();
  matrix_fp8_mfma<2>();
  matrix_fp8_mfma<3>();
  matrix_fp8_mfma<4>();
}
