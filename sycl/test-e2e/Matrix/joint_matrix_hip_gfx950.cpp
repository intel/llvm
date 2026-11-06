//===---joint_matrix_hip_gfx950.cpp - DPC++ joint_matrix-------------------===//
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

#include "joint_matrix_hip_apply.hpp"
#include "joint_matrix_hip_copy.hpp"
#include "joint_matrix_hip_fill.hpp"
#include "joint_matrix_hip_mfma.hpp"

// gfx950 (CDNA4) supports every gfx942 shape and adds double-K bf16
// (16x16x32, 32x32x16) and int8 (16x16x64, 32x32x32) MFMAs.
template <size_t KX, layout OutLayout> void matrix_mfma() {
  hip_matrix_mfma<int8_t, int32_t, 32, 32, 16, KX, OutLayout>();
  hip_matrix_mfma<int8_t, int32_t, 16, 16, 32, KX, OutLayout>();
  hip_matrix_mfma<int8_t, int32_t, 32, 32, 32, KX, OutLayout>();
  hip_matrix_mfma<int8_t, int32_t, 16, 16, 64, KX, OutLayout>();
  hip_matrix_mfma<bfloat16, float, 32, 32, 8, KX, OutLayout>();
  hip_matrix_mfma<bfloat16, float, 16, 16, 16, KX, OutLayout>();
  hip_matrix_mfma<bfloat16, float, 32, 32, 16, KX, OutLayout>();
  hip_matrix_mfma<bfloat16, float, 16, 16, 32, KX, OutLayout>();
  hip_matrix_mfma<float, float, 32, 32, 2, KX, OutLayout>();
  hip_matrix_mfma<float, float, 16, 16, 4, KX, OutLayout>();
  hip_matrix_mfma<double, double, 16, 16, 4, KX, OutLayout>();
}

template <layout OutLayout> void matrix_copy() {
  hip_matrix_copy<int8_t, int32_t, 32, 32, 16, OutLayout>();
  hip_matrix_copy<int8_t, int32_t, 16, 16, 32, OutLayout>();
  hip_matrix_copy<int8_t, int32_t, 32, 32, 32, OutLayout>();
  hip_matrix_copy<int8_t, int32_t, 16, 16, 64, OutLayout>();
  hip_matrix_copy<bfloat16, float, 32, 32, 8, OutLayout>();
  hip_matrix_copy<bfloat16, float, 16, 16, 16, OutLayout>();
  hip_matrix_copy<bfloat16, float, 32, 32, 16, OutLayout>();
  hip_matrix_copy<bfloat16, float, 16, 16, 32, OutLayout>();
  hip_matrix_copy<float, float, 32, 32, 2, OutLayout>();
  hip_matrix_copy<float, float, 16, 16, 4, OutLayout>();
  hip_matrix_copy<double, double, 16, 16, 4, OutLayout>();
}

int main() {
  check_hip_col_major_rectangles();

  matrix_mfma<1, layout::row_major>();
  matrix_mfma<2, layout::row_major>();
  matrix_mfma<3, layout::row_major>();
  matrix_mfma<4, layout::row_major>();
  matrix_mfma<1, layout::col_major>();
  matrix_mfma<2, layout::col_major>();
  matrix_mfma<3, layout::col_major>();
  matrix_mfma<4, layout::col_major>();

  matrix_copy<layout::row_major>();
  matrix_copy<layout::col_major>();

  hip_matrix_fill<int8_t, int32_t, 32, 32, 16>();
  hip_matrix_fill<int8_t, int32_t, 16, 16, 32>();
  hip_matrix_fill<int8_t, int32_t, 32, 32, 32>();
  hip_matrix_fill<int8_t, int32_t, 16, 16, 64>();
  hip_matrix_fill<bfloat16, float, 32, 32, 8>();
  hip_matrix_fill<bfloat16, float, 16, 16, 16>();
  hip_matrix_fill<bfloat16, float, 32, 32, 16>();
  hip_matrix_fill<bfloat16, float, 16, 16, 32>();
  hip_matrix_fill<float, float, 32, 32, 2>();
  hip_matrix_fill<float, float, 16, 16, 4>();
  hip_matrix_fill<double, double, 16, 16, 4>();

  hip_matrix_apply<int8_t, int32_t, 32, 32, 16>();
  hip_matrix_apply<int8_t, int32_t, 16, 16, 32>();
  hip_matrix_apply<int8_t, int32_t, 32, 32, 32>();
  hip_matrix_apply<int8_t, int32_t, 16, 16, 64>();
  hip_matrix_apply<bfloat16, float, 32, 32, 8>();
  hip_matrix_apply<bfloat16, float, 16, 16, 16>();
  hip_matrix_apply<bfloat16, float, 32, 32, 16>();
  hip_matrix_apply<bfloat16, float, 16, 16, 32>();
  hip_matrix_apply<float, float, 32, 32, 2>();
  hip_matrix_apply<float, float, 16, 16, 4>();
  hip_matrix_apply<double, double, 16, 16, 4>();
}
