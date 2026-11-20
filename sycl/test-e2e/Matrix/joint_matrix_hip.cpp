//===---joint_matrix_hip.cpp - DPC++ joint_matrix--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa %amd_arch_options %s -o %t.out
// RUN: %{run} %t.out

// REQUIRES: target-amd

// mfma (K tiling 1-4, row/col-major output), copy, fill and apply tests of the
// int8, bfloat16, float and double joint_matrix shapes supported by the AMD GPU
// the test is compiled for.

#include "joint_matrix_hip_apply.hpp"
#include "joint_matrix_hip_copy.hpp"
#include "joint_matrix_hip_fill.hpp"
#include "joint_matrix_hip_mfma.hpp"

// The target macros are set to 1 in both the host and the device compilation,
// so the host only runs the shapes compiled for the device. sycl.hpp defines
// the other target macros as 0, so test their values rather than defined().
#if __SYCL_TARGET_AMD_GPU_GFX950__
#define CDNA4 1
#endif
#if __SYCL_TARGET_AMD_GPU_GFX940__ || __SYCL_TARGET_AMD_GPU_GFX941__ ||        \
    __SYCL_TARGET_AMD_GPU_GFX942__ || defined(CDNA4)
#define CDNA3_OR_LATER 1
#endif
#if __SYCL_TARGET_AMD_GPU_GFX90A__
#define CDNA2 1
#endif

template <typename TA, typename TC, size_t M, size_t N, size_t K> void test() {
  hip_matrix_mfma<TA, TC, M, N, K, 1, layout::row_major>();
  hip_matrix_mfma<TA, TC, M, N, K, 2, layout::row_major>();
  hip_matrix_mfma<TA, TC, M, N, K, 3, layout::row_major>();
  hip_matrix_mfma<TA, TC, M, N, K, 4, layout::row_major>();
  hip_matrix_mfma<TA, TC, M, N, K, 1, layout::col_major>();
  hip_matrix_mfma<TA, TC, M, N, K, 2, layout::col_major>();
  hip_matrix_mfma<TA, TC, M, N, K, 3, layout::col_major>();
  hip_matrix_mfma<TA, TC, M, N, K, 4, layout::col_major>();
  hip_matrix_copy<TA, TC, M, N, K, layout::row_major>();
  hip_matrix_copy<TA, TC, M, N, K, layout::col_major>();
  hip_matrix_fill<TA, TC, M, N, K>();
  hip_matrix_apply<TA, TC, M, N, K>();
}

int main() {
  // No CDNA MFMA accumulator has M != N. Check the col-major index on
  // rectangles before launching the square device tiles.
  check_hip_col_major_rectangles();

#if defined(CDNA2) || defined(CDNA3_OR_LATER)
  test<bfloat16, float, 32, 32, 8>();
  test<bfloat16, float, 16, 16, 16>();
  test<double, double, 16, 16, 4>();
#endif
#if defined(CDNA2)
  test<int8_t, int32_t, 32, 32, 8>();
  test<int8_t, int32_t, 16, 16, 16>();
#endif
#if defined(CDNA3_OR_LATER)
  test<int8_t, int32_t, 32, 32, 16>();
  test<int8_t, int32_t, 16, 16, 32>();
  test<float, float, 32, 32, 2>();
  test<float, float, 16, 16, 4>();
#endif
#if defined(CDNA4)
  test<int8_t, int32_t, 32, 32, 32>();
  test<int8_t, int32_t, 16, 16, 64>();
  test<bfloat16, float, 32, 32, 16>();
  test<bfloat16, float, 16, 16, 32>();
#endif
  return 0;
}
