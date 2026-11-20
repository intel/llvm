//===---joint_matrix_hip_half.cpp - DPC++ joint_matrix---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa %amd_arch_options %s -o %t.out
// RUN: %{run} %t.out

// REQUIRES: target-amd
// REQUIRES: aspect-fp16

// mfma (K tiling 1-4, row/col-major output), copy, fill and apply tests of the
// half joint_matrix shapes supported by the AMD GPU the test is compiled for.

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
#if __SYCL_TARGET_AMD_GPU_GFX90A__ || __SYCL_TARGET_AMD_GPU_GFX940__ ||        \
    __SYCL_TARGET_AMD_GPU_GFX941__ || __SYCL_TARGET_AMD_GPU_GFX942__ ||        \
    defined(CDNA4)
#define CDNA2_OR_LATER 1
#endif

template <size_t M, size_t N, size_t K> void test() {
  hip_matrix_mfma<sycl::half, float, M, N, K, 1, layout::row_major>();
  hip_matrix_mfma<sycl::half, float, M, N, K, 2, layout::row_major>();
  hip_matrix_mfma<sycl::half, float, M, N, K, 3, layout::row_major>();
  hip_matrix_mfma<sycl::half, float, M, N, K, 4, layout::row_major>();
  hip_matrix_mfma<sycl::half, float, M, N, K, 1, layout::col_major>();
  hip_matrix_mfma<sycl::half, float, M, N, K, 2, layout::col_major>();
  hip_matrix_mfma<sycl::half, float, M, N, K, 3, layout::col_major>();
  hip_matrix_mfma<sycl::half, float, M, N, K, 4, layout::col_major>();
  hip_matrix_copy<sycl::half, float, M, N, K, layout::row_major>();
  hip_matrix_copy<sycl::half, float, M, N, K, layout::col_major>();
  hip_matrix_fill<sycl::half, float, M, N, K>();
  hip_matrix_apply<sycl::half, float, M, N, K>();
}

int main() {
#if defined(CDNA2_OR_LATER)
  test<32, 32, 8>();
  test<16, 16, 16>();
#endif
#if defined(CDNA4)
  test<32, 32, 16>();
  test<16, 16, 32>();
#endif
  return 0;
}
