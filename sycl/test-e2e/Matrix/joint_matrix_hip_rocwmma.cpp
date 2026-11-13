//===---joint_matrix_hip_rocwmma.cpp - DPC++ joint_matrix------------------===//
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

// rocWMMA-style GEMM and load/store contamination tests of the int8, half,
// bfloat16, float and double joint_matrix shapes supported by the AMD GPU the
// test is compiled for.

#include "Inputs/joint_matrix_hip_rocwmma.hpp"

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
  hip_rocwmma_gemm<TA, TA, TC, M, N, K>();
  hip_rocwmma_contamination<TA, TA, TC, M, N, K>();
}

int main() {
#if defined(CDNA2) || defined(CDNA3_OR_LATER)
  test<half, float, 32, 32, 8>();
  test<half, float, 16, 16, 16>();
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
  test<half, float, 32, 32, 16>();
  test<half, float, 16, 16, 32>();
  test<bfloat16, float, 32, 32, 16>();
  test<bfloat16, float, 16, 16, 32>();
#endif
  return 0;
}
