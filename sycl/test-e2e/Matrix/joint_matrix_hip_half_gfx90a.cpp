//===---joint_matrix_hip_half_gfx90a.cpp - DPC++ joint_matrix--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx90a %s -o %t.out
// RUN: %{run} %t.out

// REQUIRES: arch-amd_gpu_gfx90a
// REQUIRES: aspect-fp16
// REQUIRES: build-and-run-mode

#include "joint_matrix_hip_apply.hpp"
#include "joint_matrix_hip_copy.hpp"
#include "joint_matrix_hip_fill.hpp"
#include "joint_matrix_hip_mfma.hpp"

template <size_t KX> void half_matrix_mfma() {
  hip_matrix_mfma<sycl::half, float, 32, 32, 8, KX, layout::row_major>();
  hip_matrix_mfma<sycl::half, float, 16, 16, 16, KX, layout::row_major>();
  hip_matrix_mfma<sycl::half, float, 32, 32, 8, KX, layout::col_major>();
  hip_matrix_mfma<sycl::half, float, 16, 16, 16, KX, layout::col_major>();
}

int main() {
  half_matrix_mfma<1>();
  half_matrix_mfma<2>();
  half_matrix_mfma<3>();
  half_matrix_mfma<4>();

  hip_matrix_copy<sycl::half, float, 32, 32, 8, layout::row_major>();
  hip_matrix_copy<sycl::half, float, 16, 16, 16, layout::row_major>();
  hip_matrix_copy<sycl::half, float, 32, 32, 8, layout::col_major>();
  hip_matrix_copy<sycl::half, float, 16, 16, 16, layout::col_major>();

  hip_matrix_fill<sycl::half, float, 32, 32, 8>();
  hip_matrix_fill<sycl::half, float, 16, 16, 16>();

  hip_matrix_apply<sycl::half, float, 32, 32, 8>();
  hip_matrix_apply<sycl::half, float, 16, 16, 16>();
}
