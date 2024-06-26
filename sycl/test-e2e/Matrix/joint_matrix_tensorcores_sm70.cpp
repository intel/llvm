//===---joint_matrix_tensorcores_sm70.cpp - DPC++ joint_matrix-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: cuda
// RUN: %{build} -Xsycl-target-backend --cuda-gpu-arch=sm_70 -o %t.out
// RUN: %{run} %t.out
//
// This tests the unified matrix extension interfaces for the cuda backend.
// This test must be compiled with -Xsycl-target-backend --cuda-gpu-arch=sm_xx,
// where sm_xx >= sm_70.
// For some devices it is important to use the sm version (Compute Capability)
// corresponding to the device that will run the program when specifying e.g.
// `-fsycl-targets=nvidia_gpu_sm_xx` during compilation. This particularly
// affects matrix operations using `half` such as those in this test. For more
// information on this issue consult
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-restrictions

#include "joint_matrix_apply_cuda.hpp"
#include "joint_matrix_gemm_cuda.hpp"

static constexpr size_t M = 16;
static constexpr size_t N = 16;
static constexpr size_t MATRIX_M = M * nWGperDim;
static constexpr size_t MATRIX_N = N * nWGperDim;

int main() {

  queue Q;
  auto computeCapability =
      std::stof(Q.get_device().get_info<sycl::info::device::backend_version>());

  if (computeCapability >= 7.0) {
    // A/B half, Accumulator float
    test<half, float, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>(
        Q);
    test<half, float, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>(
        Q);
    test<half, float, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>(
        Q);

    test<const half, const float, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         16, 16, 16>(Q);
    test<const half, const float, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         8, 16, 32>(Q);
    test<const half, const float, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         32, 16, 8>(Q);

    // A/B/Accumulator half
    test<half, half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>(
        Q);
    test<half, half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>(Q);
    test<half, half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>(Q);

    test<const half, const half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         16, 16, 16>(Q);
    test<const half, const half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8,
         16, 32>(Q);
    test<const half, const half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         32, 16, 8>(Q);

    // A/B/D half, C float
    test<half, float, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>(
        Q);
    test<half, float, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>(
        Q);
    test<half, float, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>(
        Q);

    test<const half, const float, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         16, 16, 16>(Q);
    test<const half, const float, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         8, 16, 32>(Q);
    test<const half, const float, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         32, 16, 8>(Q);

    // A/B/C half, D float
    test<half, half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>(
        Q);
    test<half, half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>(
        Q);
    test<half, half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>(
        Q);

    test<const half, const half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         16, 16, 16>(Q);
    test<const half, const half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         8, 16, 32>(Q);
    test<const half, const half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         32, 16, 8>(Q);

    // test different layout combinations for one case

    test<const half, const half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         32, 16, 8, layout::row_major, layout::row_major, layout::col_major>(Q);
    test<const half, const half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         32, 16, 8, layout::row_major, layout::col_major, layout::row_major>(Q);
    test<const half, const half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         32, 16, 8, layout::col_major, layout::row_major, layout::row_major>(Q);
    test<const half, const half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         32, 16, 8, layout::col_major, layout::col_major, layout::row_major>(Q);

    // joint_matrix_apply tests

    auto apply_add = [](auto &x) { x = x + 2; };
    float D[MATRIX_M][MATRIX_N];
    big_matrix<float, MATRIX_M, MATRIX_N> MD_f((float *)&D);

    matrix_verify_lambda<half, float, M, 16, N>(Q, MD_f, 0.0, apply_add);
  }

  return 0;
};
