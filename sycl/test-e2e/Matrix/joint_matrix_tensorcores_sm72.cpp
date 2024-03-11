
// REQUIRES: cuda
// RUN: %{build} -Xsycl-target-backend --cuda-gpu-arch=sm_72 -o %t.out
// RUN: %{run} %t.out
//
// This tests the unified matrix extension interfaces for the cuda backend.
// This test must be compiled with -Xsycl-target-backend --cuda-gpu-arch=sm_xx,
// where sm_xx >= sm_72.

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

  if (computeCapability >= 7.2) {
    test<int8_t, int32_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16,
         16, 16>(Q);
    test<int8_t, int32_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16,
         32>(Q);
    test<int8_t, int32_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32,
         16, 8>(Q);

    test<const int8_t, const int32_t, int32_t, SUB_TILES_M, SUB_TILES_K,
         SUB_TILES_N, 16, 16, 16>(Q);
    test<const int8_t, const int32_t, int32_t, SUB_TILES_M, SUB_TILES_K,
         SUB_TILES_N, 8, 16, 32>(Q);
    test<const int8_t, const int32_t, int32_t, SUB_TILES_M, SUB_TILES_K,
         SUB_TILES_N, 32, 16, 8>(Q);

    test<uint8_t, int32_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16,
         16, 16>(Q);
    test<uint8_t, int32_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8,
         16, 32>(Q);
    test<uint8_t, int32_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32,
         16, 8>(Q);

    test<const uint8_t, const int32_t, int32_t, SUB_TILES_M, SUB_TILES_K,
         SUB_TILES_N, 16, 16, 16>(Q);
    test<const uint8_t, const int32_t, int32_t, SUB_TILES_M, SUB_TILES_K,
         SUB_TILES_N, 8, 16, 32>(Q);
    test<const uint8_t, const int32_t, int32_t, SUB_TILES_M, SUB_TILES_K,
         SUB_TILES_N, 32, 16, 8>(Q);

    // test different layout combinations for one case

    test<const uint8_t, const int32_t, int32_t, SUB_TILES_M, SUB_TILES_K,
         SUB_TILES_N, 32, 16, 8, layout::row_major, layout::row_major,
         layout::col_major>(Q);
    test<const uint8_t, const int32_t, int32_t, SUB_TILES_M, SUB_TILES_K,
         SUB_TILES_N, 32, 16, 8, layout::col_major, layout::row_major,
         layout::row_major>(Q);
    test<const uint8_t, const int32_t, int32_t, SUB_TILES_M, SUB_TILES_K,
         SUB_TILES_N, 32, 16, 8, layout::row_major, layout::col_major,
         layout::row_major>(Q);
    test<const uint8_t, const int32_t, int32_t, SUB_TILES_M, SUB_TILES_K,
         SUB_TILES_N, 32, 16, 8, layout::col_major, layout::col_major,
         layout::row_major>(Q);

    // joint_matrix_apply tests

    auto apply_add = [](auto &x) { x = x + 2; };

    int32_t D_i[MATRIX_M][MATRIX_N];
    big_matrix<int32_t, MATRIX_M, MATRIX_N> MD_i((int32_t *)&D_i);

    matrix_verify_lambda<uint8_t, int32_t, M, 16, N>(Q, MD_i, 0, apply_add);
    matrix_verify_lambda<int8_t, int32_t, M, 16, N>(Q, MD_i, 0, apply_add);
  }

  return 0;
};
