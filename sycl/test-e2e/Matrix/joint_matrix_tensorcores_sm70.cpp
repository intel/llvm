
// REQUIRES: cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_70 %s -o %t.out
// RUN: %t.out
//
// This tests the unified matrix extension interfaces for the cuda backend.
// This test must be compiled with -Xsycl-target-backend --cuda-gpu-arch=sm_xx, where sm_xx >= sm_70.

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
    test<half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>(Q);
    test<half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>(Q);
    test<half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>(Q);

    test<const half, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16,
         16>(Q);
    test<const half, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16,
         32>(Q);
    test<const half, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16,
         8>(Q);

    // A/B/Accumulator half
    test<half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>(Q);
    test<half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>(Q);
    test<half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>(Q);

    test<const half, const half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16,
         16>(Q);
    test<const half, const half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16,
         32>(Q);
    test<const half, const half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16,
         8>(Q);

    auto apply_add = [](auto &x) { x = x + 2; };
    float D[MATRIX_M][MATRIX_N];
    big_matrix<float, MATRIX_M, MATRIX_N> MD_f((float *)&D);

    // joint_matrix_apply tests

    matrix_verify_lambda<half, float, M, 16, N>(Q, MD_f, 0.0, apply_add);

    // get_wi_data() Deprecated tests

    matrix_verify_op<half, float, M, 16, N>(Q, MD_f, 0.0, std::plus<half>{});
    matrix_verify_op<half, float, M, 16, N>(Q, MD_f, 0.0, Logical{});
    matrix_verify_op<half, float, M, 16, N>(Q, MD_f, 16.0,
                                            std::multiplies<half>{});
    matrix_verify_op<half, float, M, 16, N>(Q, MD_f, -56.0,
                                            std::divides<half>{});
    matrix_verify_op<half, float, M, 16, N>(Q, MD_f, -64.0, std::minus<half>{});
  }

  return 0;
};
