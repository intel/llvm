
// REQUIRES: cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_80 %s -o %t.out
// RUN: %t.out
//
// This tests the unified matrix extension interfaces for the cuda backend.
// This test must be compiled with -Xsycl-target-backend --cuda-gpu-arch=sm_xx,
// where sm_xx >= sm_80.

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

  if (computeCapability >= 8.0) {
    test<double, double, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 4, 8>(Q);
    test<const double, const double, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8,
         4, 8>(Q);

    test<bfloat16, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>(Q);
    test<bfloat16, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>(Q);
    test<bfloat16, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>(Q);

    test<const bfloat16, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16,
         16, 16>(Q);
    test<const bfloat16, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8,
         16, 32>(Q);
    test<const bfloat16, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32,
         16, 8>(Q);

    // A/B tf32
    test<float, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 8, 16,
         precision::tf32>(Q);
    test<const float, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 8,
         16, precision::tf32>(Q);

    float D[MATRIX_M][MATRIX_N];
    big_matrix<float, MATRIX_M, MATRIX_N> MD_f((float *)&D);

    double D_d[MATRIX_M / 2][MATRIX_N / 2];
    big_matrix<double, 8 * nWGperDim, 8 * nWGperDim> MD_d((double *)&D_d);
    auto apply_add = [](auto &x) { x = x + 2; };

    // joint_matrix_apply tests
    matrix_verify_lambda<bfloat16, float, 16, 16, 16>(Q, MD_f, 0.0, apply_add);

    matrix_verify_lambda<double, double, 8, 4, 8>(Q, MD_d, -60.0, apply_add);

    // get_wi_data() Deprecated

    matrix_verify_op<bfloat16, float, 16, 16, 16>(Q, MD_f, 0.0,
                                                  std::plus<bfloat16>{});
    matrix_verify_op<bfloat16, float, 16, 16, 16>(Q, MD_f, 0.0, Logical{});
    matrix_verify_op<bfloat16, float, 16, 16, 16>(Q, MD_f, 16.0,
                                                  std::multiplies<bfloat16>{});
    matrix_verify_op<bfloat16, float, 16, 16, 16>(Q, MD_f, -56.0,
                                                  std::divides<bfloat16>{});
    matrix_verify_op<bfloat16, float, 16, 16, 16>(Q, MD_f, -64.0,
                                                  std::minus<bfloat16>{});

    matrix_verify_op<double, double, 8, 4, 8>(Q, MD_d, -60.0,
                                              std::plus<double>{});
    matrix_verify_op<double, double, 8, 4, 8>(Q, MD_d, -60.0, Logical{});
    matrix_verify_op<double, double, 8, 4, 8>(Q, MD_d, -56.0,
                                              std::multiplies<double>{});
    matrix_verify_op<double, double, 8, 4, 8>(Q, MD_d, -74.0,
                                              std::divides<double>{});
    matrix_verify_op<double, double, 8, 4, 8>(Q, MD_d, -76.0,
                                              std::minus<double>{});
  }
  return 0;
};
