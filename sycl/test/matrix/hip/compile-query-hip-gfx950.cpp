// REQUIRES: hip
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx950 %s -o %t.out

#include <sycl/ext/oneapi/experimental/float_4bit/types.hpp>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::matrix;
using sycl::ext::oneapi::bfloat16;

constexpr auto Arch = architecture::amd_gpu_gfx950;

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K>
void check_valid() {
  using params = matrix_params<Arch, TA, TB, TC, TC, M, N, K>;
  static_assert(params::M == M && params::N == N && params::K == K);
  static_assert(std::is_same_v<
                typename params::template joint_matrix_a<sub_group,
                                                         layout::row_major>,
                joint_matrix<sub_group, TA, use::a, M, K, layout::row_major>>);
  static_assert(std::is_same_v<
                typename params::template joint_matrix_b<sub_group,
                                                         layout::col_major>,
                joint_matrix<sub_group, TB, use::b, K, N, layout::col_major>>);
}

template <typename TA, typename TB, typename TC, size_t K>
void check_default() {
  using params = matrix_params<Arch, TA, TB, TC, TC>;
  static_assert(params::M == 16 && params::N == 16 && params::K == K);
}

int main() {
  // gfx942 combinations.
  check_valid<int8_t, int8_t, int32_t, 32, 32, 16>();
  check_valid<int8_t, int8_t, int32_t, 16, 16, 32>();
  check_valid<half, half, float, 32, 32, 8>();
  check_valid<bfloat16, bfloat16, float, 16, 16, 16>();
  check_valid<float, float, float, 32, 32, 2>();
  check_valid<double, double, double, 16, 16, 4>();

  // Combinations added by gfx950.
  check_valid<int8_t, int8_t, int32_t, 32, 32, 32>();
  check_valid<int8_t, int8_t, int32_t, 16, 16, 64>();
  check_valid<half, half, float, 32, 32, 16>();
  check_valid<half, half, float, 16, 16, 32>();
  check_valid<bfloat16, bfloat16, float, 32, 32, 16>();
  check_valid<bfloat16, bfloat16, float, 16, 16, 32>();

  // FP8, including mixed formats, at the FP8 and F8F6F4 shapes.
  check_valid<fp8_e4m3, fp8_e4m3, float, 16, 16, 32>();
  check_valid<fp8_e4m3, fp8_e5m2, float, 32, 32, 16>();
  check_valid<fp8_e5m2, fp8_e4m3, float, 16, 16, 128>();
  check_valid<fp8_e5m2, fp8_e5m2, float, 32, 32, 64>();

  // FP4 only at the F8F6F4 shapes, alone or mixed with FP8.
  check_valid<fp4_e2m1, fp4_e2m1, float, 16, 16, 128>();
  check_valid<fp4_e2m1, fp8_e4m3, float, 32, 32, 64>();
  check_valid<fp8_e5m2, fp4_e2m1, float, 16, 16, 128>();

  // Default sizes use the largest K of the 16x16 shapes.
  check_default<int8_t, int8_t, int32_t, 64>();
  check_default<half, half, float, 32>();
  check_default<bfloat16, bfloat16, float, 32>();
  check_default<float, float, float, 4>();
  check_default<double, double, double, 4>();
  check_default<fp8_e4m3, fp8_e4m3, float, 128>();
  check_default<fp8_e5m2, fp4_e2m1, float, 128>();
  check_default<fp4_e2m1, fp4_e2m1, float, 128>();

  return 0;
}
