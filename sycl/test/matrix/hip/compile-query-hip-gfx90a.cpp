// REQUIRES: hip
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx90a %s -o compile-query-hip

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
  // Compile-time query to validate the matrix parameters
  using myparams = matrix_params<architecture::amd_gpu_gfx90a, int8_t, int8_t,
                                 int32_t, int32_t, 32, 32, 8>;

  static_assert(myparams::M == 32);
  static_assert(myparams::N == 32);
  static_assert(myparams::K == 8);

  // Sizes-only compile-time query: types are given, generate default sizes
  using myparams2 = matrix_params<architecture::amd_gpu_gfx90a, int8_t, int8_t,
                                  int32_t, int32_t>;
  static_assert(myparams2::M == 16);
  static_assert(myparams2::N == 16);
  static_assert(myparams2::K == 4);

  // Validate the fp32 (float/float) MFMA combinations supported on gfx90a.
  using myparams_f32 =
      matrix_params<architecture::amd_gpu_gfx90a, float, float, float, float,
                    16, 16, 4>;
  static_assert(myparams_f32::M == 16);
  static_assert(myparams_f32::N == 16);
  static_assert(myparams_f32::K == 4);

  using myparams_f32_2 =
      matrix_params<architecture::amd_gpu_gfx90a, float, float, float, float,
                    32, 32, 2>;
  static_assert(myparams_f32_2::M == 32);
  static_assert(myparams_f32_2::N == 32);
  static_assert(myparams_f32_2::K == 2);

  // Sizes-only compile-time query for float: default float shape is 16x16x4.
  using myparams_f32_default =
      matrix_params<architecture::amd_gpu_gfx90a, float, float, float, float>;
  static_assert(myparams_f32_default::M == 16);
  static_assert(myparams_f32_default::N == 16);
  static_assert(myparams_f32_default::K == 4);

  return 0;
};
