// REQUIRES: hip
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx942 %s -o compile-query-hip

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
  // Compile-time query to validate the matrix parameters
  using myparams = matrix_params<architecture::amd_gpu_gfx942, int8_t, int8_t,
                                 int32_t, int32_t, 32, 32, 16>;

  static_assert(myparams::M == 32);
  static_assert(myparams::N == 32);
  static_assert(myparams::K == 16);

  // Sizes-only compile-time query: types are given, generate default sizes
  using myparams2 = matrix_params<architecture::amd_gpu_gfx942, int8_t, int8_t,
                                  int32_t, int32_t>;
  static_assert(myparams2::M == 16);
  static_assert(myparams2::N == 16);
  static_assert(myparams2::K == 32);

  return 0;
};
