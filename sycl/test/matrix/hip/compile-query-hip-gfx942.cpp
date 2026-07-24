// REQUIRES: hip
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx942 %s -o compile-query-hip

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::matrix;

// gfx940, gfx941 and gfx942 are CDNA3 parts and expose identical joint_matrix
// support, so the compile-time query must succeed for all three architectures.
template <architecture Arch> void check_arch() {
  // Compile-time query to validate the matrix parameters
  using myparams =
      matrix_params<Arch, int8_t, int8_t, int32_t, int32_t, 32, 32, 16>;
  using floatparams =
      matrix_params<Arch, float, float, float, float, 32, 32, 2>;

  static_assert(myparams::M == 32);
  static_assert(myparams::N == 32);
  static_assert(myparams::K == 16);
  static_assert(floatparams::M == 32);
  static_assert(floatparams::N == 32);
  static_assert(floatparams::K == 2);

  // Sizes-only compile-time query: types are given, generate default sizes
  using myparams2 = matrix_params<Arch, int8_t, int8_t, int32_t, int32_t>;
  using floatparams2 = matrix_params<Arch, float, float, float, float>;
  static_assert(myparams2::M == 16);
  static_assert(myparams2::N == 16);
  static_assert(myparams2::K == 32);
  static_assert(floatparams2::M == 16);
  static_assert(floatparams2::N == 16);
  static_assert(floatparams2::K == 4);
}

int main() {
  check_arch<architecture::amd_gpu_gfx940>();
  check_arch<architecture::amd_gpu_gfx941>();
  check_arch<architecture::amd_gpu_gfx942>();

  return 0;
};
