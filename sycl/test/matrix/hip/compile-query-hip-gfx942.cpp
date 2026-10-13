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

  // fp8 (E4M3) / bf8 (E5M2) with fp32 accumulator, including mixed A/B formats.
  using fp8params =
      matrix_params<Arch, fp8_e4m3, fp8_e4m3, float, float, 16, 16, 32>;
  using bf8params =
      matrix_params<Arch, fp8_e5m2, fp8_e5m2, float, float, 32, 32, 16>;
  using mixed8params =
      matrix_params<Arch, fp8_e4m3, fp8_e5m2, float, float, 16, 16, 32>;
  static_assert(fp8params::M == 16 && fp8params::N == 16 && fp8params::K == 32);
  static_assert(bf8params::M == 32 && bf8params::N == 32 && bf8params::K == 16);
  static_assert(mixed8params::K == 32);

  // Sizes-only query for fp8: default square shape with K == 32.
  using fp8params2 = matrix_params<Arch, fp8_e4m3, fp8_e4m3, float, float>;
  using mixed8params2 = matrix_params<Arch, fp8_e5m2, fp8_e4m3, float, float>;
  static_assert(fp8params2::M == 16 && fp8params2::N == 16 &&
                fp8params2::K == 32);
  static_assert(mixed8params2::M == 16 && mixed8params2::N == 16 &&
                mixed8params2::K == 32);
}

int main() {
  check_arch<architecture::amd_gpu_gfx940>();
  check_arch<architecture::amd_gpu_gfx941>();
  check_arch<architecture::amd_gpu_gfx942>();

  return 0;
};
