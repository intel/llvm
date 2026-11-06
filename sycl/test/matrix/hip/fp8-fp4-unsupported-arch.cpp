// REQUIRES: hip
// The amd_arch lit parameter overrides the architectures of the RUN lines.
// UNSUPPORTED: hip-arch-gfx950
// RUN: %clangxx -fsycl -fsycl-device-only -fsycl-targets=amd_gpu_gfx942 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsycl-targets=amd_gpu_gfx90a -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// The FP8/FP4 joint_matrix multiplicands are only implemented for gfx950: the
// gfx94x FP8 MFMAs use the FNUZ encodings, which differ from the OCP ones of
// sycl_ext_oneapi_fp8.

#include <sycl/ext/oneapi/experimental/float_4bit/types.hpp>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::matrix;

template <typename TA, typename TB, size_t M, size_t N, size_t K>
void mad(sub_group sg) {
  joint_matrix<sub_group, float, use::accumulator, M, N> sub_c;
  joint_matrix<sub_group, TA, use::a, M, K, layout::row_major> sub_a;
  joint_matrix<sub_group, TB, use::b, K, N, layout::row_major> sub_b;
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
}

SYCL_EXTERNAL void fp8(nd_item<1> item) {
  // expected-error@sycl/ext/oneapi/matrix/matrix-hip.hpp:* {{joint_matrix with fp8_e4m3, fp8_e5m2 or fp4_e2m1 multiplicands is only supported on AMD gfx950}}
  mad<fp8_e4m3, fp8_e5m2, 16, 16, 32>(item.get_sub_group());
}

SYCL_EXTERNAL void fp4(nd_item<1> item) {
  // expected-error@sycl/ext/oneapi/matrix/matrix-hip.hpp:* {{joint_matrix with fp8_e4m3, fp8_e5m2 or fp4_e2m1 multiplicands is only supported on AMD gfx950}}
  mad<fp4_e2m1, fp4_e2m1, 16, 16, 128>(item.get_sub_group());
}
