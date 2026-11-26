// REQUIRES: hip
// RUN: %clangxx -fsycl -fsycl-device-only -fsycl-targets=amd_gpu_gfx950 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsycl-targets=amd_gpu_gfx90a -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// FP4 joint_matrix elements are the packed fp4_e2m1_x<2>, two 4-bit values
// per byte, so that joint_matrix_load and joint_matrix_apply see whole bytes.
// The unpacked fp4_e2m1_x<1> must be rejected.

#include <sycl/ext/oneapi/experimental/float_4bit/types.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::matrix;

SYCL_EXTERNAL void fp4_unpacked(nd_item<1> item) {
  // expected-error@sycl/ext/oneapi/matrix/matrix-hip.hpp:* {{joint_matrix supports fp4_e2m1 only in the packed fp4_e2m1_x<2> form}}
  joint_matrix<sub_group, fp4_e2m1_x<1>, use::a, 16, 128, layout::row_major> m;
  (void)m;
}
