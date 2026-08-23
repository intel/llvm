// Check that the round-to-nearest-even saturating float->int8 conversions lower
// to the hardware `cvt.rni.sat.{s8,u8}.f32` instructions on NVPTX.
//
// REQUIRES: cuda
//
// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   -Xsycl-target-backend --cuda-gpu-arch=sm_90 -S -Xclang -emit-llvm %s \
// RUN:   -o - | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/saturating_conversion.hpp>

using namespace sycl::ext::oneapi::experimental;

// CHECK: call{{.*}}asm{{.*}}cvt.rni.sat.s8.f32
SYCL_EXTERNAL int8_t test_s8(float x) { return float_to_int8_rn(x); }

// CHECK: call{{.*}}asm{{.*}}cvt.rni.sat.u8.f32
SYCL_EXTERNAL uint8_t test_u8(float x) { return float_to_uint8_rn(x); }

// The packed variant applies the signed conversion to each of the four lanes.
// CHECK: call{{.*}}asm{{.*}}cvt.rni.sat.s8.f32
SYCL_EXTERNAL int32_t test_packed_s(sycl::vec<float, 4> v) {
  return float4_to_int8x4_rn(v);
}
