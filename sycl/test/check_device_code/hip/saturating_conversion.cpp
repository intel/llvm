// Check that the round-to-nearest-even saturating float->int8 conversions lower
// to a portable round + clamp + convert sequence on AMDGCN -- there is no
// single-instruction signed f32->sat.s8 there, so no inline asm / target
// intrinsic is used. We check for `llvm.rint` (round to nearest even),
// `llvm.maxnum`/`llvm.minnum` (saturate) and `fptosi` (convert), which the
// AMDGCN backend selects to `v_rndne_f32` + `v_max_f32`/`v_min_f32` +
// `v_cvt_i32_f32`.
//
// REQUIRES: hip
//
// RUN: %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa \
// RUN:   -Xsycl-target-backend --offload-arch=gfx90a -S -Xclang -emit-llvm %s \
// RUN:   -o - | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/saturating_conversion.hpp>

using namespace sycl::ext::oneapi::experimental;

// CHECK-LABEL: @_Z7test_s8f
// CHECK: call float @llvm.rint.f32
// CHECK: call{{.*}}float @llvm.maxnum.f32(float %{{.*}}, float -1.280000e+02)
// CHECK: call{{.*}}float @llvm.minnum.f32(float %{{.*}}, float 1.270000e+02)
// CHECK: fptosi float %{{.*}} to i32
SYCL_EXTERNAL int8_t test_s8(float x) { return float_to_int8_rn(x); }

// CHECK-LABEL: @_Z7test_u8f
// CHECK: call float @llvm.rint.f32
// CHECK: call{{.*}}float @llvm.maxnum.f32(float %{{.*}}, float 0.000000e+00)
// CHECK: call{{.*}}float @llvm.minnum.f32(float %{{.*}}, float 2.550000e+02)
// CHECK: fptosi float %{{.*}} to i32
SYCL_EXTERNAL uint8_t test_u8(float x) { return float_to_uint8_rn(x); }
