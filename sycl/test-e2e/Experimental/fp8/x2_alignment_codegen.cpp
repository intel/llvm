// RUN: %clangxx -fsycl -fsycl-device-only -DNDEBUG -S -Xclang -emit-llvm %s -o - | FileCheck %s

// UNSUPPORTED: target-nvidia, target-amd
// UNSUPPORTED-INTENDED: relies on SPIR-V FP8 conversion builtins

#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>

using namespace sycl::ext::oneapi::experimental;

// Encode: two halfs are converted and stored into the `vals` array through a
// single aligned `<2 x i8>` store.
//
// CHECK-LABEL: define {{.*}}encode_e4m3
// CHECK: store <2 x i8> {{%[a-zA-Z0-9._]+}}, ptr {{.*}}, align 2
SYCL_EXTERNAL void encode_e4m3(sycl::half a, sycl::half b, fp8_e4m3_x2 *out) {
  sycl::half in[2] = {a, b};
  *out = fp8_e4m3_x2(in);
}

// Decode: the two packed FP8 values are read through a single aligned
// `<2 x i8>` load.
//
// CHECK-LABEL: define {{.*}}decode_e4m3
// CHECK: load <2 x i8>, ptr {{.*}}, align 2
SYCL_EXTERNAL void decode_e4m3(const fp8_e4m3_x2 *in, sycl::half *out) {
  sycl::marray<sycl::half, 2> m = static_cast<sycl::marray<sycl::half, 2>>(*in);
  out[0] = m[0];
  out[1] = m[1];
}

// Same aligned vector store/load for the e5m2 variant.
//
// CHECK-LABEL: define {{.*}}encode_e5m2
// CHECK: store <2 x i8> {{%[a-zA-Z0-9._]+}}, ptr {{.*}}, align 2
SYCL_EXTERNAL void encode_e5m2(sycl::half a, sycl::half b, fp8_e5m2_x2 *out) {
  sycl::half in[2] = {a, b};
  *out = fp8_e5m2_x2(in);
}

// CHECK-LABEL: define {{.*}}decode_e5m2
// CHECK: load <2 x i8>, ptr {{.*}}, align 2
SYCL_EXTERNAL void decode_e5m2(const fp8_e5m2_x2 *in, sycl::half *out) {
  sycl::marray<sycl::half, 2> m = static_cast<sycl::marray<sycl::half, 2>>(*in);
  out[0] = m[0];
  out[1] = m[1];
}
