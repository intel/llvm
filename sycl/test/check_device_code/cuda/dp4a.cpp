// Check that the sycl_ext_oneapi_dot_accumulate extension (dot_acc / dp4a)
// lowers to the hardware `dp4a` instruction on NVPTX (sm_61+).
//
// REQUIRES: cuda
//
// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   -Xsycl-target-backend --cuda-gpu-arch=sm_90 -S -Xclang -emit-llvm %s \
// RUN:   -o - | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/dot_product.hpp>

using namespace sycl::ext::oneapi;

// CHECK: dp4a.s32.s32
SYCL_EXTERNAL int32_t test_ss(int32_t a, int32_t b, int32_t c) {
  return dot_acc(a, b, c);
}

// CHECK: dp4a.u32.u32
SYCL_EXTERNAL int32_t test_uu(uint32_t a, uint32_t b, int32_t c) {
  return dot_acc(a, b, c);
}

// CHECK: dp4a.s32.u32
SYCL_EXTERNAL int32_t test_su(int32_t a, uint32_t b, int32_t c) {
  return dot_acc(a, b, c);
}

// CHECK: dp4a.u32.s32
SYCL_EXTERNAL int32_t test_us(uint32_t a, int32_t b, int32_t c) {
  return dot_acc(a, b, c);
}

// CHECK: dp4a.s32.s32
SYCL_EXTERNAL int32_t test_vec(sycl::vec<int8_t, 4> a, sycl::vec<int8_t, 4> b,
                               int32_t c) {
  return dot_acc(a, b, c);
}
