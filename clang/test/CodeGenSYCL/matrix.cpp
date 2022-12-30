// RUN: %clang_cc1 -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - -no-opaque-pointers | FileCheck %s
// Test that SPIR-V codegen generates the expected LLVM struct name for the
// JointMatrixINTEL type.
#include <stddef.h>
#include <stdint.h>

namespace __spv {
  template <typename T, size_t R, size_t C, uint32_t U, uint32_t S>
  struct __spirv_JointMatrixINTEL;
}

// CHECK: @_Z2f1{{.*}}(%spirv.JointMatrixINTEL._float_5_10_0_1
void f1(__spv::__spirv_JointMatrixINTEL<float, 5, 10, 0, 1> *matrix) {}

// CHECK: @_Z2f2{{.*}}(%spirv.JointMatrixINTEL._long_10_2_0_0
void f2(__spv::__spirv_JointMatrixINTEL<uint64_t, 10, 2, 0, 0> *matrix) {}

// CHECK: @_Z2f3{{.*}}(%spirv.JointMatrixINTEL._char_10_2_0_0
void f3(__spv::__spirv_JointMatrixINTEL<char, 10, 2, 0, 0> *matrix) {}

namespace sycl {
  class half {};
  class bfloat16 {};
  class tf32 {};
}
typedef sycl::half my_half;

// CHECK: @_Z2f4{{.*}}(%spirv.JointMatrixINTEL._half_10_2_0_0
void f4(__spv::__spirv_JointMatrixINTEL<my_half, 10, 2, 0, 0> *matrix) {}

// CHECK: @_Z2f5{{.*}}(%spirv.JointMatrixINTEL._bfloat16_10_2_0_0
void f5(__spv::__spirv_JointMatrixINTEL<sycl::bfloat16, 10, 2, 0, 0> *matrix) {}

// CHECK: @_Z2f6{{.*}}(%spirv.JointMatrixINTEL._i128_10_2_0_0
void f6(__spv::__spirv_JointMatrixINTEL<_BitInt(128), 10, 2, 0, 0> *matrix) {}

// CHECK: @_Z2f7{{.*}}(%spirv.JointMatrixINTEL._tf32_10_2_0_0
void f7(__spv::__spirv_JointMatrixINTEL<sycl::tf32, 10, 2, 0, 0> *matrix) {}

// CHECK: @_Z2f8{{.*}}(%spirv.JointMatrixINTEL._double_5_10_0_1
void f8(__spv::__spirv_JointMatrixINTEL<double, 5, 10, 0, 1> *matrix) {}
