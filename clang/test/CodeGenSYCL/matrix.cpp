// RUN: %clang_cc1 -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// Test that SPIR-V codegen generates the expected LLVM struct name for the
// JointMatrixINTEL type.
#include <stddef.h>
#include <stdint.h>

namespace __spv {
  template <typename T, size_t R, size_t C, uint32_t L, uint32_t S, uint32_t U>
  struct __spirv_JointMatrixINTEL;
}

// CHECK: @_Z2f1{{.*}}(target("spirv.JointMatrixINTEL", float, 5, 10, 0, 1, 0)
void f1(__spv::__spirv_JointMatrixINTEL<float, 5, 10, 0, 1, 0> *matrix) {}

// CHECK: @_Z2f2{{.*}}(target("spirv.JointMatrixINTEL", i64, 10, 2, 0, 0, 0)
void f2(__spv::__spirv_JointMatrixINTEL<uint64_t, 10, 2, 0, 0, 0> *matrix) {}

// CHECK: @_Z2f3{{.*}}(target("spirv.JointMatrixINTEL", i8, 10, 2, 0, 0, 0)
void f3(__spv::__spirv_JointMatrixINTEL<char, 10, 2, 0, 0, 0> *matrix) {}

namespace sycl {
  class half {};
  class bfloat16 {};
  class tf32 {};
}
typedef sycl::half my_half;

// CHECK: @_Z2f4{{.*}}(target("spirv.JointMatrixINTEL", half, 10, 2, 0, 0, 0)
void f4(__spv::__spirv_JointMatrixINTEL<my_half, 10, 2, 0, 0, 0> *matrix) {}

// CHECK: @_Z2f5{{.*}}(target("spirv.JointMatrixINTEL", i16, 10, 2, 0, 0, 0, 1)
void f5(__spv::__spirv_JointMatrixINTEL<sycl::bfloat16, 10, 2, 0, 0, 0> *matrix) {}

// CHECK: @_Z2f6{{.*}}(target("spirv.JointMatrixINTEL", i128, 10, 2, 0, 0, 0)
void f6(__spv::__spirv_JointMatrixINTEL<_BitInt(128), 10, 2, 0, 0, 0> *matrix) {}

// CHECK: @_Z2f7{{.*}}(target("spirv.JointMatrixINTEL", float, 10, 2, 0, 0, 0, 0)
void f7(__spv::__spirv_JointMatrixINTEL<sycl::tf32, 10, 2, 0, 0, 0> *matrix) {}

// CHECK: @_Z2f8{{.*}}(target("spirv.JointMatrixINTEL", double, 5, 10, 0, 1, 0)
void f8(__spv::__spirv_JointMatrixINTEL<double, 5, 10, 0, 1, 0> *matrix) {}
