// RUN: %clang_cc1 -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// Test that SPIR-V codegen generates the expected LLVM struct name for the
// CooperativeMatrixKHR type.
#include <stddef.h>
#include <stdint.h>

namespace __spv {
  template <typename T, uint32_t S, size_t R, size_t C, uint32_t U>
  struct __spirv_CooperativeMatrixKHR;
}

// CHECK: @_Z2f1{{.*}}(target("spirv.CooperativeMatrixKHR", float, 3, 5, 10, 0)
void f1(__spv::__spirv_CooperativeMatrixKHR<float, 3, 5, 10, 0> *matrix) {}

// CHECK: @_Z2f2{{.*}}(target("spirv.CooperativeMatrixKHR", i64, 3, 10, 2, 1)
void f2(__spv::__spirv_CooperativeMatrixKHR<uint64_t, 3, 10, 2, 1> *matrix) {}

// CHECK: @_Z2f3{{.*}}(target("spirv.CooperativeMatrixKHR", i8, 3, 10, 2, 2)
void f3(__spv::__spirv_CooperativeMatrixKHR<char, 3, 10, 2, 2> *matrix) {}

namespace sycl {
  class half {};
  class bfloat16 {};
  class tf32 {};
}
typedef sycl::half my_half;

// CHECK: @_Z2f4{{.*}}(target("spirv.CooperativeMatrixKHR", half, 3, 10, 2, 0)
void f4(__spv::__spirv_CooperativeMatrixKHR<my_half, 3, 10, 2, 0> *matrix) {}

// CHECK: @_Z2f5{{.*}}(target("spirv.CooperativeMatrixKHR", i16, 3, 10, 2, 0)
void f5(__spv::__spirv_CooperativeMatrixKHR<sycl::bfloat16, 3, 10, 2, 0> *matrix) {}

// CHECK: @_Z2f6{{.*}}(target("spirv.CooperativeMatrixKHR", i128, 3, 10, 2, 0)
void f6(__spv::__spirv_CooperativeMatrixKHR<_BitInt(128), 3, 10, 2, 0> *matrix) {}

// CHECK: @_Z2f7{{.*}}(target("spirv.CooperativeMatrixKHR", float, 3, 10, 2, 0)
void f7(__spv::__spirv_CooperativeMatrixKHR<sycl::tf32, 3, 10, 2, 0> *matrix) {}

// CHECK: @_Z2f8{{.*}}(target("spirv.CooperativeMatrixKHR", double, 3, 5, 10, 0)
void f8(__spv::__spirv_CooperativeMatrixKHR<double, 3, 5, 10, 0> *matrix) {}
