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
  namespace ext::oneapi::experimental {
    template <size_t N> class fp8_e4m3_x { uint8_t vals[N]; };
    template <size_t N> class fp8_e5m2_x { uint8_t vals[N]; };
    template <size_t N> class fp8_e8m0_x { uint8_t vals[N]; };
    template <size_t N> class fp4_e2m1_x { uint8_t vals[(N + 1) / 2]; };
  }
}
namespace syclex = sycl::ext::oneapi::experimental;
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

// CHECK: @_Z2f9{{.*}}(target("spirv.CooperativeMatrixKHR", i8, 3, 8, 32, 0)
void f9(__spv::__spirv_CooperativeMatrixKHR<syclex::fp8_e4m3_x<1>, 3, 8, 32, 0> *matrix) {}

// CHECK: @_Z3f10{{.*}}(target("spirv.CooperativeMatrixKHR", i8, 3, 32, 16, 1)
void f10(__spv::__spirv_CooperativeMatrixKHR<syclex::fp8_e5m2_x<1>, 3, 32, 16, 1> *matrix) {}

// CHECK: @_Z3f11{{.*}}(target("spirv.CooperativeMatrixKHR", i8, 3, 8, 1, 0)
void f11(__spv::__spirv_CooperativeMatrixKHR<syclex::fp8_e8m0_x<1>, 3, 8, 1, 0> *matrix) {}

// CHECK: @_Z3f12{{.*}}(target("spirv.CooperativeMatrixKHR", i4, 3, 8, 64, 0)
void f12(__spv::__spirv_CooperativeMatrixKHR<syclex::fp4_e2m1_x<2>, 3, 8, 64, 0> *matrix) {}

// A second specialization of the same template gets a uniqued struct name.
// CHECK: @_Z3f13{{.*}}(target("spirv.CooperativeMatrixKHR", i8, 3, 8, 32, 0)
void f13(__spv::__spirv_CooperativeMatrixKHR<syclex::fp8_e4m3_x<2>, 3, 8, 32, 0> *matrix) {}

// CHECK: @_Z3f14{{.*}}(target("spirv.CooperativeMatrixKHR", i4, 3, 64, 16, 1)
void f14(__spv::__spirv_CooperativeMatrixKHR<syclex::fp4_e2m1_x<1>, 3, 64, 16, 1> *matrix) {}
