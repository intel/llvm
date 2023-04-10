//==-------------- vec_arg_call_conv_ext.cpp  - DPC++ ESIMD feature test ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Intel GPU is not really required, but the test does not make sense for
// others.
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -Xclang -opaque-pointers -fsycl-device-only -Xclang -emit-llvm -o %t.comp.ll %s
// RUN: sycl-post-link -ir-output-only -lower-esimd -S %t.comp.ll -o %t.out.ll
// RUN: FileCheck --input-file=%t.out.ll %s

// Checks that ESIMDOptimizeVecArgCallConv does the right job as
// a part of sycl-post-link.

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;

// clang-format off

//------------------------
// Test1: Optimized parameter interleaves non - optimizeable ones.

__attribute__((noinline))
SYCL_EXTERNAL simd<int, 8> callee__sret__x_param_x(int i, simd<int, 8> x, int j) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func <8 x i32> @_Z23callee__sret__x_param_x{{.*}}(i32 noundef %{{.*}}, <8 x i32> %{{.*}}, i32 noundef %{{.*}})
  return x + (i + j);
}

__attribute__((noinline))
SYCL_EXTERNAL simd<int, 8> test__sret__x_param_x(simd<int, 8> x) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func <8 x i32> @_Z21test__sret__x_param_x{{.*}}(<8 x i32> %{{.*}})
  return callee__sret__x_param_x(2, x, 1);
// CHECK:  %{{.*}} = call spir_func <8 x i32> @_Z23callee__sret__x_param_x{{.*}}(i32 2, <8 x i32> %{{.*}}, i32 1)
}

//------------------------
// Test2: "2-level fall through"

__attribute__((noinline))
SYCL_EXTERNAL simd<double, 32> callee__all_fall_through0(simd<double, 32> x) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func <32 x double> @_Z25callee__all_fall_through0{{.*}}(<32 x double> %{{.*}})
  return x;
}

__attribute__((noinline))
SYCL_EXTERNAL simd<double, 32> callee__all_fall_through1(simd<double, 32> x) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func <32 x double> @_Z25callee__all_fall_through1{{.*}}(<32 x double> %{{.*}})
  return callee__all_fall_through0(x);
// CHECK:  %{{.*}} = call spir_func <32 x double> @_Z25callee__all_fall_through0{{.*}}(<32 x double> %{{.*}})
}

__attribute__((noinline))
SYCL_EXTERNAL simd<double, 32> test__all_fall_through(simd<double, 32> x) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func <32 x double> @_Z22test__all_fall_through{{.*}}(<32 x double> %{{.*}})
  return callee__all_fall_through1(x);
// CHECK:  %{{.*}} = call spir_func <32 x double> @_Z25callee__all_fall_through1{{.*}}(<32 x double> %{{.*}})
}

//------------------------
// Test3. First argument is passed by reference and updated in the callee,
// must not be optimized.

__attribute__((noinline))
SYCL_EXTERNAL void callee_void__noopt_opt(simd<int, 8>& x, simd<int, 8> y) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func void @_Z22callee_void__noopt_opt{{.*}}(ptr {{.*}} %{{.*}}, <8 x i32> %{{.*}})
  x = x + y;
}

__attribute__((noinline))
SYCL_EXTERNAL simd<int, 8> test__sret__noopt_opt(simd<int, 8> x) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func <8 x i32> @_Z21test__sret__noopt_opt{{.*}}(ptr noundef %{{.*}})
  callee_void__noopt_opt(x, x);
// CHECK:  call spir_func void @_Z22callee_void__noopt_opt{{.*}}(ptr addrspace(4) %{{.*}}, <8 x i32> %{{.*}})
  return x;
}

//------------------------

// Dummy main to satisfy linker.
int main() {
  return 0;
}
