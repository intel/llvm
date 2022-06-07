// RUN: %clangxx -I %sycl_include -S -emit-llvm -fsycl-device-only %s -o - -Xclang -disable-llvm-passes -Xclang -no-enable-noundef-analysis | FileCheck %s
//
//==---- ap_fixed.cpp - SYCL FPGA arbitrary precision fixed point test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/__spirv/spirv_ops.hpp"

template <int W, int rW, bool S, int I, int rI>
void sqrt() {
  sycl::detail::ap_int<W> a;
  auto ap_fixed_Sqrt = __spirv_FixedSqrtINTEL<W, rW>(a, S, I, rI);
  // CHECK: %{{.*}} = call spir_func signext i5 @_Z[[#]]__spirv_FixedSqrtINTEL{{.*}}(i13 signext  %[[#]], i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
}

template <int W, int rW, bool S, int I, int rI>
void recip() {
  sycl::detail::ap_int<W> a;
  auto ap_fixed_Recip = __spirv_FixedRecipINTEL<W, rW>(a, S, I, rI);
  // CHECK: %{{.*}} = call spir_func signext i8 @_Z[[#]]__spirv_FixedRecipINTEL{{.*}}(i3 signext %[[#]], i1 zeroext true, i32 4, i32 4, i32 0, i32 0)
}

template <int W, int rW, bool S, int I, int rI>
void rsqrt() {
  sycl::detail::ap_int<W> a;
  auto ap_fixed_Rsqrt = __spirv_FixedRsqrtINTEL<W, rW>(a, S, I, rI);
  // CHECK: %{{.*}} = call spir_func signext i10 @_Z[[#]]__spirv_FixedRsqrtINTEL{{.*}}(i11 signext %[[#]], i1 zeroext false, i32 8, i32 6, i32 0, i32 0)
}

template <int W, int rW, bool S, int I, int rI>
void sin() {
  sycl::detail::ap_int<W> a;
  auto ap_fixed_Sin = __spirv_FixedSinINTEL<W, rW>(a, S, I, rI);
  // CHECK: %{{.*}} = call spir_func signext i11 @_Z[[#]]__spirv_FixedSinINTEL{{.*}}(i17 signext %[[#]], i1 zeroext true, i32 7, i32 5, i32 0, i32 0)
}

template <int W, int rW, bool S, int I, int rI>
void cos() {
  sycl::detail::ap_int<W> a;
  auto ap_fixed_Cos = __spirv_FixedCosINTEL<W, rW>(a, S, I, rI);
  // CHECK: %{{.*}} = call spir_func signext i28 @_Z[[#]]__spirv_FixedCosINTEL{{.*}}(i35 %[[#]], i1 zeroext false, i32 9, i32 3, i32 0, i32 0)
}

template <int W, int rW, bool S, int I, int rI>
void sin_cos() {
  sycl::detail::ap_int<W> a;
  auto ap_fixed_SinCos = __spirv_FixedSinCosINTEL<W, rW>(a, S, I, rI);
  // CHECK: %{{.*}} = call spir_func i40 @_Z[[#]]__spirv_FixedSinCosINTEL{{.*}}(i31 signext %[[#]], i1 zeroext true, i32 10, i32 12, i32 0, i32 0)
}

template <int W, int rW, bool S, int I, int rI>
void sin_pi() {
  sycl::detail::ap_int<W> a;
  auto ap_fixed_SinPi = __spirv_FixedSinPiINTEL<W, rW>(a, S, I, rI);
  // CHECK: %{{.*}} = call spir_func signext i5 @_Z[[#]]__spirv_FixedSinPiINTEL{{.*}}(i60 %[[#]], i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
}

template <int W, int rW, bool S, int I, int rI>
void cos_pi() {
  sycl::detail::ap_int<W> a;
  auto ap_fixed_CosPi = __spirv_FixedCosPiINTEL<W, rW>(a, S, I, rI);
  // CHECK: %{{.*}} = call spir_func signext i16 @_Z[[#]]__spirv_FixedCosPiINTEL{{.*}}(i28 signext %[[#]], i1 zeroext false, i32 8, i32 5, i32 0, i32 0)
}

template <int W, int rW, bool S, int I, int rI>
void sin_cos_pi() {
  sycl::detail::ap_int<W> a;
  auto ap_fixed_SinCosPi = __spirv_FixedSinCosPiINTEL<W, rW>(a, S, I, rI);
  // CHECK: %{{.*}} = call spir_func signext i10 @_Z[[#]]__spirv_FixedSinCosPiINTEL{{.*}}(i13 signext %[[#]], i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
}

template <int W, int rW, bool S, int I, int rI>
void log() {
  sycl::detail::ap_int<W> a;
  auto ap_fixed_Log = __spirv_FixedLogINTEL<W, rW>(a, S, I, rI);
  // CHECK: %{{.*}} = call spir_func i44 @_Z[[#]]__spirv_FixedLogINTEL{{.*}}(i64 %[[#]], i1 zeroext true, i32 24, i32 22, i32 0, i32 0)
}

template <int W, int rW, bool S, int I, int rI>
void exp() {
  sycl::detail::ap_int<W> a;
  auto ap_fixed_Exp = __spirv_FixedExpINTEL<W, rW>(a, S, I, rI);
  // CHECK: %{{.*}} = call spir_func i34 @_Z[[#]]__spirv_FixedExpINTEL{{.*}}(i44 %[[#]], i1 zeroext false, i32 20, i32 20, i32 0, i32 0)
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    sqrt<13, 5, false, 2, 2>();
    recip<3, 8, true, 4, 4>();
    rsqrt<11, 10, false, 8, 6>();
    sin<17, 11, true, 7, 5>();
    cos<35, 28, false, 9, 3>();
    sin_cos<31, 20, true, 10, 12>();
    sin_pi<60, 5, false, 2, 2>();
    cos_pi<28, 16, false, 8, 5>();
    sin_cos_pi<13, 5, false, 2, 2>();
    log<64, 44, true, 24, 22>();
    exp<44, 34, false, 20, 20>();
  });
  return 0;
}
