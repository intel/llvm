// RUN: %clangxx -I %sycl_include -S -emit-llvm -fno-sycl-early-optimizations -fsycl-device-only %s -o - -Xclang -disable-llvm-passes -Xclang -no-enable-noundef-analysis | FileCheck %s
//
//==---- ap_fixed.cpp - SYCL FPGA arbitrary precision fixed point test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

SYCL_EXTERNAL void test_sqrt(sycl::detail::ap_int<13> a) {
  auto ap_fixed_Sqrt = __spirv_FixedSqrtINTEL<13, 5>(a, false, 2, 2);
  // CHECK: %{{.*}} = call spir_func signext i5 @_Z[[#]]__spirv_FixedSqrtINTEL{{.*}}(i13 signext  %[[#]], i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
}

SYCL_EXTERNAL void test_recip(sycl::detail::ap_int<3> a) {
  auto ap_fixed_Recip = __spirv_FixedRecipINTEL<3, 8>(a, true, 4, 4);
  // CHECK: %{{.*}} = call spir_func signext i8 @_Z[[#]]__spirv_FixedRecipINTEL{{.*}}(i3 signext %[[#]], i1 zeroext true, i32 4, i32 4, i32 0, i32 0)
}

SYCL_EXTERNAL void test_rsqrt(sycl::detail::ap_int<11> a) {
  auto ap_fixed_Rsqrt = __spirv_FixedRsqrtINTEL<11, 10>(a, false, 8, 6);
  // CHECK: %{{.*}} = call spir_func signext i10 @_Z[[#]]__spirv_FixedRsqrtINTEL{{.*}}(i11 signext %[[#]], i1 zeroext false, i32 8, i32 6, i32 0, i32 0)
}

SYCL_EXTERNAL void test_sin(sycl::detail::ap_int<17> a) {
  auto ap_fixed_Sin = __spirv_FixedSinINTEL<17, 11>(a, true, 7, 5);
  // CHECK: %{{.*}} = call spir_func signext i11 @_Z[[#]]__spirv_FixedSinINTEL{{.*}}(i17 signext %[[#]], i1 zeroext true, i32 7, i32 5, i32 0, i32 0)
}

SYCL_EXTERNAL void test_cos(sycl::detail::ap_int<35> a) {
  auto ap_fixed_Cos = __spirv_FixedCosINTEL<35, 28>(a, false, 9, 3);
  // CHECK: %{{.*}} = call spir_func signext i28 @_Z[[#]]__spirv_FixedCosINTEL{{.*}}(i35 %[[#]], i1 zeroext false, i32 9, i32 3, i32 0, i32 0)
}

SYCL_EXTERNAL void test_sin_cos(sycl::detail::ap_int<31> a) {
  auto ap_fixed_Sin = __spirv_FixedSinCosINTEL<31, 20>(a, true, 10, 12);
  // CHECK: %{{.*}} = call spir_func i40 @_Z[[#]]__spirv_FixedSinCosINTEL{{.*}}(i31 signext %[[#]], i1 zeroext true, i32 10, i32 12, i32 0, i32 0)
}

SYCL_EXTERNAL void test_sin_pi(sycl::detail::ap_int<60> a) {
  auto ap_fixed_SinPi = __spirv_FixedSinPiINTEL<60, 5>(a, false, 2, 2);
  // CHECK: %{{.*}} = call spir_func signext i5 @_Z[[#]]__spirv_FixedSinPiINTEL{{.*}}(i60 %[[#]], i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
}

SYCL_EXTERNAL void test_cos_pi(sycl::detail::ap_int<28> a) {
  auto ap_fixed_CosPi = __spirv_FixedCosPiINTEL<28, 16>(a, false, 8, 5);
  // CHECK: %{{.*}} = call spir_func signext i16 @_Z[[#]]__spirv_FixedCosPiINTEL{{.*}}(i28 signext %[[#]], i1 zeroext false, i32 8, i32 5, i32 0, i32 0)
}

SYCL_EXTERNAL void test_sin_cos_pi(sycl::detail::ap_int<13> a) {
  auto ap_fixed_SinCosPi = __spirv_FixedSinCosPiINTEL<13, 5>(a, false, 2, 2);
  // CHECK: %{{.*}} = call spir_func signext i10 @_Z[[#]]__spirv_FixedSinCosPiINTEL{{.*}}(i13 signext %[[#]], i1 zeroext false, i32 2, i32 2, i32 0, i32 0)
}

SYCL_EXTERNAL void test_log(sycl::detail::ap_int<64> a) {
  auto ap_fixed_Log = __spirv_FixedLogINTEL<64, 44>(a, true, 24, 22);
  // CHECK: %{{.*}} = call spir_func i44 @_Z[[#]]__spirv_FixedLogINTEL{{.*}}(i64 %[[#]], i1 zeroext true, i32 24, i32 22, i32 0, i32 0)
}

SYCL_EXTERNAL void test_exp(sycl::detail::ap_int<44> a) {
  auto ap_fixed_Exp = __spirv_FixedExpINTEL<44, 34>(a, false, 20, 20);
  // CHECK: %{{.*}} = call spir_func i34 @_Z[[#]]__spirv_FixedExpINTEL{{.*}}(i44 %[[#]], i1 zeroext false, i32 20, i32 20, i32 0, i32 0)
}