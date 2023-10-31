//==------- windows_msvc_math.cpp - DPC++ ESIMD build test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-windows -aux-triple x86_64-pc-windows-msvc -fsyntax-only -verify %s
// expected-no-diagnostics

// The tests validates an ability to build ESIMD code on windows platform.

extern __attribute__((sycl_device)) short _FDtest(float *px);

__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void kern() {
  float a;
  _FDtest(&a);
}
