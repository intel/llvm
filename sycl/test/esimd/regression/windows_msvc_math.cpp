//==------- windows_msvc_math.cpp - DPC++ ESIMD build test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: windows
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

// The tests validates an ability to build ESIMD code on windows platform.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

extern __DPCPP_SYCL_EXTERNAL short _FDtest(float *px);

int main() {
  queue q;

  q.single_task([=]() SYCL_ESIMD_KERNEL {
     float a;
     _FDtest(&a);
   }).wait();

  return 0;
}
