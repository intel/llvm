//===-----------------------assert_in_one_kernel.cpp-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests device assertion handling in an application with a single kernel.
/// Each work-item triggers an assertion. Verifies that the assertion is
/// reported and the application terminates before printing "The test ended.".
/// Windows version of test: assert_in_one_kernel_win.cpp
///
//===----------------------------------------------------------------------===//
// REQUIRES: linux

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/7634

// XFAIL: (opencl && gpu)
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/11364

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out &> %t.txt ; \
// RUN: FileCheck %s --input-file %t.txt --check-prefixes=CHECK,%if native_cpu %{NATIVE%} %else %{OTHER%}

// NATIVE:     {{.*}}assert_in_one_kernel.hpp:24: void kernelFunc(int *, int):
// OTHER:      {{.*}}assert_in_one_kernel.hpp:24: void kernelFunc(int *, int):
// OTHER-SAME: {{.*}} [{{[0-3]}},0,0]
// CHECK-SAME: Assertion `Buf[wiID] != 0 && "from assert statement"{{[`']}} failed
// CHECK-NOT:  The test ended.

#include "assert_in_one_kernel.hpp"
