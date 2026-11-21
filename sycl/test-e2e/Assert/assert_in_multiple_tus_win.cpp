//===--------------------assert_in_multiple_tus_win.cpp--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests device-side assertion failures in kernels defined in two translation
/// units. Defines NDEBUG in the first translation unit only. Verifies that an
/// assertion diagnostic from the first translation unit is printed, none from
/// the second is printed, and the program does not reach normal completion.
/// Linux version of test: assert_in_multiple_tus.cpp
///
//===----------------------------------------------------------------------===//
// REQUIRES: windows
// REQUIRES-INTEL-DRIVER: win: 101.8247

// XFAIL: (opencl && gpu)
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/11364

// RUN: %{build} %cxx_std_optionc++20 %S/Inputs/kernels_in_file2.cpp -o %t.out
// RUN: %{run} %t.out &> %t.txt ; \
// RUN: FileCheck %s --input-file %t.txt --check-prefixes=CHECK,%if native_cpu %{NATIVE%} %else %{OTHER%}

// FIXME Windows version prints '(null)' instead of '<unknown func>' once in a
// while for some insane reason.
// NATIVE:     {{.*}}kernels_in_file2.cpp:28: {{<unknown func>|\(null\)}}:
// OTHER:      {{.*}}kernels_in_file2.cpp:28: {{<unknown func>|\(null\)}}:
// OTHER-SAME: {{.*}} [15,0,0]
// CHECK-SAME: Assertion `X && "this message from calculus"{{[`']}} failed.
// CHECK-NOT:  this message from file2
// CHECK-NOT:  The test ended.

#include "assert_in_multiple_tus.hpp"
