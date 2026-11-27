//===----------------------assert_in_kernels_win.cpp-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests device assertion handling in an application with a multiple kernels,
/// one kernel working at a time. Each work-item triggers an assertion.
/// Verifies that the correct assertion is reported
/// and the application terminates before printing "The test ended.".
/// Linux version of test: assert_in_kernels.cpp
///
//===----------------------------------------------------------------------===//
// REQUIRES: windows
// REQUIRES-INTEL-DRIVER: win: 101.8247

// XFAIL: (opencl && gpu)
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/11364

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out &> %t.txt ; \
// RUN: FileCheck %s --input-file %t.txt --check-prefixes=CHECK,%if native_cpu %{NATIVE%} %else %{OTHER%}

// CHECK-NOT:  One shouldn't see this message
// FIXME Windows version prints '(null)' instead of '<unknown func>' once in a
// while for some insane reason.
// NATIVE:     {{.*}}assert_in_kernels.hpp:39: {{<unknown func>|\(null\)}}:
// OTHER:      {{.*}}assert_in_kernels.hpp:39: {{<unknown func>|\(null\)}}:
// OTHER-SAME: {{.*}} [{{[02]}},0,0]
// CHECK-SAME: Assertion `Buf[wiID] == 0 && "from assert statement"{{[`']}} failed.
// CHECK-NOT:  test aborts earlier, one shouldn't see this message
// CHECK-NOT:  The test ended.

#include "assert_in_kernels.hpp"
