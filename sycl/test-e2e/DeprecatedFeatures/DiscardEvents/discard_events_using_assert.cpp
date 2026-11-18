//===-------------------discard_events_using_assert.cpp--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Verifies that the last parameter is not `nullptr` for
/// urEnqueueKernelLaunchWithArgsExp when the queue has the discard_events
/// property.
///
//===----------------------------------------------------------------------===//
// FIXME unsupported on HIP until fallback libdevice becomes available
// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/22300

// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out &> %t.txt
// RUN: FileCheck %s --input-file %t.txt

// CHECK-NOT: <--- urEnqueueKernelLaunchWithArgsExp({{.*}}.phEvent = nullptr
// CHECK: <--- urEnqueueKernelLaunchWithArgsExp
// CHECK: -> UR_RESULT_SUCCESS
// CHECK: The test passed.

#include "discard_events_kernel_using_assert.hpp"
