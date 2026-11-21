//===---------------------assert_in_kernels_ndebug.cpp---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests disabling device assertion handling in an application.
/// Verifies that the assertion isn't reported and the application prints
/// "The test ended." before termination.
///
//===----------------------------------------------------------------------===//
// RUN: %{build} -DNDEBUG %S/assert_in_kernels.cpp -o %t.out
// RUN: %{run} %t.out | FileCheck %s
//
// CHECK-NOT: One shouldn't see this message
// CHECK-NOT: from assert statement
// CHECK-NOT: test aborts earlier, one shouldn't see this message
// CHECK: The test ended.
