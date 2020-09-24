//==----------- spec_const_hw_accelerator.cpp ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The test checks that the specialization constant feature works correctly -
// tool chain processes them correctly and runtime can correctly execute the
// program.

// TODO: re-enable after CI drivers are updated to newer which support spec
// constants:
// XFAIL: opencl && accelerator
// UNSUPPORTED: cuda || level_zero
#include "spec_const_hw.cpp"
// RUN: %ACC_RUN_PLACEHOLDER %t.out
