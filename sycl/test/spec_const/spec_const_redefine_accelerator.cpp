//==----------- spec_const_redefine.cpp ------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The test checks that:
// - a specialization constant can be redifined and correct new value is used
//   after redefinition.
// - the program is JITted only once per a unique set of specialization
//   constants values.

// TODO: re-enable after CI drivers are updated to newer which support spec
// constants:
// XFAIL: opencl && accelerator
// UNSUPPORTED: cuda || level_zero
#include "spec_const_redefine_accelerator.cpp"
// RUN: env SYCL_PI_TRACE=2 %ACC_RUN_PLACEHOLDER %t.out 2>&1 %ACC_CHECK_PLACEHOLDER
