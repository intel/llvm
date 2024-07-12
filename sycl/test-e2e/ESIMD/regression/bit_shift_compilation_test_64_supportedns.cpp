//==- bit_shift_compilation_test_64_supportedns.cpp - Test 64-bit bit shift
// functions -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// TODO: Enable on Gen12 once internal tracker is fixed.
// REQUIRES: arch-intel_gpu_pvc

#define TEST_INT64
#define SUP
#include "bit_shift_compilation_test.cpp"
