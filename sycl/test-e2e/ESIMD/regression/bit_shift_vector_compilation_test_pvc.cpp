//=- bit_shift_vector_compilation_test_pvc.cpp - test vector shifts on PVC -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------===//

// RUN: %{build} -fsycl-device-code-split=per_kernel -std=c++20 -o %t.out
// RUN: %{run} %t.out
// REQUIRES: arch-intel_gpu_pvc

// This is a basic test to validate the vector bit shifting functions on PVC.

#define TEST_PVC

#include "bit_shift_vector_compilation_test.cpp"
