//==- rotate_pvc.cpp - Test to verify ror/rol functionality ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -fsycl-device-code-split=per_kernel -std=c++20 -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -fsycl-device-code-split=per_kernel -std=c++20 -o %t1.out -DEXP
// RUN: %{run} %t1.out
// REQUIRES: arch-intel_gpu_pvc

// This is a basic test to validate the ror/rol functions on PVC.

#define TEST_PVC
#include "rotate.cpp"
