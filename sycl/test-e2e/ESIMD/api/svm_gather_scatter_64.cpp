//==---------- svm_gather_scatter_64.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu && !gpu-intel-pvc
// UNSUPPORTED: gpu-intel-gen9 && windows
// UNSUPPORTED: cuda || hip
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Regression test for gather/scatter API.
// 64 bit offset variant of the test - uses 64 bit offsets.

#define USE_64_BIT_OFFSET

#include "svm_gather_scatter.cpp"
