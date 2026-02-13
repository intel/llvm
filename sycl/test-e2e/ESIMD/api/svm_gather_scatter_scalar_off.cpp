//==------ svm_gather_scatter_scalar_off.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Regression test for gather/scatter API.
// scalar offset variant of the test - uses scalar offset.

#define USE_SCALAR_OFFSET

#include "svm_gather_scatter.cpp"
