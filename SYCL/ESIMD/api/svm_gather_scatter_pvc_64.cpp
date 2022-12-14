//==---------- svm_gather_scatter_pvc_64.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Regression test for SVM gather/scatter API.
// PVC variant of the test - adds tfloat32 and uses 64 bit offsets.

#define USE_TF32
#define USE_64_BIT_OFFSET

#include "svm_gather_scatter.cpp"
