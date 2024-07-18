//==---------- simd_copy_to_from_pvc.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks simd::copy_from/to methods with alignment flags.
// PVC variant of the test - adds tfloat32.

#define USE_TF32

#include "simd_copy_to_from.cpp"
