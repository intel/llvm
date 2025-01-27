//==--- simd_view_subscript_operator_pvc.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test checks that it's possible to write through the simd_view subscript
// operator. E.g.:
//   simd<int, 4> v = 1;
//   auto v1 = v.select<2, 1>(0);
//   v1[0] = 0; // v1[0] returns writable simd_view
// PVC variant of the test - adds tfloat32.

#define USE_TF32

#include "simd_view_subscript_operator.cpp"
