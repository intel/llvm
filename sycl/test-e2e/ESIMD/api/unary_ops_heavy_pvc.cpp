//==--------------- unary_ops_heavy_pvc.cpp  - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests various unary operations applied to simd objects.
// PVC variant of the test - adds bfloat16.

#define USE_BF16

#include "unary_ops_heavy.cpp"
