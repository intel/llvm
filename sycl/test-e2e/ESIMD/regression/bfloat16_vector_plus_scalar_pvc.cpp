//==- bfloat16_vector_plus_scalar_pvc.cpp - Test for bfloat16 operators -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test validates operations between simd vector and scalars for
// bfloat16 and tfloat32 types that are available only on PVC.
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define USE_BF16
#define USE_TF32
#include "bfloat16_vector_plus_scalar.cpp"