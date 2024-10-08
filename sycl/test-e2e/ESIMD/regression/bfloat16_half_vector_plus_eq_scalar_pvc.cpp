//= bfloat16_half_vector_plus_eq_scalar_pvc.cpp - Test for bfloat16 operators=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define USE_BF16
#define USE_TF32
#include "bfloat16_half_vector_plus_eq_scalar.cpp"