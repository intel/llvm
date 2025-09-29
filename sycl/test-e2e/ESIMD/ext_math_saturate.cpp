//==----------- ext_math_saturate.cpp  - DPC++ ESIMD extended math test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES-INTEL-DRIVER: lin: 27012, win: 101.4576
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: arch-intel_gpu_pvc

// This test checks extended math operations called with saturation.
// Combinations of
// - argument type - half, float
// - math function - sin, cos, ..., div_ieee, pow
// - SYCL vs ESIMD APIs

#define SATURATION_ON

#include "ext_math.cpp"
