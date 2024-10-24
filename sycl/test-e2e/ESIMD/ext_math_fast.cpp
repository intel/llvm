//==----------- ext_math_fast.cpp  - DPC++ ESIMD extended math test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// DEFINE: %{slpflags} = %if cl_options %{/clang:-fno-slp-vectorize%} %else %{-fno-slp-vectorize%}
// RUN: %{build} -fsycl-device-code-split=per_kernel -ffast-math %{slpflags} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: arch-intel_gpu_pvc

// This test checks extended math operations. Combinations of
// - argument type - half, float
// - math function - sin, cos, ..., div_ieee, pow
// - SYCL vs ESIMD APIs

// This version of the test checks -ffast-math option which may cause code-gen
// of different-precision variants of math functions.
// The option -fno-slp-vectorize prevents vectorization of code in kernel
// operator() to avoid the extra difficulties in results verification.

#define SKIP_NEW_GPU_DRIVER_VERSION_CHECK 1
#define TEST_FAST_MATH 1

#include "ext_math.cpp"
