//==------- lsc_usm_store_u32_64.cpp - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc || gpu-intel-dg2

// Windows compiler causes this test to fail due to handling of floating
// point arithmetics. Fixing requires using
// -ffp-exception-behavior=maytrap option to disable some floating point
// optimizations to produce correct result.
// DEFINE: %{fpflags} = %if cl_options %{/clang:-ffp-exception-behavior=maytrap%} %else %{-ffp-exception-behavior=maytrap%}

// RUN: %{build} %{fpflags} -o %t.out
// RUN: %{run} %t.out

// 64 bit offset variant of the test - uses 64 bit offsets.

#define USE_64_BIT_OFFSET

#include "lsc_usm_store_u32.cpp"
