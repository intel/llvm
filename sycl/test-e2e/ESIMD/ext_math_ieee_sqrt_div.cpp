//==-- ext_math_ieee_qsrt_div.cpp  - DPC++ ESIMD ieee sqrt/div  test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-gen9 || gpu-intel-dg2 || gpu-intel-pvc || esimd_emulator

// RUN: %clangxx -fsycl-device-code-split=per_kernel -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks ieee_sqrt() and ieee_sqrt() with float and double types.

#define TEST_IEEE_DIV_REM 1
#include "ext_math.cpp"
