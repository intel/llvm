//==-- ext_math_ieee_qsrt_div.cpp  - DPC++ ESIMD ieee sqrt/div  test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-gen9 || arch-intel_gpu_pvc

// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
// RUN: %{build} -fsycl-device-code-split=per_kernel %{mathflags} -o %t.out
// RUN: %{run} %t.out

// This test checks ieee_sqrt() and ieee_sqrt() with float and double types.

#define SKIP_NEW_GPU_DRIVER_VERSION_CHECK 1
#define TEST_IEEE_DIV_REM 1
#include "ext_math.cpp"
