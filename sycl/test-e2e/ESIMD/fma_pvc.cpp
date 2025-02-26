//==-------- fma_pvc.cpp - DPC++ ESIMD on-device test--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// REQUIRES-INTEL-DRIVER: lin: 29138, win: 101.5499
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#define TEST_PVC

#include "fma.cpp"
