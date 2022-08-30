//==---------------- dword_atomic_smoke.cpp  - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks DWORD atomic operations.
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
//
// TODO: esimd_emulator fails due to unsupported __esimd_svm_atomic0/1/2
// TODO: fails on a regular gpu with
// "SYCL exception caught: Native API failed. Native API returns: -1"
// REQUIRES: TEMPORARY_DISABLED
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This macro disables usage of LSC atomics in the included test.
#define UNDEF_USE_LSC_ATOMICS

#include "lsc/atomic_smoke.cpp"
