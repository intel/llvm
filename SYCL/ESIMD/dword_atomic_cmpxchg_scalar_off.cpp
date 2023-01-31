//==-- dword_atomic_cmpxchg_scalar_off.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks LSC atomic operations.
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to random timeouts (_XFAIL_: esimd_emulator)
// UNSUPPORTED: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This macro enables only cmpxch tests. They may require more time to execute,
// and have higher probablity to hit kernel execution time limit, so they are
// separated.
#define CMPXCHG_TEST

// This macro enforces usage of dword atomics in the included test.
#define USE_DWORD_ATOMICS
// This macro enforces scalar offset variant of the test - uses scalar offsets.
#define USE_SCALAR_OFFSET

#include "lsc/atomic_smoke.cpp"
