//=dword_local_accessor_atomic_smoke_cmpxchg_scalar_off.cpp-DPC++ ESIMD
// on-device test=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks DWORD local accessor cmpxchg atomic operations with scalar
// offset.
//===----------------------------------------------------------------------===//
// REQUIRES-INTEL-DRIVER: lin: 26690, win: 101.4576
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//

#define USE_DWORD_ATOMICS
#define USE_SCALAR_OFFSET
#define CMPXCHG_TEST

#include "lsc/local_accessor_atomic_smoke.cpp"
