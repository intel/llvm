//==------ dword_atomic_accessor_smoke.cpp  - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks DWORD atomic operations.
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This macro enforces usage of dword atomics in the included test.
#define USE_DWORD_ATOMICS
// This macro enforces usage of accessor based API in the included test
#define USE_ACCESSORS
#include "lsc/atomic_smoke.cpp"
