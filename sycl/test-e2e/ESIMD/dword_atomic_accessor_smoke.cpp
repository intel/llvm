//==------ dword_atomic_accessor__smoke.cpp  - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks DWORD atomic operations.
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: gpu-intel-gen9 && windows
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This macro enforces usage of dword atomics in the included test.
#define USE_DWORD_ATOMICS
// This macro enforces usage of accessor based API in the included test
#define USE_ACCESSORS
#include "lsc/atomic_smoke.cpp"
