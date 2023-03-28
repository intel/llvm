//==--- lsc_usm_prefetch_u64_scalar_off.cpp - DPC++ ESIMD on-device test---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// scalar offset variant of the test - uses scalar offsets.

#define USE_SCALAR_OFFSET

#include "lsc_usm_prefetch_u64.cpp"
