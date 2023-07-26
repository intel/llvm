//==---dword_local_acessor_atomic_smoke.cpp  - DPC++ ESIMD on-device test-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks DWORD local accessor atomic operations.
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// TODO: Enable the test when GPU driver is ready/fixed.
// UNSUPPORTED: gpu
// UNSUPPORTED: esimd_emulator

#define USE_DWORD_ATOMICS
#include "lsc/local_accessor_atomic_smoke.cpp"
