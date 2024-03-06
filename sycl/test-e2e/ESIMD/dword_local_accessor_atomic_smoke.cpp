//==---dword_local_acessor_atomic_smoke.cpp  - DPC++ ESIMD on-device test-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks DWORD local accessor atomic operations.
//===----------------------------------------------------------------------===//
// REQUIRES-INTEL-DRIVER: lin: 26690, win: 101.4576
// TODO: disabled temporarily because of flaky issue.
// UNSUPPORTED: windows
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//

#define USE_DWORD_ATOMICS
#include "lsc/local_accessor_atomic_smoke.cpp"
