//==------- atomic_update_slm_cmpxchg.cpp - DPC++ ESIMD on-device test  ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES-L0-DRIVER: 26918
// REQUIRES-INTEL-WINDOWS-DRIVER: 101.4953
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define CMPXCHG_TEST

#include "atomic_update_slm.cpp"
