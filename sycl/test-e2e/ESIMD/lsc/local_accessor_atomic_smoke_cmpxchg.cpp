//==local_accessor_atomic_smoke_cmpxchg.cpp - DPC++ ESIMD on-device test=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks local accessor cmpxchg atomic operations.
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// TODO: Enable the test when GPU driver is ready/fixed.
// XFAIL: opencl || windows || gpu-intel-pvc
// UNSUPPORTED: esimd_emulator

// This macro enables only cmpxchg tests. They may require more time to execute,
// and have higher probablity to hit kernel execution time limit, so they are
// separated.

#define CMPXCHG_TEST

#include "local_accessor_atomic_smoke.cpp"
