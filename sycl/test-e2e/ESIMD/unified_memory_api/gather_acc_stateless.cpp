//==------- gather_acc_stateless.cpp - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Use per-kernel compilation to have more information about failing cases.
// RUN: %{build} -fsycl-device-code-split=per_kernel -fsycl-esimd-force-stateless-mem -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::gather() functions accepting ACCESSOR
// and optional compile-time esimd::properties in `stateless` memory mode.
// The gather() calls in this test do not use cache-hint properties
// or VS > 1 (number of loads per offset) to not impose using DG2/PVC features.

#include "gather_acc.cpp"
