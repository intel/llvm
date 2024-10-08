//==------- gather_usm_legacy.cpp - DPC++ ESIMD on-device test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Use per-kernel compilation to have more information about failing cases.
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::gather() functions accepting USM pointer
// and optional compile-time esimd::properties.
// The gather() calls in this test do not use cache-hint properties
// or VS > 1 (number of loads per offset) to not impose using PVC features.
//
// TODO: Remove this test when GPU driver issue with llvm.masked.gather is
// resolved and ESIMD starts using llvm.masked.gather by default.
// This "_legacy" test also does not use gather() calls accepting "pass_thru"
// operand to avoid usage of LSC instructions when llvm.masked.gather is
// not available. "-D__ESIMD_GATHER_SCATTER_LLVM_IR" is not used here.

#include "gather_usm.cpp"
