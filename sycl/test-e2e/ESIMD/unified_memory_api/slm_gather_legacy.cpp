//==------- slm_gather_legacy.cpp - DPC++ ESIMD on-device test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// GPU driver had an error in handling of SLM aligned block_loads/stores,
// which has been fixed only in "1.3.26816", and in win/opencl version going
// _after_ 101.4575.
// REQUIRES-INTEL-DRIVER: lin: 26816, win: 101.4576
// Use per-kernel compilation to have more information about failing cases.
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::slm_gather() functions accepting optional
// compile-time esimd::properties. The slm_gather() calls in this test do not
// use  VS > 1 (number of loads per offset) to not impose using PVC features.
//
// TODO: Remove this test when GPU driver issue with llvm.masked.gather is
// resolved and ESIMD starts using llvm.masked.gather by default.
// This "_legacy" test also does not use gather() calls accepting "pass_thru"
// operand to avoid usage of LSC instructions when llvm.masked.gather is
// not available. "-D__ESIMD_GATHER_SCATTER_LLVM_IR" is not used here.

#include "slm_gather.cpp"
