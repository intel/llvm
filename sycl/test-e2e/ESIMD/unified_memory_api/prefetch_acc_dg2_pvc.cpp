//==------- prefetch_acc_dg2_pvc.cpp - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-dg2 || gpu-intel-pvc
// RUN: %{build} -fsycl-device-code-split=per_kernel -D__ESIMD_GATHER_SCATTER_LLVM_IR  -o %t.out
// RUN: %{run} %t.out
// The test verifies esimd::prefetch() functions accepting accessor
// and optional compile-time esimd::properties.
// The prefetch() calls in this test require DG2 or PVC to run.
#include "prefetch_acc_stateful_dg2_pvc.cpp"