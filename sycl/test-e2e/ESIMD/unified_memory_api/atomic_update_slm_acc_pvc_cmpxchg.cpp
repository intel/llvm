//==- atomic_update_slm_acc_pvc_cmpxchg.cpp -- DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: arch-intel_gpu_pvc

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define CMPXCHG_TEST

#include "atomic_update_slm_acc_pvc.cpp"
