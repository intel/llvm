//==-- atomic_update_acc_dg2_pvc_64_2.cpp - DPC++ ESIMD on-device test----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

// REQUIRES: arch-intel_gpu_pvc || gpu-intel-dg2

// RUN: %{build} -fsycl-esimd-force-stateless-mem -o %t.out
// RUN: %{run} %t.out

// 64-bit offset is supported for accessors only in stateless mode
#define USE_64_BIT_OFFSET

#include "atomic_update_acc_dg2_pvc_2.cpp"
