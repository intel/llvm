//==---------- replicate_smoke_pvc.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test checks main functionality of the esimd::replicate_vs_w_hs function.
// PVC variant of the test - adds tfloat32.

#define USE_TF32

#include "replicate_smoke.cpp"
