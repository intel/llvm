//==-------- fma_pvc.cpp - DPC++ ESIMD on-device test--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#define TEST_PVC

#include "fma.cpp"
