//==------------ lsc_usm_64.cpp - DPC++ ESIMD on-device test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || esimd_emulator
// UNSUPPORTED: cuda || hip
// TODO : esimd_emulator does not support lsc-atomic yet
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define USE_64_BIT_OFFSET

#include "lsc_usm.cpp"
