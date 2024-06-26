//==------- lsc_usm_store_u8_u16_64.cpp - DPC++ ESIMD on-device test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || gpu-intel-dg2
// TODO: GPU Driver fails with "add3 src operand only supports integer D/W type"
// error. Enable the test when it is fixed.
// UNSUPPORTED: gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// 64 bit offset variant of the test - uses 64 bit offsets.

#define USE_64_BIT_OFFSET

#include "lsc_usm_store_u8_u16.cpp"
