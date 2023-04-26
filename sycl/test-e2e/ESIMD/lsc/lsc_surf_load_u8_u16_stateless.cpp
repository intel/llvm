//==----- lsc_surf_load_u8_u16_stateless.cpp - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || esimd_emulator
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -fsycl-esimd-force-stateless-mem %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "lsc_surf_load_u8_u16.cpp"