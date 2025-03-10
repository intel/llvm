//==----- lsc_surf_load_u8_u16_stateless.cpp - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -fsycl-esimd-force-stateless-mem -o %t.out
// RUN: %{run} %t.out

#include "lsc_surf_load_u8_u16.cpp"
