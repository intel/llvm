//==------------ lsc_surf.cpp - DPC++ ESIMD on-device test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || gpu-intel-dg2
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test checks functionality of the lsc_block_load, lsc_prefetch,
// lsc_gather, lsc_scatter, lsc_atomic_update accessor-based ESIMD intrinsics.

#include "Inputs/lsc_surf.hpp"
