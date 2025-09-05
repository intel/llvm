//==-------- acc_gather_scatter_rgba.cpp  - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// The test checks functionality of the gather_rgba/scatter_rgba accessor-based
// ESIMD intrinsics.

#include "acc_gather_scatter_rgba.hpp"
