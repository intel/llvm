//==- acc_gather_scatter_rgba_stateless.cpp  - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -fsycl-esimd-force-stateless-mem -o %t.out
// RUN: %{run} %t.out
//
// The test checks functionality of the gather_rgba/scatter_rgba accessor-based
// ESIMD intrinsics when stateless memory accesses are enforced, i.e. accessor
// based accesses are automatically converted to stateless accesses.

#include "acc_gather_scatter_rgba.hpp"
