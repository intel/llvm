//==- acc_gather_scatter_rgba_stateless.cpp  - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: gpu-intel-gen9 && windows
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -fsycl-esimd-force-stateless-mem %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks functionality of the gather_rgba/scatter_rgba accessor-based
// ESIMD intrinsics when stateless memory accesses are enforced, i.e. accessor
// based accesses are automatically converted to stateless accesses.

#include "acc_gather_scatter_rgba.hpp"
