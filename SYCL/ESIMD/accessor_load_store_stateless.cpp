//==--- accessor_load_store_stateless.cpp  - DPC++ ESIMD on-device test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The test checks functionality of the scalar load/store accessor-based ESIMD
// intrinsics when stateless memory accesses are enforced, i.e. accessor
// based accesses are automatically converted to stateless accesses.

// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -fsycl-esimd-force-stateless-mem %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "accessor_load_store.hpp"
