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

// RUN: %{build} -fsycl-esimd-force-stateless-mem -o %t.out
// RUN: %{run} %t.out

#include "accessor_load_store.hpp"
