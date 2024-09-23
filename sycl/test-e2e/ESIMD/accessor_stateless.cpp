//==---------------- accessor_stateless.cpp  - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -fsycl-esimd-force-stateless-mem -D_CRT_SECURE_NO_WARNINGS=1 -o %t.out
// RUN: %{run} %t.out

// This test checks that accessor-based memory accesses work correctly in ESIMD
// when stateless memory accesses are enforced, i.e. accessor based accesses
// are automatically converted to stateless accesses.

#include "accessor.hpp"
