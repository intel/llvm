//==---------------- accessor.cpp  - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -D_CRT_SECURE_NO_WARNINGS=1 %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks that accessor-based memory accesses work correctly in ESIMD.

#include "accessor.hpp"
