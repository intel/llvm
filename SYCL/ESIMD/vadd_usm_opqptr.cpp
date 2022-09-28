//==---------------- vadd_usm_opqptr.cpp  - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -Xclang -opaque-pointers %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// TODO Running existing tests in opaque pointer mode should be supported by the
// CI.

// The test checks if vadd_usm.cpp works in opaque pointer mode.

#include "vadd_usm.cpp"
