
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Test hangs on AMD
// UNSUPPORTED: hip

//==- hier_par_wgscope.cpp --- hierarchical parallelism test for WG scope---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test checks correctness of hierarchical kernel execution when there is
// code and data in the work group scope.

#include "Inputs/hier_par_wgscope_impl.hpp"

int main() { return run(); }
