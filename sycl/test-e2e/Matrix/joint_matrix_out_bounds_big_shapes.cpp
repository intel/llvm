//==----joint_matrix_out_bounds_big_shapes.cpp - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: target-spir

// REQUIRES: aspect-ext_intel_matrix

// UNSUPPORTED: gpu-intel-dg2, cpu
// UNSUPPORTED-INTENDED: Checked load/stores are not supported by DG2 and CPU HW

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -o %t32.out -DSG_SZ=32
// RUN: %{run} %t32.out

// XFAIL:gpu
// XFAIL-TRACKER: GSD-5768

#include "common.hpp"
#include "joint_matrix_out_bounds_impl.hpp"

int main() {
  std::cout << "A row major, B row major:\n";
  test_all_big_shapes<layout::row_major, layout::row_major>();
  std::cout << "A row major, B packed:\n";
  test_all_big_shapes<layout::row_major, layout::ext_intel_packed>();
  std::cout << "A col major, B col major:\n";
  test_all_big_shapes<layout::col_major, layout::col_major>();
}
