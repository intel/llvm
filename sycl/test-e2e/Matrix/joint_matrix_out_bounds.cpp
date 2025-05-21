//==-------- joint_matrix_out_bounds.cpp  - DPC++ joint_matrix--------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: target-spir || target-native_cpu

// REQUIRES: aspect-ext_intel_matrix
// UNSUPPORTED: gpu-intel-dg2, cpu
// UNSUPPORTED-INTENDED: Checked load/stores are not supported by DG2 and CPU HW

// Make sure that at least some optimization level is used as we perform
// reference matrix multiplication on host and that is very slow at O0.
// RUN: %{build} -O2 -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"
#include "joint_matrix_out_bounds_impl.hpp"

int main() {
  std::cout << "A row major, B row major:\n";
  test_all<layout::row_major, layout::row_major>();
  std::cout << "A row major, B packed:\n";
  test_all<layout::row_major, layout::ext_intel_packed>();
}
