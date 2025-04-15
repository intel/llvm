//==-------- joint_matrix_out_bounds.cpp  - DPC++ joint_matrix--------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: target-nvidia, target-amd
// UNSUPPORTED-INTENDED: aspect-ext_intel_matrix isn't currently supported for
// other triples

// REQUIRES: aspect-ext_intel_matrix
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943

// UNSUPPORTED: gpu-intel-dg2, cpu
// UNSUPPORTED-INTENDED: Checked load/stores are not supported by DG2 and CPU HW

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"
#define SG_SZ 32
#include "joint_matrix_out_bounds_impl.hpp"

int main() {
  std::cout << "A row major, B row major:\n";
  test_all<layout::row_major, layout::row_major>();
  std::cout << "A row major, B packed:\n";
  test_all<layout::row_major, layout::ext_intel_packed>();
}
