//==----joint_matrix_out_bounds_colmajor.cpp  - DPC++ joint_matrix---------==//
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

// UNSUPPORTED: gpu-intel-dg2, cpu
// UNSUPPORTED-INTENDED: Checked load/stores are not supported by DG2 and CPU HW

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -o %t32.out -DSG_SZ=32
// RUN: %{run} %t32.out

#include "common.hpp"
#include "joint_matrix_out_bounds_impl.hpp"

int main() {
  std::cout << "A col major, B col major:\n";
  test_all<layout::col_major, layout::col_major>();
}
