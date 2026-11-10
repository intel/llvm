//===---col-major-index.cpp - DPC++ joint_matrix---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

// Host check for the col-major accumulator index used by the HIP joint_matrix
// e2e tests. CDNA MFMA accumulators are 16x16 or 32x32, so those tests never
// see M != N. The cases below do.

#include "../../../test-e2e/Matrix/Inputs/joint_matrix_hip_layout.hpp"

int main() {
  check_hip_col_major_rectangles();
  return 0;
}
