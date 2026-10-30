//===---joint_matrix_hip_layout.hpp - DPC++ joint_matrix-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_TEST_E2E_MATRIX_JOINT_MATRIX_HIP_LAYOUT_HPP
#define SYCL_TEST_E2E_MATRIX_JOINT_MATRIX_HIP_LAYOUT_HPP

#include <cassert>
#include <vector>

// Map a row-major linear index onto a col-major MxN buffer.
// Element i is (row, col) = (i / cols, i % cols), stored at col * rows + row.
// The row divisor is the column count. Dividing by the row count agrees with
// this only when the tile is square, which is every CDNA MFMA accumulator.
inline int hip_col_major_index(int i, int rows, int cols) {
  return (i % cols) * rows + i / cols;
}

inline void check_hip_col_major_index(int rows, int cols) {
  assert(rows > 0 && cols > 0);
  std::vector<int> hits(rows * cols, 0);
  bool legacy_differs = false;
  for (int i = 0; i < rows * cols; ++i) {
    const int row = i / cols;
    const int col = i % cols;
    const int got = hip_col_major_index(i, rows, cols);
    assert(got == col * rows + row);
    assert(got >= 0 && got < rows * cols);
    hits[got] += 1;
    const int legacy = (i % cols) * rows + i / rows;
    if (got != legacy)
      legacy_differs = true;
  }
  for (int hit : hits)
    assert(hit == 1 && "col-major index is not a permutation");
  if (rows != cols)
    assert(legacy_differs && "M != N must distinguish i/N from i/M");
  else
    assert(!legacy_differs);
}

// Rectangles on both sides of the square, plus the square tiles the e2e
// tests actually launch. The rectangles are what fail if the divisor is M.
inline void check_hip_col_major_rectangles() {
  check_hip_col_major_index(32, 16);
  check_hip_col_major_index(16, 32);
  check_hip_col_major_index(8, 32);
  check_hip_col_major_index(32, 8);
  check_hip_col_major_index(4, 16);
  check_hip_col_major_index(16, 4);
  check_hip_col_major_index(16, 16);
  check_hip_col_major_index(32, 32);
}

#endif
