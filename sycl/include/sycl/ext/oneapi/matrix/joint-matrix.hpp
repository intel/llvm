//===---- joint-matrix.hpp - SYCL matrix extension joint_matrix ----*- C++
//-*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#ifndef JOINT_MATRIX
#define JOINT_MATRIX

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
  namespace ext {
  namespace oneapi {
  namespace experimental {
  namespace matrix {

  enum class matrix_use { a, b, accumulator };

  enum class layout { row_major, col_major, packed, dynamic };

  namespace precision {
  class tf32 {
    tf32() = delete;
  };
  } // namespace precision

  // TODO: how are the default params for Rows/Cols used in Intel backend?
  // TODO: could we use Cond to distinguish between Intel AMX and tensor cores
  // joint_matrix definitions: e.g. Rows == dynamic_extent?
  template <typename T, matrix_use Use, size_t Rows = sycl::dynamic_extent,
            size_t Cols = sycl::dynamic_extent, layout Layout = layout::dynamic,
            typename Group = sycl::sub_group, typename Cond = void>
  struct joint_matrix;

  } // namespace matrix
  } // namespace experimental
  } // namespace oneapi
  } // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
#endif
