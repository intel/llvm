//===------------ matrix-aot-amx.hpp - SYCL matrix ------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
///
/// We provide new interfaces for matrix muliply in this patch:
/// 1. A new class called joint_matrix is introduced, and the user needs to
/// specify the type of the elements, sizes, and the memory layout.
///
/// 2. joint_matrix_load is used for loading data from main memory to tiles of
/// AMX or kernel's local memory.
///
/// 3. joint_matrix_store is used for storing data tiles of AMX or kernel's
/// local memory to main memory.
///
/// 4. joint_matrix_mad is used for the matrix multiply and add function.
/// It performs the multiply operation on the matrices A and B, accumulates the
/// result with C and returns the result.
///
/// The following operation can be realized with the interfaces:
///  C = A*B+C
/// 1. All cases where A(int8, any-size, row_major), B(int8, any-size,
/// packed_b), C(int32, any-size, row_major)
/// 2. All cases where A(bf16, any-size, row_major), B(bf16, any-size,
/// packed_b), C(float, any-size, row_major)
///
///
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>
#include <immintrin.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace detail {
template <typename T> class submatrix {
public:
  _tile1024i tile;
  short rows, cols;
};

// TODO: we are adding it this way until sycl::dynamic_extent gets implemented.
constexpr size_t dynamic_extent = std::numeric_limits<size_t>::max();

template <typename T> struct elems_per_dword {
  static constexpr size_t value = 1;
};

#define ELEMS_PER_DWORD(TYPE, NUM)                                             \
  template <> struct elems_per_dword<TYPE> {                                   \
    static constexpr size_t value = NUM;                                       \
  };

ELEMS_PER_DWORD(int8_t, 4)
ELEMS_PER_DWORD(unsigned short, 2)

} // namespace detail

namespace experimental::matrix {
#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL extern "C" _tile1024i
_tileloadd64_internal(short row, short col, char *buf, size_t stride);
SYCL_EXTERNAL extern "C" _tile1024i
_tdpbssd_internal(unsigned short m, unsigned short n, unsigned short k,
                  _tile1024i dst, _tile1024i src1, _tile1024i src2);
SYCL_EXTERNAL extern "C" _tile1024i
_tdpbf16ps_internal(unsigned short m, unsigned short n, unsigned short k,
                    _tile1024i dst, _tile1024i src1, _tile1024i src2);
SYCL_EXTERNAL extern "C" void _tilestored64_internal(short row, short col,
                                                     char *buf, size_t stride,
                                                     _tile1024i tile);
static _tile1024i tileloadd64_internal(short row, short col, char *buf,
                                       size_t stride) {
  return _tileloadd64_internal(row, col, buf, stride);
}
static _tile1024i tdpbssd_internal(unsigned short m, unsigned short n,
                                   unsigned short k, _tile1024i dst,
                                   _tile1024i src1, _tile1024i src2) {
  return _tdpbssd_internal(m, n, k, dst, src1, src2);
}
static _tile1024i tdpbf16ps_internal(unsigned short m, unsigned short n,
                                     unsigned short k, _tile1024i dst,
                                     _tile1024i src1, _tile1024i src2) {
  return _tdpbf16ps_internal(m, n, k, dst, src1, src2);
}
static void tilestored64_internal(short row, short col, char *buf,
                                  size_t stride, _tile1024i tile) {
  return _tilestored64_internal(row, col, buf, stride, tile);
}
#else
static _tile1024i tileloadd64_internal(short row, short col, char *buf,
                                       size_t stride) {
  return __builtin_ia32_tileloadd64_internal(row, col, buf, stride);
}
static _tile1024i tdpbssd_internal(unsigned short m, unsigned short n,
                                   unsigned short k, _tile1024i dst,
                                   _tile1024i src1, _tile1024i src2) {
  return __builtin_ia32_tdpbssd_internal(m, n, k, dst, src1, src2);
}
static _tile1024i tdpbf16ps_internal(unsigned short m, unsigned short n,
                                     unsigned short k, _tile1024i dst,
                                     _tile1024i src1, _tile1024i src2) {
  return __builtin_ia32_tdpbf16ps_internal(m, n, k, dst, src1, src2);
}
static void tilestored64_internal(short row, short col, char *buf,
                                  size_t stride, _tile1024i tile) {
  __builtin_ia32_tilestored64_internal(row, col, buf, stride, tile);
}
#endif

enum class matrix_layout { row_major, col_major, packed_a, packed_b };

inline constexpr size_t tile_size = 16;

template <typename Group, typename T, size_t NumRows = detail::dynamic_extent,
          size_t NumCols = detail::dynamic_extent,
          matrix_layout Layout = matrix_layout::row_major,
          typename Enabled = void>
struct joint_matrix {
  joint_matrix(Group sg) {}
  joint_matrix(Group sg, size_t Size) {
    static_assert((NumRows != detail::dynamic_extent &&
                   NumCols != detail::dynamic_extent),
                  "AMX implementation does not support dynamic allocation");
  }
  joint_matrix(Group sg, size_t Rows, size_t Cols) {
    static_assert((NumRows != detail::dynamic_extent &&
                   NumCols != detail::dynamic_extent),
                  "AMX implementation does not support dynamic allocation");
  }
};

// This template specialization handles cases where matrix can't be accommodated
// by a tile. In this case, we create raw_storage for the matrix and the size
// is the multiply of (TILE*TILE*4).
template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout>
struct joint_matrix<
    Group, T, NumRows, NumCols, Layout,
    typename std::enable_if<!((NumRows <= tile_size) &&
                              (NumCols * sizeof(T) / 4 <= tile_size) &&
                              (Layout != matrix_layout::col_major))>::type> {
public:
  // trows: Num of tiles in row.
  // If T=int8, NumRows==33, trows should be 3=(33+15)/16
  static constexpr size_t trows = (NumRows + tile_size - 1) / tile_size;
  // tcols: Num of tiles in column.
  static constexpr size_t tcols =
      (NumCols * sizeof(T) / 4 + tile_size - 1) / tile_size;
  // if T=int8, NumRows==33, NumCols==33*4, tile_size==16, then size of
  // raw_storage should be 48*48*4.
  // FIXME: Greedy Regalloc for tile seems has some limitation and currently we
  // do tileload for (16,16*4) instead of varying shapes, so raw_storage's size
  // is multiple of (16*16*4)
  static constexpr size_t size = trows * tcols * tile_size * tile_size * 4;
  // stride is aligned to T instead of int8
  static constexpr size_t stride = tcols * tile_size * 4 / sizeof(T);
  int8_t raw_storage[size];
  static constexpr bool isSmall = false;

public:
  matrix_layout layout;
  // We do zero-padding for matrix whose size is not fitted into tiles in ctor.
  joint_matrix(Group sg) { memset(raw_storage, 0x00, size); }
};

// This template specialization handles cases where matrix can be put into a
// tile and users specify layout is packed_a or packed_b
template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout>
struct joint_matrix<
    Group, T, NumRows, NumCols, Layout,
    typename std::enable_if<(NumRows <= tile_size) &&
                            (NumCols * sizeof(T) / 4 <= tile_size)>::type> {
public:
  static constexpr size_t trows = (NumRows + tile_size - 1) / tile_size;
  // tcols: Num of tiles in column.
  static constexpr size_t tcols =
      (NumCols * sizeof(T) / 4 + tile_size - 1) / tile_size;
  static constexpr size_t size = trows * tcols * tile_size * tile_size * 4;
  // stride is aligned to T instead of int8
  static constexpr size_t stride = tcols * tile_size * 4 / sizeof(T);
  _tile1024i tile;
  static constexpr bool isSmall = true;
  matrix_layout layout;
  // We do zero-padding for matrix whose size is not fitted into tiles in ctor.
  joint_matrix(Group sg) {}
};

} // namespace experimental::matrix

namespace detail {

using namespace experimental;

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix::matrix_layout Layout>
inline __SYCL_ALWAYS_INLINE static
    typename std::enable_if<(NumRows > matrix::tile_size) ||
                                (NumCols * sizeof(T) / 4 > matrix::tile_size),
                            void>::type
    submatrix_load(detail::submatrix<T> &sub_m,
                   matrix::joint_matrix<Group, T, NumRows, NumCols, Layout> jm,
                   uint32_t row, uint32_t col, size_t stride,
                   matrix::matrix_layout layout, bool shouldreload) {
  uint32_t offset = (row * stride + col);
  T *ptr = reinterpret_cast<T *>(jm.raw_storage);
  ptr += offset;
  stride *= sizeof(T);
  sub_m.rows = matrix::tile_size;
  sub_m.cols = matrix::tile_size * 4;
  sub_m.tile = matrix::tileloadd64_internal(
      sub_m.rows, sub_m.cols, reinterpret_cast<char *>(ptr), stride);
}

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix::matrix_layout Layout>
inline __SYCL_ALWAYS_INLINE static
    typename std::enable_if<(NumRows <= matrix::tile_size) &&
                                (NumCols * sizeof(T) / 4 <= matrix::tile_size),
                            void>::type
    submatrix_load(detail::submatrix<T> &sub_m,
                   matrix::joint_matrix<Group, T, NumRows, NumCols, Layout> &jm,
                   uint32_t row, uint32_t col, size_t stride,
                   matrix::matrix_layout layout, bool shouldreload) {
  if (shouldreload) {
    // Force sub_m.tile's shape to be matrix::tile_size *
    // matrix::tile_size * 4
    int8_t NewjmC[matrix::tile_size * matrix::tile_size * 4];
    matrix::tilestored64_internal(NumRows, NumCols * sizeof(T),
                                  reinterpret_cast<char *>(NewjmC),
                                  matrix::tile_size * 4, jm.tile);
    sub_m.rows = matrix::tile_size;
    sub_m.cols = matrix::tile_size * 4;
    sub_m.tile = matrix::tileloadd64_internal(sub_m.rows, sub_m.cols,
                                              reinterpret_cast<char *>(NewjmC),
                                              matrix::tile_size * 4);
    return;
  }
  sub_m.rows = NumRows;
  sub_m.cols = NumCols * sizeof(T);
  sub_m.tile = jm.tile;
}

// This handles cases where T1 is int8, T2 is int32.
inline __SYCL_ALWAYS_INLINE static void
submatrix_mad(detail::submatrix<int8_t> &sub_ma,
              detail::submatrix<int8_t> &sub_mb,
              detail::submatrix<int32_t> &sub_mc) {
  sub_mc.tile = matrix::tdpbssd_internal(sub_mc.rows, sub_mc.cols, sub_ma.cols,
                                         sub_mc.tile, sub_ma.tile, sub_mb.tile);
}

// This handles cases where T1 is int16(bfloat16), T2 is float.
inline __SYCL_ALWAYS_INLINE static void
submatrix_mad(detail::submatrix<unsigned short> &sub_ma,
              detail::submatrix<unsigned short> &sub_mb,
              detail::submatrix<float> &sub_mc) {
  sub_mc.tile =
      matrix::tdpbf16ps_internal(sub_mc.rows, sub_mc.cols, sub_ma.cols,
                                 sub_mc.tile, sub_ma.tile, sub_mb.tile);
}

template <typename Group, typename T, size_t NumRows, size_t NumCols>
inline __SYCL_ALWAYS_INLINE static
    typename std::enable_if<(NumRows > matrix::tile_size) ||
                                (NumCols * sizeof(T) / 4 > matrix::tile_size),
                            void>::type
    submatrix_store(detail::submatrix<T> &sub_m,
                    matrix::joint_matrix<Group, T, NumRows, NumCols> &jm,
                    uint32_t row, uint32_t col, size_t stride,
                    matrix::matrix_layout layout, bool shouldreload) {
  uint32_t offset = (row * stride + col);
  T *ptr = reinterpret_cast<T *>(jm.raw_storage);
  ptr += offset;
  stride *= sizeof(T);
  matrix::tilestored64_internal(sub_m.rows, sub_m.cols,
                                reinterpret_cast<char *>(ptr), stride,
                                sub_m.tile);
}

template <typename Group, typename T, size_t NumRows, size_t NumCols>
inline __SYCL_ALWAYS_INLINE static
    typename std::enable_if<(NumRows <= matrix::tile_size) &&
                                (NumCols * sizeof(T) / 4 <= matrix::tile_size),
                            void>::type
    submatrix_store(detail::submatrix<T> &sub_m,
                    matrix::joint_matrix<Group, T, NumRows, NumCols> &jm,
                    uint32_t row, uint32_t col, size_t stride,
                    matrix::matrix_layout layout, bool shouldreload) {
  if (shouldreload) {
    int8_t NewjmC[matrix::tile_size * matrix::tile_size * 4];
    matrix::tilestored64_internal(matrix::tile_size, matrix::tile_size * 4,
                                  reinterpret_cast<char *>(NewjmC),
                                  matrix::tile_size * 4, sub_m.tile);
    jm.tile = matrix::tileloadd64_internal(NumRows, NumCols * sizeof(T),
                                           reinterpret_cast<char *>(NewjmC),
                                           matrix::tile_size * 4);
    return;
  }
  jm.tile = sub_m.tile;
}

} // namespace detail

namespace experimental::matrix {

// This handles cases where matrix can't be accommodated by a tile
template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout, access::address_space Space>
inline __SYCL_ALWAYS_INLINE typename std::enable_if<
    (NumRows > tile_size) || (NumCols * sizeof(T) / 4 > tile_size), void>::type
joint_matrix_load(Group sg,
                  joint_matrix<Group, T, NumRows, NumCols, Layout> &jm,
                  multi_ptr<T, Space> src, size_t stride,
                  matrix_layout layout) {
  T *mem = src.get();
  // memcpy from mem to jm.raw_storage
  for (int i = 0; i < NumRows; ++i) {
    char *srcptr = reinterpret_cast<char *>(mem) + i * stride * sizeof(T);
    char *dstptr =
        reinterpret_cast<char *>(jm.raw_storage) + i * jm.stride * sizeof(T);
    // TODO: we may reformat layout.
    memcpy(dstptr, srcptr, NumCols * sizeof(T));
  }
  jm.layout = layout;
}

// This handles cases where matrix can be put into a tile
template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout, access::address_space Space>
inline __SYCL_ALWAYS_INLINE
    typename std::enable_if<(NumRows <= tile_size) &&
                                (NumCols * sizeof(T) / 4 <= tile_size),
                            void>::type
    joint_matrix_load(Group sg,
                      joint_matrix<Group, T, NumRows, NumCols, Layout> &jm,
                      multi_ptr<T, Space> src, size_t stride,
                      matrix_layout layout) {
  T *mem = src.get();
  // tileload happens!
  jm.tile =
      tileloadd64_internal(NumRows, NumCols * sizeof(T),
                           reinterpret_cast<char *>(mem), stride * sizeof(T));
  jm.layout = layout;
}

// This handles cases where matrix can't be accommodated by a tile
template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout, access::address_space Space>
inline __SYCL_ALWAYS_INLINE typename std::enable_if<
    (NumRows > tile_size) || (NumCols * sizeof(T) / 4 > tile_size), void>::type
joint_matrix_store(Group sg,
                   joint_matrix<Group, T, NumRows, NumCols, Layout> &jm,
                   multi_ptr<T, Space> dst, size_t stride,
                   matrix_layout layout) {
  T *mem = dst.get();
  for (int i = 0; i < NumRows; ++i) {
    char *dstptr = reinterpret_cast<char *>(mem) + i * stride * sizeof(T);
    char *srcptr =
        reinterpret_cast<char *>(jm.raw_storage) + i * jm.stride * sizeof(T);
    // TODO: we may reformat layout.
    memcpy(dstptr, srcptr, NumCols * sizeof(T));
  }
  return;
}

// This handles cases where matrix can be put into a tile
template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout, access::address_space Space>
inline __SYCL_ALWAYS_INLINE
    typename std::enable_if<(NumRows <= tile_size) &&
                                (NumCols * sizeof(T) / 4 <= tile_size),
                            void>::type
    joint_matrix_store(Group sg,
                       joint_matrix<Group, T, NumRows, NumCols, Layout> &jm,
                       multi_ptr<T, Space> dst, size_t stride,
                       matrix_layout layout) {
  T *mem = dst.get();
  // tilestore happens!
  tilestored64_internal(NumRows, NumCols * sizeof(T),
                        reinterpret_cast<char *>(mem), stride * sizeof(T),
                        jm.tile);
  return;
}

template <typename Group, typename T1, typename T2, size_t NumRowsA,
          size_t NumColsA, size_t NumRowsB, size_t NumColsB, size_t NumRowsC,
          size_t NumColsC, matrix_layout LayoutA, matrix_layout LayoutB,
          matrix_layout LayoutC>
inline __SYCL_ALWAYS_INLINE typename std::enable_if<
    ((std::is_same<T1, int8_t>::value && std::is_same<T2, int32_t>::value) ||
     (std::is_same<T1, unsigned short>::value &&
      std::is_same<T2, float>::value)) &&
        (LayoutA == matrix_layout::row_major) &&
        (LayoutB == matrix_layout::packed_b) &&
        (LayoutC == matrix_layout::row_major),
    joint_matrix<Group, T2, NumRowsC, NumColsC, LayoutC>>::type
joint_matrix_mad(Group sg,
                 joint_matrix<Group, T1, NumRowsA, NumColsA, LayoutA> &jmA,
                 joint_matrix<Group, T1, NumRowsB, NumColsB, LayoutB> &jmB,
                 joint_matrix<Group, T2, NumRowsC, NumColsC, LayoutC> &jmC) {
  joint_matrix<Group, T2, NumRowsC, NumColsC, LayoutC> res(jmC);
  constexpr size_t epd = detail::elems_per_dword<T1>::value;
  // If A is large and C is small, in joint_matrix_load, we do memcpy for A, and
  // we do tileload for C whose shape is not tile_size*tile_size*4. In
  // joint_matrix_mad, we do tileload for A and shape is tile_size*tile_size*4.
  // So we need to reshape C before we do dpbssd.
  bool Cshouldreload = res.isSmall && !jmA.isSmall && !jmB.isSmall;
  bool Ashouldreload = jmA.isSmall && !jmB.isSmall;
  bool Bshouldreload = jmB.isSmall && !jmA.isSmall;

  for (int m = 0; m < res.trows; ++m) {
    for (int n = 0; n < res.tcols; ++n) {
      detail::submatrix<T2> sub_c;

      // AMX: 8 register tiles : 1k byte size, SMmaxxSKmax =16x64
      submatrix_load(sub_c, res, m * tile_size, n * tile_size, res.stride,
                     matrix_layout::row_major, Cshouldreload);
      for (int k = 0; k < jmA.tcols; ++k) { // K->int8_t
        detail::submatrix<T1> sub_a;
        detail::submatrix<T1> sub_b;
        submatrix_load(sub_a, jmA, m * tile_size, k * tile_size * epd,
                       jmA.stride, matrix_layout::packed_a, Ashouldreload);
        // Assume we alreay in vnni format.
        submatrix_load(sub_b, jmB, k * tile_size, n * tile_size * epd,
                       jmB.stride, matrix_layout::packed_b, Bshouldreload);
        submatrix_mad(sub_a, sub_b, sub_c);
      }
      submatrix_store(sub_c, res, m * tile_size, n * tile_size, res.stride,
                      matrix_layout::row_major, Cshouldreload);
    }
  }
  return res;
}

} // namespace experimental::matrix
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
