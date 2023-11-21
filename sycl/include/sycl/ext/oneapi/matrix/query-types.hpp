//==--------------- query-types.hpp - SYCL matrix --------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <sycl/ext/oneapi/bfloat16.hpp>                    // for bfloat16
#include <sycl/ext/oneapi/matrix/matrix-unified-utils.hpp> // for tf32

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental::matrix {

enum class matrix_type {
  bf16,
  fp16,
  tf32,
  fp32,
  fp64,
  sint8,
  sint16,
  sint32,
  sint64,
  uint8,
  uint16,
  uint32,
  uint64
};

struct combination {
  size_t max_msize;
  size_t max_nsize;
  size_t max_ksize;
  size_t msize;
  size_t nsize;
  size_t ksize;
  matrix_type atype;
  matrix_type btype;
  matrix_type ctype;
  matrix_type dtype;
};

} // namespace ext::oneapi::experimental::matrix

// Type to matrix type string conversion used in compile-time
namespace detail {
template <typename T> constexpr const char *convertTypeToMatrixTypeString() {
  return "";
}
template <>
constexpr const char *
convertTypeToMatrixTypeString<sycl::ext::oneapi::bfloat16>() {
  return "matrix_type::bf16";
}
template <> constexpr const char *convertTypeToMatrixTypeString<sycl::half>() {
  return "matrix_type::fp16";
}
template <>
constexpr const char *convertTypeToMatrixTypeString<
    sycl::ext::oneapi::experimental::matrix::precision::tf32>() {
  return "matrix_type::tf32";
}
template <> constexpr const char *convertTypeToMatrixTypeString<float>() {
  return "matrix_type::fp32";
}
template <> constexpr const char *convertTypeToMatrixTypeString<double>() {
  return "matrix_type::fp64";
}
template <> constexpr const char *convertTypeToMatrixTypeString<int8_t>() {
  return "matrix_type::sint8";
}
template <> constexpr const char *convertTypeToMatrixTypeString<int16_t>() {
  return "matrix_type::sint16";
}
template <> constexpr const char *convertTypeToMatrixTypeString<int32_t>() {
  return "matrix_type::sint32";
}
template <> constexpr const char *convertTypeToMatrixTypeString<int64_t>() {
  return "matrix_type::sint64";
}
template <> constexpr const char *convertTypeToMatrixTypeString<uint8_t>() {
  return "matrix_type::uint8";
}
template <> constexpr const char *convertTypeToMatrixTypeString<uint16_t>() {
  return "matrix_type::uint16";
}
template <> constexpr const char *convertTypeToMatrixTypeString<uint32_t>() {
  return "matrix_type::uint32";
}
template <> constexpr const char *convertTypeToMatrixTypeString<uint64_t>() {
  return "matrix_type::uint64";
}
} // namespace detail
} // namespace _V1
} // namespace sycl
