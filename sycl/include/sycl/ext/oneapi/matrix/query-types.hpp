//==--------------- query-types.hpp - SYCL matrix --------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/matrix-unified-utils.hpp>

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

namespace detail {
template <typename T> constexpr const char *convertTypeToMatrixTypeString() {
  if (std::is_same_v<T, sycl::ext::oneapi::bfloat16>)
    return "matrix_type::bf16";
  else if (std::is_same_v<T, sycl::half>)
    return "matrix_type::fp16";
  else if (std::is_same_v<
               T, sycl::ext::oneapi::experimental::matrix::precision::tf32>)
    return "matrix_type::tf32";
  else if (std::is_same_v<T, float>)
    return "matrix_type::fp32";
  else if (std::is_same_v<T, double>)
    return "matrix_type::fp64";
  else if (std::is_same_v<T, int8_t>)
    return "matrix_type::sint8";
  else if (std::is_same_v<T, int16_t>)
    return "matrix_type::sint16";
  else if (std::is_same_v<T, int32_t>)
    return "matrix_type::sint32";
  else if (std::is_same_v<T, int64_t>)
    return "matrix_type::sint64";
  else if (std::is_same_v<T, uint8_t>)
    return "matrix_type::uint8";
  else if (std::is_same_v<T, uint16_t>)
    return "matrix_type::uint16";
  else if (std::is_same_v<T, uint32_t>)
    return "matrix_type::uint32";
  else if (std::is_same_v<T, uint64_t>)
    return "matrix_type::uint64";
  return "unsupported-type";
}
} // namespace detail

} // namespace ext::oneapi::experimental::matrix
} // namespace _V1
} // namespace sycl
