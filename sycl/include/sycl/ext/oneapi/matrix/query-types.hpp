//==--------------- query-types.hpp - SYCL matrix --------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

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
} // namespace _V1
} // namespace sycl
