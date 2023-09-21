//===------- utils.hpp - SYCL matrix extension ----*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/multi_ptr.hpp>

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

} // ext::oneapi::experimental::matrix

namespace detail {

// Helper to return decorated pointer for different values
// of access::decorated parameter.
// If access::decorated::legacy is removed in the future
// this helper usage can be replaced with ptr.get_decorated().
template <typename DecorT, typename T, access::address_space Space,
          access::decorated IsDecorated>
DecorT *getDecorated(multi_ptr<T, Space, IsDecorated> ptr) {
  if constexpr (IsDecorated == access::decorated::legacy)
    return ptr.get();
  else
    return ptr.get_decorated();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
