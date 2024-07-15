//==---------------- common.hpp - DPC++ Explicit SIMD API   ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Common definitions used in experimental Explicit SIMD APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/esimd/common.hpp>
#include <sycl/ext/intel/esimd/detail/defines_elementary.hpp>
#include <sycl/ext/intel/esimd/native/common.hpp>
#include <sycl/ext/intel/esimd/xmx/common.hpp>

#include <cstdint>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental::esimd {

/// @addtogroup sycl_esimd_core
/// @{

using lsc_data_size = __ESIMD_DNS::lsc_data_size;

namespace detail {

using lsc_vector_size = __ESIMD_DNS::lsc_vector_size;

using lsc_data_order = __ESIMD_DNS::lsc_data_order;

template <lsc_vector_size VS> constexpr void check_lsc_vector_size() {
  __ESIMD_DNS::check_lsc_vector_size<VS>();
}

template <int VS> constexpr void check_lsc_vector_size() {
  __ESIMD_DNS::check_lsc_vector_size<VS>();
}

template <typename T, lsc_data_size DS> constexpr void check_lsc_data_size() {
  __ESIMD_DNS::check_lsc_data_size<T, DS>();
}

template <lsc_vector_size VS> constexpr uint8_t to_int() {
  return __ESIMD_DNS::to_int<VS>();
}

template <int VS> constexpr lsc_vector_size to_lsc_vector_size() {
  return __ESIMD_DNS::to_lsc_vector_size<VS>();
}

template <typename T, lsc_data_size DS>
constexpr lsc_data_size finalize_data_size() {
  return __ESIMD_DNS::finalize_data_size<T, DS>();
}

constexpr lsc_data_size expand_data_size(lsc_data_size DS) {
  return __ESIMD_DNS::expand_data_size(DS);
}

template <typename T> struct lsc_expand_type {
  using type = __ESIMD_DNS::lsc_expand_type<T>::type;
};

} // namespace detail

/// L1 or L2 cache hint kinds.
using cache_hint = __ESIMD_NS::cache_hint;

/// Represents a split barrier action.
enum class split_barrier_action : uint8_t {
  wait = 0,   // split barrier wait
  signal = 1, // split barrier signal
};

/// @} sycl_esimd_core

} // namespace ext::intel::experimental::esimd
} // namespace _V1
} // namespace sycl
