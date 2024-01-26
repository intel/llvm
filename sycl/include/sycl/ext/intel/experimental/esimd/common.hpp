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

/// The scope that lsc_fence operation should apply to
/// Supported platforms: DG2, PVC
enum class __SYCL_DEPRECATED(
    "use sycl::ext::intel::esimd::fence_scope") lsc_scope : uint8_t {
  group = 0,  /// flush out to the threadgroup's scope
  local = 1,  /// flush out to the local scope
  tile = 2,   /// tile, flush out to several DSSs
  gpu = 3,    /// entire GPU, flush out to the GPUs LLC
  gpus = 4,   /// all GPUs in the system, flush out to memory shared by all GPUs
  system = 5, /// the entire system memory space
  sysacq = 6, /// the entire system memory space with system-acquire semantics
};

/// The lsc_fence operation to apply to caches
/// Supported platforms: DG2, PVC
enum class __SYCL_DEPRECATED("use sycl::ext::intel::esimd::fence_flush_op")
    lsc_fence_op : uint8_t {
      none = 0,       /// no operation
      evict = 1,      /// dirty lines evicted and invalidated from L1
      invalidate = 2, /// invalidate all clean lines
      discard = 3,    /// direct and clean lines are discarded w/o eviction
      clean = 4,   /// dirty lines are written to memory, but retained in cache
                   /// in clean state
      flushl3 = 5, /// flush only L3
    };

/// The specific LSC shared function to fence with lsc_fence
/// Supported platforms: DG2, PVC
enum class __SYCL_DEPRECATED("use sycl::ext::intel::esimd::memory_kind")
    lsc_memory_kind : uint8_t {
      untyped_global = 0,         /// untyped global memory
      untyped_global_low_pri = 1, /// low-priority untyped global memory
      typed_global = 2,           /// typed global memory
      shared_local = 3,           /// shared local memory
    };

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

template <typename T> struct lsc_bitcast_type {
public:
  using type = __ESIMD_DNS::lsc_bitcast_type<T>::type;
};

} // namespace detail

/// L1 or L3 cache hint kinds.
using cache_hint = __ESIMD_NS::cache_hint;

namespace detail {
// TODO: These enum and the function are kept here temporarily, while used
// inside this header file here. Remove it after all experimental functions
// working with cache hints are moved out of experimental namespace.
using lsc_action = __ESIMD_DNS::cache_action;
template <lsc_action Action, cache_hint L1, cache_hint L3>
constexpr void check_lsc_cache_hint() {
  __ESIMD_DNS::check_cache_hint<Action, L1, L3>();
}
} // namespace detail

/// Represents a split barrier action.
enum class split_barrier_action : uint8_t {
  wait = 0,   // split barrier wait
  signal = 1, // split barrier signal
};

/// @} sycl_esimd_core

} // namespace ext::intel::experimental::esimd
} // namespace _V1
} // namespace sycl
