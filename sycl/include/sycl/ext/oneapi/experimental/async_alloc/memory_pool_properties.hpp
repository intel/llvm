//==------ memory_pool_properties.hpp --- SYCL asynchronous allocation -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/ext/oneapi/properties/property.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

// Forward declare memory_pool.
class memory_pool;

// Property that determines the initial threshold of a memory pool.
struct initial_threshold
    : detail::run_time_property_key<initial_threshold,
                                    detail::PropKind::InitialThreshold> {
  initial_threshold(size_t initialThreshold) : value(initialThreshold) {}
  size_t value;
};

using initial_threshold_key = initial_threshold;
inline bool operator==(const initial_threshold &lhs,
                       const initial_threshold &rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(const initial_threshold &lhs,
                       const initial_threshold &rhs) {
  return !(lhs == rhs);
}

// Property that determines the maximum size of a memory pool.
struct maximum_size
    : detail::run_time_property_key<maximum_size,
                                    detail::PropKind::MaximumSize> {
  maximum_size(size_t maxSize) : value(maxSize) {}
  size_t value;
};

using maximum_size_key = maximum_size;
inline bool operator==(const maximum_size &lhs, const maximum_size &rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(const maximum_size &lhs, const maximum_size &rhs) {
  return !(lhs == rhs);
}

// Property that initial allocations to a pool (not subsequent allocations from
// prior frees) are iniitialised to zero.
// enum class zero_init_enum { none, zero_init };
struct zero_init
    : detail::run_time_property_key<zero_init, detail::PropKind::ZeroInit> {
  zero_init() {};
};

using zero_init_key = zero_init;
inline bool operator==(const zero_init &, const zero_init &) { return true; }
inline bool operator!=(const zero_init &lhs, const zero_init &rhs) {
  return !(lhs == rhs);
}

template <>
struct is_property_key_of<initial_threshold_key, memory_pool> : std::true_type {
};

template <>
struct is_property_key_of<maximum_size_key, memory_pool> : std::true_type {};

template <>
struct is_property_key_of<zero_init_key, memory_pool> : std::true_type {};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
