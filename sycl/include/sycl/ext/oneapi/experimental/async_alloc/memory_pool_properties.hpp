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

// Property that provides a performance hint that all allocations from this pool
// will only be read from within SYCL kernel functions.
// enum class read_only_enum { none, read_only };
struct read_only
    : detail::run_time_property_key<read_only, detail::PropKind::ReadOnly> {
  // read_only(read_only_enum mode) : value(mode) {}
  read_only() {};

  // read_only_enum value;
};

using read_only_key = read_only;
inline bool operator==(const read_only &, const read_only &) { return true; }
inline bool operator!=(const read_only &lhs, const read_only &rhs) {
  return !(lhs == rhs);
}

// Property that initial allocations to a pool (not subsequent allocations from
// prior frees) are iniitialised to zero.
// enum class zero_init_enum { none, zero_init };
struct zero_init
    : detail::run_time_property_key<zero_init, detail::PropKind::ZeroInit> {
  // zero_init(zero_init_enum mode) : value(mode) {}
  zero_init() {};

  // zero_init_enum value;
};

using zero_init_key = zero_init;
inline bool operator==(const zero_init &, const zero_init &) { return true; }
inline bool operator!=(const zero_init &lhs, const zero_init &rhs) {
  return !(lhs == rhs);
}

template <>
struct is_property_key_of<initial_threshold, memory_pool> : std::true_type {};

template <>
struct is_property_key_of<maximum_size, memory_pool> : std::true_type {};

template <>
struct is_property_key_of<read_only, memory_pool> : std::true_type {};

template <>
struct is_property_key_of<zero_init, memory_pool> : std::true_type {};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
