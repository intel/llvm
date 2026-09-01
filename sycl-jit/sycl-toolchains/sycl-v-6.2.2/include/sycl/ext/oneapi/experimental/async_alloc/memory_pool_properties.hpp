//==------ memory_pool_properties.hpp --- SYCL asynchronous allocation -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <sycl/properties/property_traits.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

// Forward declare memory_pool.
class memory_pool;

namespace property::memory_pool {

// Property that determines the initial threshold of a memory pool.
struct initial_threshold : public sycl::detail::PropertyWithData<
                               sycl::detail::MemPoolInitialThreshold> {
  initial_threshold(size_t initialThreshold)
      : initialThreshold(initialThreshold) {};
  size_t get_initial_threshold() { return initialThreshold; }

private:
  size_t initialThreshold;
};

// Property that determines the maximum size of a memory pool.
struct maximum_size
    : public sycl::detail::PropertyWithData<sycl::detail::MemPoolMaximumSize> {
  maximum_size(size_t maxSize) : maxSize(maxSize) {};
  size_t get_maximum_size() { return maxSize; }

private:
  size_t maxSize;
};

// Property that provides a performance hint that all allocations from this pool
// will only be read from within SYCL kernel functions.
struct read_only
    : public sycl::detail::DataLessProperty<sycl::detail::MemPoolReadOnly> {
  read_only() = default;
};

// Property that initial allocations to a pool (not subsequent allocations
// from prior frees) are iniitialised to zero.
struct zero_init
    : public sycl::detail::DataLessProperty<sycl::detail::MemPoolZeroInit> {
  zero_init() = default;
};
} // namespace property::memory_pool
} // namespace ext::oneapi::experimental

template <>
struct is_property<
    sycl::ext::oneapi::experimental::property::memory_pool::initial_threshold>
    : std::true_type {};

template <>
struct is_property<
    sycl::ext::oneapi::experimental::property::memory_pool::maximum_size>
    : std::true_type {};

template <>
struct is_property<
    sycl::ext::oneapi::experimental::property::memory_pool::read_only>
    : std::true_type {};

template <>
struct is_property<
    sycl::ext::oneapi::experimental::property::memory_pool::zero_init>
    : std::true_type {};

} // namespace _V1
} // namespace sycl
