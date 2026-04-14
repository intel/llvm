//==---------------- helpers.hpp - SYCL helpers ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/builder.hpp>
#include <sycl/detail/spirv_memory_semantics.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Function to get or store id, item, nd_item, group for the host
// implementation. Pass nullptr to get stored object. Pass valid address to
// store object.
template <typename T> T get_or_store(const T *obj) {
  static thread_local auto stored = *obj;
  if (obj != nullptr) {
    stored = *obj;
  }
  return stored;
}

inline constexpr bool is_power_of_two(int x) { return (x & (x - 1)) == 0; }

} // namespace detail
} // namespace _V1
} // namespace sycl
