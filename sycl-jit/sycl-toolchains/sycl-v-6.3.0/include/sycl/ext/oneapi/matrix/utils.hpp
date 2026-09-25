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
