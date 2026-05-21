//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits> // for enable_if_t, remove_const_t

namespace sycl {
inline namespace _V1 {
namespace detail {
template <typename DataT> class aligned_allocator;
} // namespace detail

template <typename T, int dimensions = 1,
          typename AllocatorT =
              detail::aligned_allocator<std::remove_const_t<T>>,
          typename __Enabled =
              std::enable_if_t<(dimensions > 0) && (dimensions <= 3)>>
class buffer;
} // namespace _V1
} // namespace sycl
