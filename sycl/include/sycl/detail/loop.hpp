//==---------- loop.hpp ---- Compile-time unrolled loop helper -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>     // for size_t
#include <type_traits> // for integral_constant
#include <utility>     // for forward, integer_sequence, make_index_sequence

namespace sycl {
inline namespace _V1 {
namespace detail {

// To ensure loop unrolling is done when processing dimensions.
template <size_t... Inds, class F>
constexpr void loop_impl(std::integer_sequence<size_t, Inds...>, F &&f) {
  (f(std::integral_constant<size_t, Inds>{}), ...);
}

template <size_t count, class F> constexpr void loop(F &&f) {
  loop_impl(std::make_index_sequence<count>{}, std::forward<F>(f));
}

} // namespace detail
} // namespace _V1
} // namespace sycl
