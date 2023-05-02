//==----------- vector_traits.hpp - SYCL vector size queries --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/stl_type_traits.hpp>

#include <type_traits>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

// 4.10.2.6 Memory layout and alignment
template <typename T, int N>
struct vector_alignment_impl
    : std::conditional_t<N == 3, std::integral_constant<int, sizeof(T) * 4>,
                         std::integral_constant<int, sizeof(T) * N>> {};

template <typename T, int N>
struct vector_alignment
    : vector_alignment_impl<std::remove_cv_t<std::remove_reference_t<T>>, N> {};
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
