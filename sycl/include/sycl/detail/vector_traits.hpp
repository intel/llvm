//==----------- vector_traits.hpp - SYCL vector size queries --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>   // for std::min and vs2017 win
#include <type_traits> // for integral_constant, conditional_t, remove_cv_t

namespace sycl {
inline namespace _V1 {
namespace detail {

// 4.10.2.6 Memory layout and alignment
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
// due to MSVC the maximum alignment for sycl::vec is 64 and this proposed
// change is being brought to the spec committee.
constexpr size_t MaxVecAlignment = 64;
template <typename T, size_t N>
struct vector_alignment_impl
    : std::conditional_t<
          N == 3,
          std::integral_constant<size_t,
                                 (std::min)(sizeof(T) * 4, MaxVecAlignment)>,
          std::integral_constant<size_t,
                                 (std::min)(sizeof(T) * N, MaxVecAlignment)>> {
};
#else
template <typename T, size_t N>
struct vector_alignment_impl
    : std::conditional_t<N == 3, std::integral_constant<int, sizeof(T) * 4>,
                         std::integral_constant<int, sizeof(T) * N>> {};
#endif

template <typename T, size_t N>
struct vector_alignment
    : vector_alignment_impl<std::remove_cv_t<std::remove_reference_t<T>>, N> {};
} // namespace detail
} // namespace _V1
} // namespace sycl
