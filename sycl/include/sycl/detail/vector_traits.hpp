//==----------- vector_traits.hpp - SYCL vector size queries --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>   // for std::min and vs2017 win
#include <limits>      // for numeric_limits
#include <type_traits> // for integral_constant, conditional_t, remove_cv_t

namespace sycl {
inline namespace _V1 {
namespace detail {

// 4.10.2.6 Memory layout and alignment
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
// SYCL 2020 vec alignment requirements have been relaxed in
// KhronosGroup/SYCL-Docs#448. New specification wording only guarantees 64-byte
// alignment of vec class and we leverage this here to avoid dealing with MSVC
// limitations (see below).
constexpr size_t MaxVecAlignment = 64;
#else
// This version is preserved to maintain API/ABI compatibility with older
// releases.
// FIXME: drop this branch once API/ABI break is allowed

#if defined(_WIN32) && (_MSC_VER)
// MSVC Compiler doesn't allow using of function arguments with alignment
// requirements. MSVC Compiler Error C2719: 'parameter': formal parameter with
// __declspec(align('#')) won't be aligned. The align __declspec modifier
// is not permitted on function parameters. Function parameter alignment
// is controlled by the calling convention used.
// For more information, see Calling Conventions
// (https://docs.microsoft.com/en-us/cpp/cpp/calling-conventions).
// For information on calling conventions for x64 processors, see
// Calling Convention
// (https://docs.microsoft.com/en-us/cpp/build/x64-calling-convention).
constexpr size_t MaxVecAlignment = 64;
#else
// To match ABI of previos releases, we don't impose any restrictions on vec
// alignment on Linux
constexpr size_t MaxVecAlignment = std::numeric_limits<size_t>::max();
#endif

#endif // __INTEL_PREVIEW_BREAKING_CHANGES

template <typename T, size_t N>
struct vector_alignment_impl
    : std::conditional_t<
          N == 3,
          std::integral_constant<size_t,
                                 (std::min)(sizeof(T) * 4, MaxVecAlignment)>,
          std::integral_constant<size_t,
                                 (std::min)(sizeof(T) * N, MaxVecAlignment)>> {
};

template <typename T, size_t N>
struct vector_alignment
    : vector_alignment_impl<std::remove_cv_t<std::remove_reference_t<T>>, N> {};
} // namespace detail
} // namespace _V1
} // namespace sycl
