//==---------------- numbers.hpp - math constants --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if __cplusplus >= 202002L && __has_include(<version>)
#include <version> // defines __cpp_lib_math_constants
#endif
#if __cpp_lib_math_constants
#include <numbers>
#endif

namespace sycl {
inline namespace _V1 {
namespace detail::numbers {

// std::numbers::pi once C++20 is the floor. Held as double rather than
// pi_v<T>: pi_v<sycl::half> is ill-formed, so callers convert at the use site.
#if __cpp_lib_math_constants
inline constexpr double pi = std::numbers::pi;
#else
inline constexpr double pi = 3.14159265358979323846;
#endif

} // namespace detail::numbers
} // namespace _V1
} // namespace sycl
