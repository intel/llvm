//===- common.hpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits>

#include <sycl/half_type.hpp>

namespace sycl {
inline namespace _V1 {

namespace ext {
namespace oneapi {
namespace experimental {

////////////////////////////////////////////////////////////////////////////////
/// FORWARD DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

template <class _Tp> struct is_genfloat;

template <class _Tp, class _Enable = void> class complex;

template <class _Tp>
class complex<_Tp, typename std::enable_if_t<is_genfloat<_Tp>::value>>;

////////////////////////////////////////////////////////////////////////////////
/// TRAITS
////////////////////////////////////////////////////////////////////////////////

template <class _Tp>
struct is_genfloat
    : std::integral_constant<bool, std::is_same_v<_Tp, double> ||
                                       std::is_same_v<_Tp, float> ||
                                       std::is_same_v<_Tp, sycl::half>> {};

template <class _Tp>
struct is_gencomplex
    : std::integral_constant<bool,
                             std::is_same_v<_Tp, complex<double>> ||
                                 std::is_same_v<_Tp, complex<float>> ||
                                 std::is_same_v<_Tp, complex<sycl::half>>> {};

////////////////////////////////////////////////////////////////////////////////
/// DEFINES
////////////////////////////////////////////////////////////////////////////////

#define _SYCL_EXT_CPLX_INLINE_VISIBILITY                                       \
  inline __attribute__((__visibility__("hidden"), __always_inline__))

} // namespace experimental
} // namespace oneapi
} // namespace ext

} // namespace _V1
} // namespace sycl
