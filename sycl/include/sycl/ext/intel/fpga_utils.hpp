//==------------- fpga_utils.hpp --- SYCL FPGA Reg Extensions --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>
#include <sycl/detail/stl_type_traits.hpp>
#include <sycl/stl.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::intel {

template <template <int32_t> class Type, class T>
struct _MatchType : std::is_same<Type<T::value>, T> {};

template <template <int32_t> class Type, class... T> struct _GetValue {
  static constexpr auto value = Type<0>::default_value;
};

template <template <int32_t> class Type, class T1, class... T>
struct _GetValue<Type, T1, T...> {
  static constexpr auto value =
      detail::conditional_t<_MatchType<Type, T1>::value, T1,
                            _GetValue<Type, T...>>::value;
};
} // namespace ext::intel

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
