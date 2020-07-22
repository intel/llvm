//==------------- fpga_utils.hpp --- SYCL FPGA Reg Extensions --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/stl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {

template <template <int32_t> class Type, class T>
struct MatchType : std::is_same<Type<T::value>, T> {};

template <template <int32_t> class Type, class... T> struct GetValue {
  static constexpr auto value = Type<0>::default_value;
};

template <template <int32_t> class Type, class T1, class... T>
struct GetValue<Type, T1, T...> {
  static constexpr auto value =
      std::conditional<MatchType<Type, T1>::value, T1,
                       GetValue<Type, T...>>::type::value;
};
} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
