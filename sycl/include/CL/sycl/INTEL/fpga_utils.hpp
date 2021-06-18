//==------------- fpga_utils.hpp --- SYCL FPGA Reg Extensions --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/stl_type_traits.hpp>
#include <CL/sycl/stl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace INTEL {

template <class _D, class _T>
struct _MatchType : std::is_same<typename _D::type_id, typename _T::type_id> {};

template <class _D, class... _T>
struct _GetValue;

template <class _D>
struct _GetValue<_D> : std::integral_constant<decltype(_D::value), _D::value> {};

template <class _D, class _T1, class... _T>
struct _GetValue<_D, _T1, _T...> {
  template <class _D2, class _T12, class _Enable = void>
  struct impl : 
    std::integral_constant<decltype(_D::value), _GetValue<_D, _T...>::value> {};

  template <class _D2, class _T12>
  struct impl<_D2, _T12, std::enable_if_t<_MatchType<_D2, _T12>::value>> : 
    std::integral_constant<decltype(_D::value), _T1::value> {};

  static constexpr auto value = impl<_D, _T1>::value;
};

} // namespace INTEL
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
