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
namespace ext {
namespace intel {

template <template <int32_t> class _Type, class _T>
struct _MatchType : std::is_same<_Type<_T::value>, _T> {};

template <template <int32_t> class _Type, class... _T> struct _GetValue {
  static constexpr auto value = _Type<0>::default_value;
};

template <template <int32_t> class _Type, class _T1, class... _T>
struct _GetValue<_Type, _T1, _T...> {
  static constexpr auto value =
      detail::conditional_t<_MatchType<_Type, _T1>::value, _T1,
                            _GetValue<_Type, _T...>>::value;
};
} // namespace intel
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::intel' instead") INTEL {
  using namespace ext::intel;
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
