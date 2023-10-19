//==------------- fpga_utils.hpp --- SYCL FPGA Reg Extensions --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/latency_control/properties.hpp> // for latency_co...
#include <sycl/ext/oneapi/properties/properties.hpp> // for empty_properties_t

#include <stdint.h>    // for int32_t
#include <type_traits> // for conditional_t

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental::detail {

template <template <int32_t> class _Type, class _T>
struct _MatchType : std::is_same<_Type<_T::value>, _T> {};

template <template <int32_t> class _Type, class... _T> struct _GetValue {
  static constexpr auto value = _Type<0>::default_value;
};

template <template <int32_t> class _Type, class _T1, class... _T>
struct _GetValue<_Type, _T1, _T...> {
  static constexpr auto value =
      std::conditional_t<_MatchType<_Type, _T1>::value, _T1,
                         _GetValue<_Type, _T...>>::value;
};

template <typename T, typename PropertyListT =
                          ext::oneapi::experimental::empty_properties_t>
class AnnotatedMemberValue {
  static_assert(oneapi::experimental::is_property_list<PropertyListT>::value,
                "Property list is invalid.");
};

template <typename T, typename... Props>
class AnnotatedMemberValue<
    T, oneapi::experimental::detail::properties_t<Props...>> {
public:
  AnnotatedMemberValue() {}
  AnnotatedMemberValue(T Value) : MValue(Value) {}
  T MValue [[__sycl_detail__::add_ir_annotations_member(
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::name...,
      ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::value...)]];
};

} // namespace ext::intel::experimental::detail
} // namespace _V1
} // namespace sycl
