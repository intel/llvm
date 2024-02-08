//==------------- fpga_utils.hpp --- SYCL FPGA Reg Extensions --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/latency_control/properties.hpp> // for latency_co...

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

// Get the specified property from the given compile-time property list. If
// the property is not provided in the property list, get the default version of
// this property.
template <typename PropListT, typename PropKeyT, typename DefaultPropValT,
          typename = void>
struct GetOrDefaultValT {
  using type = DefaultPropValT;
};
template <typename PropListT, typename PropKeyT, typename DefaultPropValT>
struct GetOrDefaultValT<
    PropListT, PropKeyT, DefaultPropValT,
    std::enable_if_t<PropListT::template has_property<PropKeyT>()>> {
  using type = decltype(PropListT::template get_property<PropKeyT>());
};

// Default latency_anchor_id property for latency control, indicating the
// applied operation is not an anchor.
using defaultLatencyAnchorIdProperty = latency_anchor_id_key::value_t<-1>;
// Default latency_constraint property for latency control, indicating the
// applied operation is not a non-anchor.
using defaultLatencyConstraintProperty =
    latency_constraint_key::value_t<0, latency_control_type::none, 0>;

} // namespace ext::intel::experimental::detail
} // namespace _V1
} // namespace sycl
