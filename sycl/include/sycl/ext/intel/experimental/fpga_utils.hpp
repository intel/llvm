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
#include <tuple>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {

enum class type {
  none, // default
  exact,
  max,
  min
};

template <int32_t _N> struct latency_anchor_id {
  static constexpr int32_t value = _N;
  static constexpr int32_t default_value = -1;
};

template <int32_t _N1, type _N2, int32_t _N3> struct latency_constraint {
  static constexpr std::tuple<int32_t, type, int32_t> value = {_N1, _N2, _N3};
  static constexpr std::tuple<int32_t, type, int32_t> default_value = {
      0, type::none, 0};
};

using ignoreParam_int_t = int32_t;
constexpr ignoreParam_int_t IgnoreParamInt{};
using ignoreParam_enum_t = type;
constexpr ignoreParam_enum_t IgnoreParamEnum{};

template <class _VType, class _T> struct _ValueExtractorImp {
  static constexpr auto _First = _T::value;
  static constexpr auto _Second = IgnoreParamEnum;
  static constexpr auto _Third = IgnoreParamInt;
};

template <class _VTypeFirst, class _VTypeSecond, class _VTypeThird, class _T>
struct _ValueExtractorImp<
    const std::tuple<_VTypeFirst, _VTypeSecond, _VTypeThird>, _T> {
  static constexpr auto _First = std::get<0>(_T::value);
  static constexpr auto _Second = std::get<1>(_T::value);
  static constexpr auto _Third = std::get<2>(_T::value);
};

template <class _T>
struct _ValueExtractor : _ValueExtractorImp<decltype(_T::value), _T> {};

template <class _VTypeFirst, class _VTypeSecond, class _VTypeThird,
          template <_VTypeFirst, _VTypeSecond, _VTypeThird> class _Type,
          class _T>
struct _MatchType
    : std::is_same<
          _Type<_ValueExtractor<_T>::_First, _ValueExtractor<_T>::_Second,
                _ValueExtractor<_T>::_Third>,
          _T> {};

template <class _VTypeFirst, class _VTypeSecond, class _VTypeThird,
          template <_VTypeFirst, _VTypeSecond, _VTypeThird> class _Type,
          class... _T>
struct _GetValue3 {
  static constexpr auto value =
      _Type<_VTypeFirst{}, _VTypeSecond{}, _VTypeThird{}>::default_value;
};

template <class _VTypeFirst, class _VTypeSecond, class _VTypeThird,
          template <_VTypeFirst, _VTypeSecond, _VTypeThird> class _Type,
          class _T1, class... _T>
struct _GetValue3<_VTypeFirst, _VTypeSecond, _VTypeThird, _Type, _T1, _T...> {
  static constexpr auto value = std::conditional<
      _MatchType<_VTypeFirst, _VTypeSecond, _VTypeThird, _Type, _T1>::value,
      _T1, _GetValue3<_VTypeFirst, _VTypeSecond, _VTypeThird, _Type, _T...>>::
      type::value;
};

template <class _VType, template <_VType> class _Type, class... _T>
struct _GetValue {
private:
  template <_VType _V1, ignoreParam_enum_t, ignoreParam_int_t>
  using _Type2 = _Type<_V1>;

public:
  static constexpr auto value =
      _GetValue3<_VType, ignoreParam_enum_t, ignoreParam_int_t, _Type2,
                 _T...>::value;
};

} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
