//==----------- accessor_properties.hpp --- SYCL accessor properties -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "CL/sycl/accessor_property_list.hpp"
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/property_helper.hpp>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace property {

class noinit : public detail::DataLessProperty<detail::NoInit> {};

} // namespace property

#if __cplusplus > 201402L

inline constexpr property::noinit noinit;

#else

namespace {

constexpr const auto &noinit =
    sycl::detail::InlineVariableHelper<property::noinit>::value;
}

#endif

namespace ext {
namespace INTEL {
namespace property {
template <int A> struct buffer_location {
  constexpr bool operator==(const buffer_location<A> &) const { return true; }
  constexpr bool operator!=(const buffer_location<A> &) const { return false; }
  template <bool B>
  constexpr bool operator==(const buffer_location<B> &) const {
    return false;
  }
  template <bool B>
  constexpr bool operator!=(const buffer_location<B> &) const {
    return true;
  }
};

} // namespace property
} // namespace INTEL
namespace ONEAPI {
namespace property {
template <bool A = true> struct no_offset {
  constexpr bool operator==(const no_offset<A> &) const {
    return true;
  }
  constexpr bool operator!=(const no_offset<A> &) const {
    return false;
  }
  template <bool B>
  constexpr bool operator==(const no_offset<B> &) const {
    return false;
  }
  template <bool B>
  constexpr bool operator!=(const no_offset<B> &) const {
    return true;
  }
};
template <bool A = true> struct no_alias {
  constexpr bool operator==(const no_alias<A> &) const {
    return true;
  }
  constexpr bool operator!=(const no_alias<A> &) const {
    return false;
  }
  template <bool B>
  constexpr bool operator==(const no_alias<B> &) const {
    return false;
  }
  template <bool B>
  constexpr bool operator!=(const no_alias<B> &) const {
    return true;
  }

};
} // namespace property

#if __cplusplus > 201402L

inline constexpr property::no_offset no_offset;
inline constexpr property::no_alias no_alias;

#else

namespace {
constexpr const auto &no_offset =
    sycl::detail::InlineVariableHelper<property::no_offset<>>::value;
constexpr const auto &no_alias =
    sycl::detail::InlineVariableHelper<property::no_alias<>>::value;
} // namespace

#endif

template <bool B>
struct is_compile_time_property<ext::ONEAPI::property::no_offset<B>>
    : std::true_type {};
template <bool B>
struct is_compile_time_property<ext::ONEAPI::property::no_alias<B>>
    : std::true_type {};
template <int I>
struct is_compile_time_property<ext::INTEL::property::buffer_location<I>>
    : std::true_type {};
} // namespace ONEAPI
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
