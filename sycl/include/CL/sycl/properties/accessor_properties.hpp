//==----------- accessor_properties.hpp --- SYCL accessor properties -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/property_helper.hpp>
#include <sycl/ext/oneapi/accessor_property_list.hpp>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace property {

class no_init : public detail::DataLessProperty<detail::NoInit> {};

class __SYCL2020_DEPRECATED("spelling is now: no_init") noinit
    : public detail::DataLessProperty<detail::NoInit> {};

} // namespace property

#if __cplusplus > 201402L

__SYCL_INLINE_CONSTEXPR property::no_init no_init;

__SYCL2020_DEPRECATED("spelling is now: no_init")
__SYCL_INLINE_CONSTEXPR property::noinit noinit;

#else

namespace {

constexpr const auto &no_init =
    sycl::detail::InlineVariableHelper<property::no_init>::value;

constexpr const auto &noinit __SYCL2020_DEPRECATED("spelling is now: no_init") =
    sycl::detail::InlineVariableHelper<property::noinit>::value;
} // namespace

#endif

namespace ext {
namespace intel {
namespace property {
struct buffer_location {
  template <int A = 0> struct instance {
    template <int B>
    constexpr bool operator==(const buffer_location::instance<B> &) const {
      return A == B;
    }
    template <int B>
    constexpr bool operator!=(const buffer_location::instance<B> &) const {
      return A != B;
    }
  };
};
} // namespace property
#if __cplusplus > 201402L
template <int A>
inline constexpr property::buffer_location::instance<A> buffer_location{};
#endif
} // namespace intel
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::intel' instead") INTEL {
  using namespace ext::intel;
}
namespace ext {
namespace oneapi {
namespace property {
struct no_offset {
  template <bool B = true> struct instance {
    constexpr bool operator==(const no_offset::instance<B> &) const {
      return true;
    }
    constexpr bool operator!=(const no_offset::instance<B> &) const {
      return false;
    }
  };
};
struct no_alias {
  template <bool B = true> struct instance {
    constexpr bool operator==(const no_alias::instance<B> &) const {
      return true;
    }
    constexpr bool operator!=(const no_alias::instance<B> &) const {
      return false;
    }
  };
};
} // namespace property

#if __cplusplus > 201402L

inline constexpr property::no_offset::instance no_offset;
inline constexpr property::no_alias::instance no_alias;

#endif

template <>
struct is_compile_time_property<ext::oneapi::property::no_offset>
    : std::true_type {};
template <>
struct is_compile_time_property<ext::oneapi::property::no_alias>
    : std::true_type {};
template <>
struct is_compile_time_property<sycl::ext::intel::property::buffer_location>
    : std::true_type {};
} // namespace oneapi
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
  using namespace ext::oneapi;
}
namespace detail {
template <int I>
struct IsCompileTimePropertyInstance<
    ext::intel::property::buffer_location::instance<I>> : std::true_type {};
template <>
struct IsCompileTimePropertyInstance<
    ext::oneapi::property::no_alias::instance<>> : std::true_type {};
template <>
struct IsCompileTimePropertyInstance<
    ext::oneapi::property::no_offset::instance<>> : std::true_type {};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
