//==----------- mm_host_properties.hpp --- __mm_host properties -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/property_helper.hpp>
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
struct interface_idx
  template <int A = 0> struct instance {
    template <int B>
    constexpr bool operator==(const interface_idx::instance<B> &) const {
      return A == B;
    }
    template <int B>
    constexpr bool operator!=(const interface_idx::instance<B> &) const {
      return A != B;
    }
  };
};
} // namespace property
#if __cplusplus > 201402L
template <int i>
inline constexpr property::interface_idx::instance<i> interface_idx {};

#endif
} // namespace intel
} // namespace ext

#if __cplusplus > 201402L
inline constexpr property::interface_idx::instance interface_idx{};
};
#endif

template <>
struct is_compile_time_property<sycl::ext::intel::property::interface_idx>
    : std::true_type {};
} // namespace oneapi
} // namespace ext

namespace detail {
template <int I>
struct IsCompileTimePropertyInstance<
    ext::intel::property::id_interface_idx::instance<I>> : std::true_type {};
template <>
 // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
