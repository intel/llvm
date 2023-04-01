//==----------- accessor_properties.hpp --- SYCL accessor properties -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/common.hpp>
#include <sycl/detail/property_helper.hpp>
#include <sycl/ext/oneapi/accessor_property_list.hpp>
#include <sycl/properties/property_traits.hpp>
#include <type_traits>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace property {

class no_init : public detail::DataLessProperty<detail::NoInit> {};

class __SYCL2020_DEPRECATED("spelling is now: no_init") noinit
    : public detail::DataLessProperty<detail::NoInit> {};

} // namespace property

inline constexpr property::no_init no_init;

__SYCL2020_DEPRECATED("spelling is now: no_init")
inline constexpr property::noinit noinit;

namespace ext::intel {
namespace property {
struct __SYCL_TYPE(buffer_location) buffer_location {
  template <int A = 0> struct instance {
    template <int B>
    constexpr bool operator==(const buffer_location::instance<B> &) const {
      return A == B;
    }
    template <int B>
    constexpr bool operator!=(const buffer_location::instance<B> &) const {
      return A != B;
    }
    int get_location() { return A; }
  };
};
} // namespace property

template <int A>
inline constexpr property::buffer_location::instance<A> buffer_location{};
} // namespace ext::intel

namespace ext::oneapi {
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
struct __SYCL_TYPE(no_alias) no_alias {
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

inline constexpr property::no_offset::instance<> no_offset;
inline constexpr property::no_alias::instance<> no_alias;

template <>
struct is_compile_time_property<ext::oneapi::property::no_offset>
    : std::true_type {};
template <>
struct is_compile_time_property<ext::oneapi::property::no_alias>
    : std::true_type {};
template <>
struct is_compile_time_property<sycl::ext::intel::property::buffer_location>
    : std::true_type {};
} // namespace ext::oneapi

// Forward declaration
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
class accessor;
template <typename DataT, int Dimensions, access::mode AccessMode>
class host_accessor;

namespace detail::acc_properties {
template <typename T> struct is_accessor : std::false_type {};
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
struct is_accessor<accessor<DataT, Dimensions, AccessMode, AccessTarget,
                            IsPlaceholder, PropertyListT>> : std::true_type {};

template <typename T> struct is_host_accessor : std::false_type {};
template <typename DataT, int Dimensions, access::mode AccessMode>
struct is_host_accessor<host_accessor<DataT, Dimensions, AccessMode>>
    : std::true_type {};
} // namespace detail::acc_properties

// Accessor property trait specializations
template <>
struct is_property<ext::oneapi::property::no_offset> : std::true_type {};
template <>
struct is_property<ext::oneapi::property::no_alias> : std::true_type {};
template <>
struct is_property<ext::intel::property::buffer_location> : std::true_type {};

template <typename T>
struct is_property_of<property::noinit, T>
    : std::bool_constant<detail::acc_properties::is_accessor<T>::value ||
                         detail::acc_properties::is_host_accessor<T>::value> {};

template <typename T>
struct is_property_of<property::no_init, T>
    : std::bool_constant<detail::acc_properties::is_accessor<T>::value ||
                         detail::acc_properties::is_host_accessor<T>::value> {};

template <typename T>
struct is_property_of<ext::oneapi::property::no_offset, T>
    : std::bool_constant<detail::acc_properties::is_accessor<T>::value> {};

template <typename T>
struct is_property_of<ext::oneapi::property::no_alias, T>
    : std::bool_constant<detail::acc_properties::is_accessor<T>::value> {};

template <typename T>
struct is_property_of<ext::intel::property::buffer_location, T>
    : std::bool_constant<detail::acc_properties::is_accessor<T>::value> {};

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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
