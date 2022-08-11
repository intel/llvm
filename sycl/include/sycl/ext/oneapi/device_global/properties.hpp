//==----- properties.hpp - SYCL properties associated with device_global ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

template <typename T, typename PropertyListT> class device_global;

struct device_image_scope_key {
  using value_t = property_value<device_image_scope_key>;
};

enum class host_access_enum : std::uint16_t { read, write, read_write, none };

struct host_access_key {
  template <host_access_enum Access>
  using value_t =
      property_value<host_access_key,
                     std::integral_constant<host_access_enum, Access>>;
};

enum class init_mode_enum : std::uint16_t { reprogram, reset };

struct init_mode_key {
  template <init_mode_enum Trigger>
  using value_t =
      property_value<init_mode_key,
                     std::integral_constant<init_mode_enum, Trigger>>;
};

struct implement_in_csr_key {
  template <bool Enable>
  using value_t =
      property_value<implement_in_csr_key, sycl::detail::bool_constant<Enable>>;
};

#if __cplusplus >= 201703L // inline variables
inline constexpr device_image_scope_key::value_t device_image_scope;

template <host_access_enum Access>
inline constexpr host_access_key::value_t<Access> host_access;
inline constexpr host_access_key::value_t<host_access_enum::read>
    host_access_read;
inline constexpr host_access_key::value_t<host_access_enum::write>
    host_access_write;
inline constexpr host_access_key::value_t<host_access_enum::read_write>
    host_access_read_write;
inline constexpr host_access_key::value_t<host_access_enum::none>
    host_access_none;

template <init_mode_enum Trigger>
inline constexpr init_mode_key::value_t<Trigger> init_mode;
inline constexpr init_mode_key::value_t<init_mode_enum::reprogram>
    init_mode_reprogram;
inline constexpr init_mode_key::value_t<init_mode_enum::reset> init_mode_reset;

template <bool Enable>
inline constexpr implement_in_csr_key::value_t<Enable> implement_in_csr;
inline constexpr implement_in_csr_key::value_t<true> implement_in_csr_on;
inline constexpr implement_in_csr_key::value_t<false> implement_in_csr_off;
#endif // __cplusplus >= 201703L

template <> struct is_property_key<device_image_scope_key> : std::true_type {};
template <> struct is_property_key<host_access_key> : std::true_type {};
template <> struct is_property_key<init_mode_key> : std::true_type {};
template <> struct is_property_key<implement_in_csr_key> : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<device_image_scope_key,
                          device_global<T, PropertyListT>> : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<host_access_key, device_global<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<init_mode_key, device_global<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<implement_in_csr_key, device_global<T, PropertyListT>>
    : std::true_type {};

namespace detail {
template <> struct PropertyToKind<device_image_scope_key> {
  static constexpr PropKind Kind = PropKind::DeviceImageScope;
};
template <> struct PropertyToKind<host_access_key> {
  static constexpr PropKind Kind = PropKind::HostAccess;
};
template <> struct PropertyToKind<init_mode_key> {
  static constexpr PropKind Kind = PropKind::InitMode;
};
template <> struct PropertyToKind<implement_in_csr_key> {
  static constexpr PropKind Kind = PropKind::ImplementInCSR;
};

template <>
struct IsCompileTimeProperty<device_image_scope_key> : std::true_type {};
template <> struct IsCompileTimeProperty<host_access_key> : std::true_type {};
template <> struct IsCompileTimeProperty<init_mode_key> : std::true_type {};
template <>
struct IsCompileTimeProperty<implement_in_csr_key> : std::true_type {};

template <> struct PropertyMetaInfo<device_image_scope_key::value_t> {
  static constexpr const char *name = "sycl-device-image-scope";
  static constexpr std::nullptr_t value = nullptr;
};
template <host_access_enum Access>
struct PropertyMetaInfo<host_access_key::value_t<Access>> {
  static constexpr const char *name = "sycl-host-access";
  static constexpr host_access_enum value = Access;
};
template <init_mode_enum Trigger>
struct PropertyMetaInfo<init_mode_key::value_t<Trigger>> {
  static constexpr const char *name = "sycl-init-mode";
  static constexpr init_mode_enum value = Trigger;
};
template <bool Enable>
struct PropertyMetaInfo<implement_in_csr_key::value_t<Enable>> {
  static constexpr const char *name = "sycl-implement-in-csr";
  static constexpr bool value = Enable;
};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
