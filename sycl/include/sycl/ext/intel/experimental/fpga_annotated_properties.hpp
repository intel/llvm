//==-- fpga_annotated_properties.hpp - SYCL properties associated with
// annotated_arg/ptr --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#include <cstdint>
#include <iosfwd>
#include <tuple>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel {
namespace experimental {

template <typename PropertyT, typename... Ts>
using property_value =
    sycl::ext::oneapi::experimental::property_value<PropertyT, Ts...>;
//===----------------------------------------------------------------------===//
//        FPGA properties of annotated_arg/annotated_ptr
//===----------------------------------------------------------------------===//
struct register_map_key {
  using value_t = property_value<register_map_key>;
};

struct conduit_key {
  using value_t = property_value<conduit_key>;
};

struct stable_key {
  using value_t = property_value<stable_key>;
};

struct buffer_location_key {
  template <int K>
  using value_t =
      property_value<buffer_location_key, std::integral_constant<int, K>>;
};

struct awidth_key {
  template <int K>
  using value_t = property_value<awidth_key, std::integral_constant<int, K>>;
};

struct dwidth_key {
  template <int K>
  using value_t = property_value<dwidth_key, std::integral_constant<int, K>>;
};

struct latency_key {
  template <int K>
  using value_t = property_value<latency_key, std::integral_constant<int, K>>;
};

enum class read_write_mode_enum : std::uint16_t { read, write, read_write };

struct read_write_mode_key {
  template <read_write_mode_enum Mode>
  using value_t =
      property_value<read_write_mode_key,
                     std::integral_constant<read_write_mode_enum, Mode>>;
};

struct maxburst_key {
  template <int K>
  using value_t = property_value<maxburst_key, std::integral_constant<int, K>>;
};

struct wait_request_key {
  template <int K>
  using value_t =
      property_value<wait_request_key, std::integral_constant<int, K>>;
};

// non-mmhost properties
inline constexpr register_map_key::value_t register_map;
inline constexpr conduit_key::value_t conduit;
inline constexpr stable_key::value_t stable;

// mmhost properties
template <int N>
inline constexpr buffer_location_key::value_t<N> buffer_location;
template <int W> inline constexpr awidth_key::value_t<W> awidth;
template <int W> inline constexpr dwidth_key::value_t<W> dwidth;
template <int N> inline constexpr latency_key::value_t<N> latency;
template <int N> inline constexpr maxburst_key::value_t<N> maxburst;
template <int Enable>
inline constexpr wait_request_key::value_t<Enable> wait_request;
inline constexpr wait_request_key::value_t<1> wait_request_requested;
inline constexpr wait_request_key::value_t<0> wait_request_not_requested;

template <read_write_mode_enum Mode>
inline constexpr read_write_mode_key::value_t<Mode> read_write_mode;
inline constexpr read_write_mode_key::value_t<read_write_mode_enum::read>
    read_write_mode_read;
inline constexpr read_write_mode_key::value_t<read_write_mode_enum::write>
    read_write_mode_write;
inline constexpr read_write_mode_key::value_t<read_write_mode_enum::read_write>
    read_write_mode_readwrite;

} // namespace experimental
} // namespace intel

namespace oneapi {
namespace experimental {
template <typename T, typename PropertyListT> class annotated_arg;
template <typename T, typename PropertyListT> class annotated_ptr;

struct alignment_key;
using register_map_key = intel::experimental::register_map_key;
using conduit_key = intel::experimental::conduit_key;
using stable_key = intel::experimental::stable_key;
using buffer_location_key = intel::experimental::buffer_location_key;
using awidth_key = intel::experimental::awidth_key;
using dwidth_key = intel::experimental::dwidth_key;
using latency_key = intel::experimental::latency_key;
using read_write_mode_key = intel::experimental::read_write_mode_key;
using maxburst_key = intel::experimental::maxburst_key;
using wait_request_key = intel::experimental::wait_request_key;
using read_write_mode_enum = intel::experimental::read_write_mode_enum;

template <> struct is_property_key<register_map_key> : std::true_type {};
template <> struct is_property_key<conduit_key> : std::true_type {};
template <> struct is_property_key<stable_key> : std::true_type {};

template <> struct is_property_key<buffer_location_key> : std::true_type {};
template <> struct is_property_key<awidth_key> : std::true_type {};
template <> struct is_property_key<dwidth_key> : std::true_type {};
template <> struct is_property_key<latency_key> : std::true_type {};
template <> struct is_property_key<read_write_mode_key> : std::true_type {};
template <> struct is_property_key<maxburst_key> : std::true_type {};
template <> struct is_property_key<wait_request_key> : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<register_map_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<conduit_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<stable_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<buffer_location_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<awidth_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<dwidth_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<latency_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<read_write_mode_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<maxburst_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<wait_request_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<register_map_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<conduit_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<stable_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<buffer_location_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<awidth_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<dwidth_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<latency_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<read_write_mode_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<maxburst_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<wait_request_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

namespace detail {
template <> struct PropertyToKind<register_map_key> {
  static constexpr PropKind Kind = PropKind::RegisterMap;
};
template <> struct PropertyToKind<conduit_key> {
  static constexpr PropKind Kind = PropKind::Conduit;
};
template <> struct PropertyToKind<stable_key> {
  static constexpr PropKind Kind = PropKind::Stable;
};
template <> struct PropertyToKind<buffer_location_key> {
  static constexpr PropKind Kind = PropKind::BufferLocation;
};
template <> struct PropertyToKind<awidth_key> {
  static constexpr PropKind Kind = PropKind::AddrWidth;
};
template <> struct PropertyToKind<dwidth_key> {
  static constexpr PropKind Kind = PropKind::DataWidth;
};
template <> struct PropertyToKind<latency_key> {
  static constexpr PropKind Kind = PropKind::Latency;
};
template <> struct PropertyToKind<read_write_mode_key> {
  static constexpr PropKind Kind = PropKind::RWMode;
};
template <> struct PropertyToKind<maxburst_key> {
  static constexpr PropKind Kind = PropKind::MaxBurst;
};
template <> struct PropertyToKind<wait_request_key> {
  static constexpr PropKind Kind = PropKind::WaitRequest;
};

template <> struct IsCompileTimeProperty<register_map_key> : std::true_type {};
template <> struct IsCompileTimeProperty<conduit_key> : std::true_type {};
template <> struct IsCompileTimeProperty<stable_key> : std::true_type {};

template <>
struct IsCompileTimeProperty<buffer_location_key> : std::true_type {};
template <> struct IsCompileTimeProperty<awidth_key> : std::true_type {};
template <> struct IsCompileTimeProperty<dwidth_key> : std::true_type {};
template <>
struct IsCompileTimeProperty<read_write_mode_key> : std::true_type {};
template <> struct IsCompileTimeProperty<latency_key> : std::true_type {};
template <> struct IsCompileTimeProperty<maxburst_key> : std::true_type {};
template <> struct IsCompileTimeProperty<wait_request_key> : std::true_type {};

template <> struct PropertyMetaInfo<register_map_key::value_t> {
  static constexpr const char *name = "sycl-register-map";
  static constexpr std::nullptr_t value = nullptr;
};
template <> struct PropertyMetaInfo<conduit_key::value_t> {
  static constexpr const char *name = "sycl-conduit";
  static constexpr std::nullptr_t value = nullptr;
};
template <> struct PropertyMetaInfo<stable_key::value_t> {
  static constexpr const char *name = "sycl-stable";
  static constexpr std::nullptr_t value = nullptr;
};

template <int N> struct PropertyMetaInfo<buffer_location_key::value_t<N>> {
  static constexpr const char *name = "sycl-buffer-location";
  static constexpr int value = N;
};
template <int W> struct PropertyMetaInfo<awidth_key::value_t<W>> {
  static constexpr const char *name = "sycl-awidth";
  static constexpr int value = W;
};
template <int W> struct PropertyMetaInfo<dwidth_key::value_t<W>> {
  static constexpr const char *name = "sycl-dwidth";
  static constexpr int value = W;
};
template <int N> struct PropertyMetaInfo<latency_key::value_t<N>> {
  static constexpr const char *name = "sycl-latency";
  static constexpr int value = N;
};
template <int N> struct PropertyMetaInfo<maxburst_key::value_t<N>> {
  static constexpr const char *name = "sycl-maxburst";
  static constexpr int value = N;
};
template <int Enable>
struct PropertyMetaInfo<wait_request_key::value_t<Enable>> {
  static constexpr const char *name = "sycl-wait-request";
  static constexpr int value = Enable;
};
template <read_write_mode_enum Mode>
struct PropertyMetaInfo<read_write_mode_key::value_t<Mode>> {
  static constexpr const char *name = "sycl-read-write-mode";
  static constexpr read_write_mode_enum value = Mode;
};

} // namespace detail

// 'buffer_location' and mmhost properties are pointers-only
template <typename T, int N>
struct is_valid_property<T, buffer_location_key::value_t<N>>
    : std::bool_constant<std::is_pointer_v<T>> {};

template <typename T, int W>
struct is_valid_property<T, awidth_key::value_t<W>>
    : std::bool_constant<std::is_pointer_v<T>> {};

template <typename T, int W>
struct is_valid_property<T, dwidth_key::value_t<W>>
    : std::bool_constant<std::is_pointer_v<T>> {};

template <typename T, int N>
struct is_valid_property<T, latency_key::value_t<N>>
    : std::bool_constant<std::is_pointer_v<T>> {};

template <typename T, read_write_mode_enum Mode>
struct is_valid_property<T, read_write_mode_key::value_t<Mode>>
    : std::bool_constant<std::is_pointer_v<T>> {};

template <typename T, int N>
struct is_valid_property<T, maxburst_key::value_t<N>>
    : std::bool_constant<std::is_pointer_v<T>> {};

template <typename T, int Enable>
struct is_valid_property<T, wait_request_key::value_t<Enable>>
    : std::bool_constant<std::is_pointer_v<T>> {};

// 'register_map',  'conduit',  'stable' are common properties for pointers
// and non pointers;
template <typename T>
struct is_valid_property<T, register_map_key::value_t> : std::true_type {};
template <typename T>
struct is_valid_property<T, conduit_key::value_t> : std::true_type {};
template <typename T>
struct is_valid_property<T, stable_key::value_t> : std::true_type {};

// buffer_location is applied on PtrAnnotation
template <>
struct propagateToPtrAnnotation<buffer_location_key> : std::true_type {};

//===----------------------------------------------------------------------===//
//   Utility for FPGA properties
//===----------------------------------------------------------------------===//
//
namespace detail {
template <typename... Args> struct checkValidFPGAPropertySet {
  using list = std::tuple<Args...>;
  static constexpr bool has_BufferLocation =
      ContainsProperty<buffer_location_key, list>::value;

  static constexpr bool has_InterfaceConfig =
      ContainsProperty<awidth_key, list>::value ||
      ContainsProperty<dwidth_key, list>::value ||
      ContainsProperty<latency_key, list>::value ||
      ContainsProperty<read_write_mode_key, list>::value ||
      ContainsProperty<maxburst_key, list>::value ||
      ContainsProperty<wait_request_key, list>::value;

  static constexpr bool value = !(!has_BufferLocation && has_InterfaceConfig);
};

template <typename... Args> struct checkHasConduitAndRegisterMap {
  using list = std::tuple<Args...>;
  static constexpr bool has_Conduit =
      ContainsProperty<conduit_key, list>::value;
  static constexpr bool has_RegisterMap =
      ContainsProperty<register_map_key, list>::value;
  static constexpr bool value = !(has_Conduit && has_RegisterMap);
};
} // namespace detail

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
