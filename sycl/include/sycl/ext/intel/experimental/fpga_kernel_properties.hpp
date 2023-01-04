//==----- fpga_kernel_properties.hpp - SYCL properties associated with FPGA
// kernel properties ---==//
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
namespace ext::intel::experimental {

template <typename T, typename PropertyListT> class fpga_kernel_attribute;

enum class streaming_interface_options_enum : std::uint16_t {
  accept_downstream_stall,
  remove_downstream_stall
};

enum class register_map_interface_options_enum : std::uint16_t {
  wait_for_done_write,
  do_not_wait_for_done_write
};

struct streaming_interface_key {
  template <streaming_interface_options_enum option>
  using value_t = ext::oneapi::experimental::property_value<
      streaming_interface_key,
      std::integral_constant<streaming_interface_options_enum, option>>;
};

struct register_map_interface_key {
  template <register_map_interface_options_enum option>
  using value_t = ext::oneapi::experimental::property_value<
      register_map_interface_key,
      std::integral_constant<register_map_interface_options_enum, option>>;
};

struct pipelined_key {
  template <int pipeline_directive_or_initiation_interval>
  using value_t = ext::oneapi::experimental::property_value<
      pipelined_key,
      std::integral_constant<int, pipeline_directive_or_initiation_interval>>;
};

template <streaming_interface_options_enum option>
inline constexpr streaming_interface_key::value_t<option> streaming_interface;

inline constexpr streaming_interface_key::value_t<
    streaming_interface_options_enum::accept_downstream_stall>
    streaming_interface_accept_downstream_stall;

inline constexpr streaming_interface_key::value_t<
    streaming_interface_options_enum::remove_downstream_stall>
    streaming_interface_remove_downstream_stall;

template <register_map_interface_options_enum option>
inline constexpr register_map_interface_key::value_t<option>
    register_map_interface;

inline constexpr register_map_interface_key::value_t<
    register_map_interface_options_enum::wait_for_done_write>
    register_map_interface_wait_for_done_write;

inline constexpr register_map_interface_key::value_t<
    register_map_interface_options_enum::do_not_wait_for_done_write>
    register_map_interface_do_not_wait_for_done_write;

template <int pipeline_directive_or_initiation_interval = -1>
inline constexpr pipelined_key::value_t<
    pipeline_directive_or_initiation_interval>
    pipelined;

} // namespace ext::intel::experimental

namespace ext::oneapi::experimental {
template <>
struct is_property_key<intel::experimental::streaming_interface_key>
    : std::true_type {};
template <>
struct is_property_key<intel::experimental::register_map_interface_key>
    : std::true_type {};
template <>
struct is_property_key<intel::experimental::pipelined_key> : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<
    intel::experimental::streaming_interface_key,
    intel::experimental::fpga_kernel_attribute<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<
    intel::experimental::register_map_interface_key,
    intel::experimental::fpga_kernel_attribute<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<
    intel::experimental::pipelined_key,
    intel::experimental::fpga_kernel_attribute<T, PropertyListT>>
    : std::true_type {};

namespace detail {
template <>
struct PropertyToKind<intel::experimental::streaming_interface_key> {
  static constexpr PropKind Kind = StreamingInterface;
};
template <>
struct PropertyToKind<intel::experimental::register_map_interface_key> {
  static constexpr PropKind Kind = RegisterMapInterface;
};
template <> struct PropertyToKind<intel::experimental::pipelined_key> {
  static constexpr PropKind Kind = Pipelined;
};

template <>
struct IsCompileTimeProperty<intel::experimental::streaming_interface_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::register_map_interface_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<intel::experimental::pipelined_key>
    : std::true_type {};

template <intel::experimental::streaming_interface_options_enum Stall>
struct PropertyMetaInfo<
    intel::experimental::streaming_interface_key::value_t<Stall>> {
  static constexpr const char *name = "sycl-streaming-interface";
  static constexpr intel::experimental::streaming_interface_options_enum value =
      Stall;
};
template <intel::experimental::register_map_interface_options_enum Wait>
struct PropertyMetaInfo<
    intel::experimental::register_map_interface_key::value_t<Wait>> {
  static constexpr const char *name = "sycl-register-map-interface";
  static constexpr intel::experimental::register_map_interface_options_enum
      value = Wait;
};
template <int Value>
struct PropertyMetaInfo<intel::experimental::pipelined_key::value_t<Value>> {
  static constexpr const char *name = "sycl-pipelined";
  static constexpr int value = Value;
};

} // namespace detail
} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
