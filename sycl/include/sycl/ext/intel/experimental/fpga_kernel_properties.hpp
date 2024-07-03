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
inline namespace _V1 {
namespace ext::intel::experimental {

template <typename T, typename PropertyListT> class fpga_kernel_attribute;

enum class streaming_interface_options_enum : uint16_t {
  accept_downstream_stall,
  remove_downstream_stall
};

enum class register_map_interface_options_enum : uint16_t {
  do_not_wait_for_done_write,
  wait_for_done_write,
};

struct streaming_interface_key
    : oneapi::experimental::detail::compile_time_property_key<
          oneapi::experimental::detail::PropKind::StreamingInterface> {
  template <streaming_interface_options_enum option>
  using value_t = ext::oneapi::experimental::property_value<
      streaming_interface_key,
      std::integral_constant<streaming_interface_options_enum, option>>;
};

struct register_map_interface_key
    : oneapi::experimental::detail::compile_time_property_key<
          oneapi::experimental::detail::PropKind::RegisterMapInterface> {
  template <register_map_interface_options_enum option>
  using value_t = ext::oneapi::experimental::property_value<
      register_map_interface_key,
      std::integral_constant<register_map_interface_options_enum, option>>;
};

struct pipelined_key : oneapi::experimental::detail::compile_time_property_key<
                           oneapi::experimental::detail::PropKind::Pipelined> {
  template <int pipeline_directive_or_initiation_interval>
  using value_t = ext::oneapi::experimental::property_value<
      pipelined_key,
      std::integral_constant<int, pipeline_directive_or_initiation_interval>>;
};

template <streaming_interface_options_enum option =
              streaming_interface_options_enum::accept_downstream_stall>
inline constexpr streaming_interface_key::value_t<option> streaming_interface;

inline constexpr streaming_interface_key::value_t<
    streaming_interface_options_enum::accept_downstream_stall>
    streaming_interface_accept_downstream_stall;

inline constexpr streaming_interface_key::value_t<
    streaming_interface_options_enum::remove_downstream_stall>
    streaming_interface_remove_downstream_stall;

template <register_map_interface_options_enum option =
              register_map_interface_options_enum::do_not_wait_for_done_write>
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
template <intel::experimental::streaming_interface_options_enum Stall_Free>
struct PropertyMetaInfo<
    intel::experimental::streaming_interface_key::value_t<Stall_Free>> {
  static constexpr const char *name = "sycl-streaming-interface";
  static constexpr intel::experimental::streaming_interface_options_enum value =
      Stall_Free;
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
} // namespace _V1
} // namespace sycl
