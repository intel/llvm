//==-- task_sequence_properties.hpp - SYCL properties associated with ------==//
//==-- task_sequence -------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>
#include <sycl/ext/intel/experimental/fpga_kernel_properties.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::intel::experimental {
template <typename ReturnT, typename... ArgsT, ReturnT (&f)(ArgsT...),
          typename PropertyListT> class task_sequence;

struct balanced_key {
  using value_t = property_value<balanced_key>;
};

struct invocation_capacity_key {
  template <uint32_t Size>
  using value_t = property_value<invocation_capacity_key,
    std::integral_constant<uint32_t, Size>>;
};

struct response_capacity_key {
  template <uint32_t Size>
  using value_t = property_value<response_capacity_key,
    std::integral_constant<uint32_t, Size>>;
};

inline constexpr balanced_key::value_t balanced;
template <uint32_t Size>
inline constexpr invocation_capacity_key::value_t<Size> invocation_capacity;
template <uint32_t Size>
inline constexpr response_capacity_key::value_t<Size> response_capacity;

template <> struct is_property_key<balanced_key> : std::true_type {};
template <> struct is_property_key<invocation_capacity_key> : std::true_type {};
template <> struct is_property_key<response_capacity_key> : std::true_type {};

template <typename ReturnT, typename ... ArgsT, ReturnT(&f) (ArgsT...),
  class propertiesT>
struct is_property_key_of<balanced_key,
  task_sequence<f, propertiesT>> : std::true_type {};
template <typename ReturnT, typename ... ArgsT, ReturnT(&f) (ArgsT...),
  class propertiesT>
struct is_property_key_of<invocation_capacity_key,
  task_sequence<f, propertiesT>> : std::true_type {};
template <typename ReturnT, typename ... ArgsT, ReturnT(&f) (ArgsT...),
  class propertiesT>
struct is_property_key_of<response_capacity_key,
  task_sequence<f, propertiesT>> : std::true_type {};

// These fpga kernel properties are also supported with task_sequence
template <typename ReturnT, typename ... ArgsT, ReturnT(&f) (ArgsT...),
  class propertiesT>
struct is_property_key_of<use_stall_enable_clusters_key,
  task_sequence<f, propertiesT>> : std::true_type {};
template <typename ReturnT, typename ... ArgsT, ReturnT(&f) (ArgsT...),
  class propertiesT>
struct is_property_key_of<pipelined_key,
  task_sequence<f, propertiesT>> : std::true_type {};

namespace detail {
template <> struct PropertyToKind<balanced_key> {
  static constexpr PropKind Kind = PropKind::Balanced;
};
template <> struct PropertyToKind<invocation_capacity_key> {
  static constexpr PropKind Kind = PropKind::InvocationCapacity;
};
template <> struct PropertyToKind<response_capacity_key> {
  static constexpr PropKind Kind = PropKind::ResponseCapacity;
};

template <> struct IsCompileTimeProperty<balanced_key> : std::true_type {};
template <> struct IsCompileTimeProperty<invocation_capacity_key>
  : std::true_type {};
template <> struct IsCompileTimeProperty<response_capacity_key>
  : std::true_type {};

template <> struct PropertyMetaInfo<balanced_key::value_t> {
  static constexpr const char *name = "sycl-task-sequence-balanced";
  static constexpr std::nullptr_t value = nullptr;
};
template <int IC> struct PropertyMetaInfo<
  invocation_capacity_key::value_t<IC>> {
  static constexpr const char *name = "sycl-task-sequence-invocation-capacity";
  static constexpr int value = IC;
};
template <int RC> struct PropertyMetaInfo<
  invocation_capacity_key::value_t<IC>> {
  static constexpr const char *name = "sycl-task-sequence-invocation-capacity";
  static constexpr int value = IC;
};

} // namespace detail

} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl