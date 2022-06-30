//==---- info_desc_helpers.hpp - SYCL information descriptor helpers -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/pi.hpp>
#include <sycl/info/info_desc.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
template <typename T> struct PiInfoCode;
template <typename T> struct is_platform_info_desc : std::false_type {};
template <typename T> struct is_context_info_desc : std::false_type {};
template <typename T> struct is_device_info_desc : std::false_type {};
template <typename T> struct is_queue_info_desc : std::false_type {};
template <typename T> struct is_kernel_info_desc : std::false_type {};
template <typename T>
struct is_kernel_device_specific_info_desc : std::false_type {};
template <typename T> struct is_event_info_desc : std::false_type {};
template <typename T> struct is_event_profiling_info_desc : std::false_type {};
#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template <> struct PiInfoCode<info::DescType::Desc> {                        \
    static constexpr pi_##DescType##_info value = PiCode;                      \
  };                                                                           \
  template <>                                                                  \
  struct is_##DescType##_info_desc<info::DescType::Desc> : std::true_type {};
#include <sycl/info/context_traits.def>
#include <sycl/info/event_traits.def>
#include <sycl/info/kernel_traits.def>
#include <sycl/info/platform_traits.def>
#include <sycl/info/queue_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC
#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template <> struct PiInfoCode<info::DescType::Desc> {                        \
    static constexpr pi_profiling_info value = PiCode;                         \
  };                                                                           \
  template <>                                                                  \
  struct is_##DescType##_info_desc<info::DescType::Desc> : std::true_type {};
#include <sycl/info/event_profiling_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

template <typename Param> struct IsSubGroupInfo : std::false_type {};
template <>
struct IsSubGroupInfo<info::kernel_device_specific::max_num_sub_groups>
    : std::true_type {};
template <>
struct IsSubGroupInfo<info::kernel_device_specific::compile_num_sub_groups>
    : std::true_type {};
template <>
struct IsSubGroupInfo<info::kernel_device_specific::max_sub_group_size>
    : std::true_type {};
template <>
struct IsSubGroupInfo<info::kernel_device_specific::compile_sub_group_size>
    : std::true_type {};

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template <> struct PiInfoCode<info::DescType::Desc> {                        \
    static constexpr                                                           \
        typename std::conditional<IsSubGroupInfo<info::DescType::Desc>::value, \
                                  pi_kernel_sub_group_info,                    \
                                  pi_kernel_group_info>::type value = PiCode;  \
  };                                                                           \
  template <>                                                                  \
  struct is_##DescType##_info_desc<info::DescType::Desc> : std::true_type {};
#define __SYCL_PARAM_TRAITS_SPEC_WITH_INPUT(DescType, Desc, ReturnT, InputT,   \
                                            PiCode)                            \
  __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)
#include <sycl/info/kernel_device_specific_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC_WITH_INPUT
#undef __SYCL_PARAM_TRAITS_SPEC
// Need a static_cast here since piDeviceGetInfo can also accept
// pi_usm_capability_query values.
#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template <> struct PiInfoCode<info::DescType::Desc> {                        \
    static constexpr pi_device_info value =                                    \
        static_cast<pi_device_info>(PiCode);                                   \
  };                                                                           \
  template <> struct is_##DescType##_info_desc<info::DescType::Desc> {         \
    static constexpr bool value = true;                                        \
  };
#include <sycl/info/device_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
