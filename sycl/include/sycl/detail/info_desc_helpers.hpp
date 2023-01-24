//==---- info_desc_helpers.hpp - SYCL information descriptor helpers -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aspects.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/info/info_desc.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
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
// Normally we would just use std::enable_if to limit valid get_info template
// arguments. However, there is a mangling mismatch of
// "std::enable_if<is*_desc::value>::type" between gcc clang (it appears that
// gcc lacks a E terminator for unresolved-qualifier-level sequence). As a
// workaround, we use return_type alias from is_*info_desc that doesn't run into
// the same problem.
// TODO remove once this gcc/clang discrepancy is resolved
#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template <> struct PiInfoCode<info::DescType::Desc> {                        \
    static constexpr pi_##DescType##_info value = PiCode;                      \
  };                                                                           \
  template <>                                                                  \
  struct is_##DescType##_info_desc<info::DescType::Desc> : std::true_type {    \
    using return_type = info::DescType::Desc::return_type;                     \
  };
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
  struct is_##DescType##_info_desc<info::DescType::Desc> : std::true_type {    \
    using return_type = info::DescType::Desc::return_type;                     \
  };
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
  struct is_##DescType##_info_desc<info::DescType::Desc> : std::true_type {    \
    using return_type = info::DescType::Desc::return_type;                     \
  };
#include <sycl/info/kernel_device_specific_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC
// Need a static_cast here since piDeviceGetInfo can also accept
// pi_usm_capability_query values.
#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template <> struct PiInfoCode<info::DescType::Desc> {                        \
    static constexpr pi_device_info value =                                    \
        static_cast<pi_device_info>(PiCode);                                   \
  };                                                                           \
  template <>                                                                  \
  struct is_##DescType##_info_desc<info::DescType::Desc> : std::true_type {    \
    using return_type = info::DescType::Desc::return_type;                     \
  };
#define __SYCL_PARAM_TRAITS_SPEC_SPECIALIZED(DescType, Desc, ReturnT, PiCode)  \
  __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)

#include <sycl/info/device_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC
#undef __SYCL_PARAM_TRAITS_SPEC_SPECIALIZED

#define __SYCL_PARAM_TRAITS_SPEC(Namespace, DescType, Desc, ReturnT, PiCode)   \
  template <> struct PiInfoCode<Namespace::info::DescType::Desc> {             \
    static constexpr pi_device_info value =                                    \
        static_cast<pi_device_info>(PiCode);                                   \
  };                                                                           \
  template <>                                                                  \
  struct is_##DescType##_info_desc<Namespace::info::DescType::Desc>            \
      : std::true_type {                                                       \
    using return_type = Namespace::info::DescType::Desc::return_type;          \
  };
#include <sycl/info/ext_intel_device_traits.def>
#include <sycl/info/ext_oneapi_device_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
