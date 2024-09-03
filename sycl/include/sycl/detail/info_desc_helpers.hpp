//==---- info_desc_helpers.hpp - SYCL information descriptor helpers -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <ur_api.h>

#include <type_traits> // for true_type

// FIXME: .def files included to this file use all sorts of SYCL objects like
// id, range, traits, etc. We have to include some headers before including .def
// files.
#include <sycl/aspects.hpp>
#include <sycl/id.hpp>
#include <sycl/info/info_desc.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
template <typename T> struct PiInfoCode;
template <typename T> struct UrInfoCode;
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

template <typename T> struct is_backend_info_desc : std::false_type {};
// Similar approach to limit valid get_backend_info template argument

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  template <> struct UrInfoCode<info::DescType::Desc> {                        \
    static constexpr ur_##DescType##_info_t value =                            \
        static_cast<ur_##DescType##_info_t>(UrCode);                           \
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

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  template <> struct UrInfoCode<info::DescType::Desc> {                        \
    static constexpr ur_profiling_info_t value = UrCode;                       \
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
template <typename Param> struct IsKernelInfo : std::false_type {};
template <>
struct IsKernelInfo<info::kernel_device_specific::ext_codeplay_num_regs>
    : std::true_type {};

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  template <> struct UrInfoCode<info::DescType::Desc> {                        \
    static constexpr typename std::conditional<                                \
        IsSubGroupInfo<info::DescType::Desc>::value,                           \
        ur_kernel_sub_group_info_t,                                            \
        std::conditional<IsKernelInfo<info::DescType::Desc>::value,            \
                         ur_kernel_info_t,                                     \
                         ur_kernel_group_info_t>::type>::type value = UrCode;  \
  };                                                                           \
  template <>                                                                  \
  struct is_##DescType##_info_desc<info::DescType::Desc> : std::true_type {    \
    using return_type = info::DescType::Desc::return_type;                     \
  };
#include <sycl/info/kernel_device_specific_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC
#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  template <> struct UrInfoCode<info::DescType::Desc> {                        \
    static constexpr ur_device_info_t value =                                  \
        static_cast<ur_device_info_t>(UrCode);                                 \
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
#define __SYCL_PARAM_TRAITS_SPEC(Namespace, DescType, Desc, ReturnT, UrCode)   \
  template <> struct UrInfoCode<Namespace::info::DescType::Desc> {             \
    static constexpr ur_device_info_t value =                                  \
        static_cast<ur_device_info_t>(UrCode);                                 \
  };                                                                           \
  template <>                                                                  \
  struct is_##DescType##_info_desc<Namespace::info::DescType::Desc>            \
      : std::true_type {                                                       \
    using return_type = Namespace::info::DescType::Desc::return_type;          \
  };
#include <sycl/info/ext_codeplay_device_traits.def>
#include <sycl/info/ext_intel_device_traits.def>
#include <sycl/info/ext_oneapi_device_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC
#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template <>                                                                  \
  struct is_backend_info_desc<info::DescType::Desc> : std::true_type {         \
    using return_type = info::DescType::Desc::return_type;                     \
  };
#include <sycl/info/sycl_backend_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace detail
} // namespace _V1
} // namespace sycl
