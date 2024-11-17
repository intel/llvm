//==--------------------------- ur_info_code.hpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/info_desc_helpers.hpp>

#include <ur_api.h>

namespace sycl {
inline namespace _V1 {
namespace detail {
template <typename T> struct UrInfoCode;

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  template <> struct UrInfoCode<info::DescType::Desc> {                        \
    static constexpr ur_##DescType##_info_t value =                            \
        static_cast<ur_##DescType##_info_t>(UrCode);                           \
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
  };
#include <sycl/info/event_profiling_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  template <> struct UrInfoCode<info::DescType::Desc> {                        \
    static constexpr typename std::conditional<                                \
        IsSubGroupInfo<info::DescType::Desc>::value,                           \
        ur_kernel_sub_group_info_t,                                            \
        std::conditional<IsKernelInfo<info::DescType::Desc>::value,            \
                         ur_kernel_info_t,                                     \
                         ur_kernel_group_info_t>::type>::type value = UrCode;  \
  };
#include <sycl/info/kernel_device_specific_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  template <> struct UrInfoCode<info::DescType::Desc> {                        \
    static constexpr ur_device_info_t value =                                  \
        static_cast<ur_device_info_t>(UrCode);                                 \
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
  };

#include <sycl/info/ext_codeplay_device_traits.def>
#include <sycl/info/ext_intel_device_traits.def>
#include <sycl/info/ext_oneapi_device_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace detail
} // namespace _V1
} // namespace sycl
