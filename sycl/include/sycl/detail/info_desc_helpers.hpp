//==---- info_desc_helpers.hpp - SYCL information descriptor helpers -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits> // for true_type

// FIXME: .def files included to this file use all sorts of SYCL objects like
// id, range, traits, etc. We have to include some headers before including .def
// files.
#include <sycl/aspects.hpp>
#include <sycl/detail/info_desc_traits.hpp>
#include <sycl/id.hpp>
#include <sycl/info/info_desc.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
// Primary templates derive from `is_info_desc_for<T, info_class::X>`, which
// matches new self-describing traits (those carrying `info_class`, `return_type`
// and `ur_code` members). Explicit specializations emitted below from the .def
// files override the primary template for legacy traits. Both forms coexist
// while migration is in progress.
template <typename T>
struct is_platform_info_desc
    : is_info_desc_for<T, info_class::platform> {};
template <typename T>
struct is_context_info_desc : is_info_desc_for<T, info_class::context> {};
template <typename T>
struct is_device_info_desc : is_info_desc_for<T, info_class::device> {};
template <typename T>
struct is_queue_info_desc : is_info_desc_for<T, info_class::queue> {};
template <typename T>
struct is_kernel_info_desc : is_info_desc_for<T, info_class::kernel> {};
template <typename T>
struct is_kernel_device_specific_info_desc
    : is_info_desc_for<T, info_class::kernel_device_specific> {};
template <typename T>
struct is_kernel_queue_specific_info_desc
    : is_info_desc_for<T, info_class::kernel_queue_specific> {};
template <typename T>
struct is_event_info_desc : is_info_desc_for<T, info_class::event> {};
template <typename T>
struct is_event_profiling_info_desc
    : is_info_desc_for<T, info_class::event_profiling> {};
// Normally we would just use std::enable_if to limit valid get_info template
// arguments. However, there is a mangling mismatch of
// "std::enable_if<is*_desc::value>::type" between gcc clang (it appears that
// gcc lacks a E terminator for unresolved-qualifier-level sequence). As a
// workaround, we use return_type alias from is_*info_desc that doesn't run into
// the same problem.
// TODO remove once this gcc/clang discrepancy is resolved

template <typename T> struct is_backend_info_desc : std::false_type {};
// Similar approach to limit valid get_backend_info template argument

#define __SYCL_PARAM_TRAITS_SPEC(Namespace, DescType, Desc, ReturnT, UrCode)   \
  template <>                                                                  \
  struct is_##DescType##_info_desc<Namespace::info::DescType::Desc>            \
      : std::true_type {                                                       \
    using return_type = Namespace::info::DescType::Desc::return_type;          \
  };
#include <sycl/info/ext_intel_kernel_info_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC


#define __SYCL_PARAM_TRAITS_SPEC(Namespace, DescType, Desc, ReturnT, UrCode)   \
  template <>                                                                  \
  struct is_##DescType##_info_desc<Namespace::info::DescType::Desc>            \
      : std::true_type {                                                       \
    using return_type = Namespace::info::DescType::Desc::return_type;          \
  };

#define __SYCL_PARAM_TRAITS_TEMPLATE_PARTIAL_SPEC(Namespace, Desctype, Desc,   \
                                                  ReturnT, UrCode)             \
  template <int Dimensions>                                                    \
  struct is_##Desctype##_info_desc<                                            \
      Namespace::info::Desctype::Desc<Dimensions>> : std::true_type {          \
    using return_type =                                                        \
        typename Namespace::info::Desctype::Desc<Dimensions>::return_type;     \
  };

#include <sycl/info/ext_oneapi_kernel_queue_specific_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC
#undef __SYCL_PARAM_TRAITS_TEMPLATE_PARTIAL_SPEC

} // namespace detail
} // namespace _V1
} // namespace sycl
