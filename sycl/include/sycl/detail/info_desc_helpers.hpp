//==---- info_desc_helpers.hpp - SYCL information descriptor helpers -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits> // for false_type

#include <sycl/detail/info_desc_traits.hpp>
#include <sycl/info/info_desc.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
// Each `is_*_info_desc<T>` matches self-describing trait structs that carry
// `info_class`, `return_type`, and (optionally) `ur_code` members. Mangling
// of `get_info` template arguments depends on the `return_type` alias exposed
// here, so the alias must remain stable across changes.
template <typename T>
struct is_platform_info_desc : is_info_desc_for<T, info_class::platform> {};
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

} // namespace detail
} // namespace _V1
} // namespace sycl
