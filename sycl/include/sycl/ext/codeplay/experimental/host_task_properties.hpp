//==-------- host_task_properties.hpp --- SYCL host task properties --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/context.hpp>
#include <sycl/detail/property_helper.hpp>
#include <sycl/properties/property_traits.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::codeplay::experimental::property::host_task {

class manual_interop_sync : public ::sycl::detail::DataLessProperty<
                                ::sycl::detail::HostTaskManualInteropSync> {};

} // namespace ext::codeplay::experimental::property::host_task

// Forward declaration
class host_task;

template <>
struct is_property<
    ext::codeplay::experimental::property::host_task::manual_interop_sync>
    : std::true_type {};

template <>
struct is_property_of<
    ext::codeplay::experimental::property::host_task::manual_interop_sync,
    host_task> : std::true_type {};

} // namespace _V1
} // namespace sycl
