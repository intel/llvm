//==----------- queue_properties.hpp --- SYCL queue properties -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/property_helper.hpp>
#include <CL/sycl/properties/property_traits.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace property {
namespace queue {
class in_order : public detail::DataLessProperty<detail::InOrder> {};
class enable_profiling
    : public detail::DataLessProperty<detail::QueueEnableProfiling> {};
} // namespace queue
} // namespace property

namespace ext {
namespace oneapi {

namespace property {
namespace queue {
class discard_events : public ::cl::sycl::detail::DataLessProperty<
                           ::cl::sycl::detail::DiscardEvents> {};
} // namespace queue
} // namespace property

namespace cuda {
namespace property {
namespace queue {
class use_default_stream : public ::cl::sycl::detail::DataLessProperty<
                               ::cl::sycl::detail::UseDefaultStream> {};
} // namespace queue
} // namespace property
} // namespace cuda
} // namespace oneapi
} // namespace ext

namespace property {
namespace queue {
namespace __SYCL2020_DEPRECATED(
    "use 'sycl::ext::oneapi::cuda::property::queue' instead") cuda {
  class use_default_stream : public ::cl::sycl::ext::oneapi::cuda::property::
                                 queue::use_default_stream {};
} // namespace cuda
} // namespace queue
} // namespace property

// Forward declaration
class queue;

// Queue property trait specializations
template <> struct is_property<property::queue::in_order> : std::true_type {};
template <>
struct is_property<property::queue::enable_profiling> : std::true_type {};
template <>
struct is_property<ext::oneapi::property::queue::discard_events>
    : std::true_type {};
template <>
struct is_property<property::queue::cuda::use_default_stream> : std::true_type {
};
template <>
struct is_property<ext::oneapi::cuda::property::queue::use_default_stream>
    : std::true_type {};

template <>
struct is_property_of<property::queue::in_order, queue> : std::true_type {};
template <>
struct is_property_of<property::queue::enable_profiling, queue>
    : std::true_type {};
template <>
struct is_property_of<ext::oneapi::property::queue::discard_events, queue>
    : std::true_type {};
template <>
struct is_property_of<property::queue::cuda::use_default_stream, queue>
    : std::true_type {};
template <>
struct is_property_of<ext::oneapi::cuda::property::queue::use_default_stream,
                      queue> : std::true_type {};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
