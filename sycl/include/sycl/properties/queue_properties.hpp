//==----------- queue_properties.hpp --- SYCL queue properties -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/property_helper.hpp>
#include <sycl/properties/property_traits.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace property::queue {
class in_order : public detail::DataLessProperty<detail::InOrder> {};
class enable_profiling
    : public detail::DataLessProperty<detail::QueueEnableProfiling> {};
} // namespace property::queue

namespace ext::oneapi {

namespace property::queue {
class discard_events
    : public ::sycl::detail::DataLessProperty<::sycl::detail::DiscardEvents> {};

class priority_normal
    : public sycl::detail::DataLessProperty<sycl::detail::QueuePriorityNormal> {
};
class priority_low
    : public sycl::detail::DataLessProperty<sycl::detail::QueuePriorityLow> {};
class priority_high
    : public sycl::detail::DataLessProperty<sycl::detail::QueuePriorityHigh> {};

} // namespace property::queue

namespace cuda::property::queue {
class use_default_stream : public ::sycl::detail::DataLessProperty<
                               ::sycl::detail::UseDefaultStream> {};
} // namespace cuda::property::queue
} // namespace ext::oneapi

namespace property ::queue {
namespace __SYCL2020_DEPRECATED(
    "use 'sycl::ext::oneapi::cuda::property::queue' instead") cuda {
class use_default_stream
    : public ::sycl::ext::oneapi::cuda::property::queue::use_default_stream {};
// clang-format off
} // namespace cuda
// clang-format on
} // namespace property::queue

namespace ext {
namespace intel {
namespace property {
namespace queue {
class compute_index : public sycl::detail::PropertyWithData<
                          sycl::detail::PropWithDataKind::QueueComputeIndex> {
public:
  compute_index(int idx) : idx(idx) {}
  int get_index() { return idx; }

private:
  int idx;
};
} // namespace queue
} // namespace property
} // namespace intel
} // namespace ext

// Forward declaration
class queue;

// Queue property trait specializations
template <>
struct is_property_of<property::queue::in_order, queue> : std::true_type {};
template <>
struct is_property_of<property::queue::enable_profiling, queue>
    : std::true_type {};
template <>
struct is_property_of<ext::oneapi::property::queue::discard_events, queue>
    : std::true_type {};
template <>
struct is_property_of<ext::oneapi::property::queue::priority_normal, queue>
    : std::true_type {};
template <>
struct is_property_of<ext::oneapi::property::queue::priority_low, queue>
    : std::true_type {};
template <>
struct is_property_of<ext::oneapi::property::queue::priority_high, queue>
    : std::true_type {};
template <>
struct is_property_of<property::queue::cuda::use_default_stream, queue>
    : std::true_type {};
template <>
struct is_property_of<ext::oneapi::cuda::property::queue::use_default_stream,
                      queue> : std::true_type {};
template <>
struct is_property_of<ext::intel::property::queue::compute_index, queue>
    : std::true_type {};

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
