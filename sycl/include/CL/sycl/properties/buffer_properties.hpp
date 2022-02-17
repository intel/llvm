//==----------- buffer_properties.hpp --- SYCL buffer properties -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/property_helper.hpp>
#include <CL/sycl/properties/property_traits.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

namespace property {
namespace buffer {
class use_host_ptr : public detail::DataLessProperty<detail::BufferUseHostPtr> {
};

class use_mutex : public detail::PropertyWithData<detail::BufferUseMutex> {
public:
  use_mutex(std::mutex &MutexRef) : MMutex(MutexRef) {}

  std::mutex *get_mutex_ptr() const { return &MMutex; }

private:
  std::mutex &MMutex;
};

class context_bound
    : public detail::PropertyWithData<detail::BufferContextBound> {
public:
  context_bound(sycl::context BoundContext) : MCtx(std::move(BoundContext)) {}

  sycl::context get_context() const { return MCtx; }

private:
  sycl::context MCtx;
};

class mem_channel : public detail::PropertyWithData<
                        detail::PropWithDataKind::BufferMemChannel> {
public:
  mem_channel(uint32_t Channel) : MChannel(Channel) {}
  uint32_t get_channel() const { return MChannel; }

private:
  uint32_t MChannel;
};

namespace detail {
class buffer_location
    : public sycl::detail::PropertyWithData<
          sycl::detail::PropWithDataKind::AccPropBufferLocation> {
public:
  buffer_location(uint64_t Location) : MLocation(Location) {}
  uint64_t get_buffer_location() const { return MLocation; }

private:
  uint64_t MLocation;
};
} // namespace detail
} // namespace buffer
} // namespace property

namespace ext {
namespace oneapi {
namespace property {
namespace buffer {

class use_pinned_host_memory : public sycl::detail::DataLessProperty<
                                   sycl::detail::BufferUsePinnedHostMemory> {};
} // namespace buffer
} // namespace property
} // namespace oneapi
} // namespace ext

// Forward declaration
template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;

// Buffer property trait specializations
template <>
struct is_property<property::buffer::use_host_ptr> : std::true_type {};
template <> struct is_property<property::buffer::use_mutex> : std::true_type {};
template <>
struct is_property<property::buffer::detail::buffer_location> : std::true_type {};
template <>
struct is_property<property::buffer::context_bound> : std::true_type {};
template <>
struct is_property<property::buffer::mem_channel> : std::true_type {};
template <>
struct is_property<ext::oneapi::property::buffer::use_pinned_host_memory>
    : std::true_type {};

template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<property::buffer::use_host_ptr,
                      buffer<T, Dimensions, AllocatorT, void>>
    : std::true_type {};
template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<property::buffer::use_mutex,
                      buffer<T, Dimensions, AllocatorT, void>>
    : std::true_type {};
template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<property::buffer::detail::buffer_location,
                      buffer<T, Dimensions, AllocatorT, void>>
    : std::true_type {};
template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<property::buffer::context_bound,
                      buffer<T, Dimensions, AllocatorT, void>>
    : std::true_type {};
template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<property::buffer::mem_channel,
                      buffer<T, Dimensions, AllocatorT, void>>
    : std::true_type {};
template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<ext::oneapi::property::buffer::use_pinned_host_memory,
                      buffer<T, Dimensions, AllocatorT, void>>
    : std::true_type {};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
