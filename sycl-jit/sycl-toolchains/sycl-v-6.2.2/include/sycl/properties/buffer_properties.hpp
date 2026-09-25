//==----------- buffer_properties.hpp --- SYCL buffer properties -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/context.hpp>                    // for context
#include <sycl/detail/property_helper.hpp>     // for PropWithDataKind, Dat...
#include <sycl/properties/property_traits.hpp> // for is_property_of

#include <mutex>       // for mutex
#include <stdint.h>    // for uint32_t, uint64_t
#include <type_traits> // for true_type
#include <utility>     // for move

namespace sycl {
inline namespace _V1 {
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  namespace NS_QUALIFIER {                                                     \
  class PROP_NAME                                                              \
      : public sycl::detail::DataLessProperty<sycl::detail::ENUM_VAL> {};      \
  }
#include <sycl/properties/buffer_properties.def>

namespace property::buffer {
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
} // namespace property::buffer

// Forward declaration
template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;

#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)                  \
  template <typename T, int Dimensions, typename AllocatorT>                   \
  struct is_property_of<NS_QUALIFIER::PROP_NAME,                               \
                        buffer<T, Dimensions, AllocatorT, void>>               \
      : std::true_type {};
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)

#include <sycl/properties/buffer_properties.def>

} // namespace _V1
} // namespace sycl
