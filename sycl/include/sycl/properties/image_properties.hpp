//==----------- image_properties.hpp --- SYCL image properties -------------==//
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
#include <type_traits> // for true_type
#include <utility>     // for move

namespace sycl {
inline namespace _V1 {
namespace property::image {
class use_host_ptr : public detail::DataLessProperty<detail::ImageUseHostPtr> {
};

class use_mutex : public detail::PropertyWithData<detail::ImageUseMutex> {
public:
  use_mutex(std::mutex &MutexRef) : MMutex(MutexRef) {}

  std::mutex *get_mutex_ptr() const { return &MMutex; }

private:
  std::mutex &MMutex;
};

class context_bound
    : public detail::PropertyWithData<detail::ImageContextBound> {
public:
  context_bound(sycl::context BoundContext) : MCtx(std::move(BoundContext)) {}

  sycl::context get_context() const { return MCtx; }

private:
  sycl::context MCtx;
};
} // namespace property::image

// Forward declaration
template <int Dimensions, typename AllocatorT> class image;
template <int Dimensions, typename AllocatorT> class sampled_image;
template <int Dimensions, typename AllocatorT> class unsampled_image;

// SYCL 1.2.1 image property trait specializations
template <int Dimensions, typename AllocatorT>
struct is_property_of<property::image::use_host_ptr,
                      image<Dimensions, AllocatorT>> : std::true_type {};
template <int Dimensions, typename AllocatorT>
struct is_property_of<property::image::use_mutex, image<Dimensions, AllocatorT>>
    : std::true_type {};
template <int Dimensions, typename AllocatorT>
struct is_property_of<property::image::context_bound,
                      image<Dimensions, AllocatorT>> : std::true_type {};

// SYCL 2020 image property trait specializations
template <int Dimensions, typename AllocatorT>
struct is_property_of<property::image::use_host_ptr,
                      sampled_image<Dimensions, AllocatorT>> : std::true_type {
};
template <int Dimensions, typename AllocatorT>
struct is_property_of<property::image::use_mutex,
                      sampled_image<Dimensions, AllocatorT>> : std::true_type {
};
template <int Dimensions, typename AllocatorT>
struct is_property_of<property::image::context_bound,
                      sampled_image<Dimensions, AllocatorT>> : std::true_type {
};
template <int Dimensions, typename AllocatorT>
struct is_property_of<property::image::use_host_ptr,
                      unsampled_image<Dimensions, AllocatorT>>
    : std::true_type {};
template <int Dimensions, typename AllocatorT>
struct is_property_of<property::image::use_mutex,
                      unsampled_image<Dimensions, AllocatorT>>
    : std::true_type {};
template <int Dimensions, typename AllocatorT>
struct is_property_of<property::image::context_bound,
                      unsampled_image<Dimensions, AllocatorT>>
    : std::true_type {};

} // namespace _V1
} // namespace sycl
