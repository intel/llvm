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
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  namespace NS_QUALIFIER {                                                     \
  class PROP_NAME                                                              \
      : public sycl::detail::DataLessProperty<sycl::detail::ENUM_VAL> {};      \
  }
#include <sycl/properties/image_properties.def>

namespace property::image {
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
#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)                  \
  template <int Dimensions, typename AllocatorT>                               \
  struct is_property_of<NS_QUALIFIER::PROP_NAME,                               \
                        image<Dimensions, AllocatorT>> : std::true_type {};
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)
#include <sycl/properties/image_properties.def>

// SYCL 2020 image property trait specializations
#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)                  \
  template <int Dimensions, typename AllocatorT>                               \
  struct is_property_of<NS_QUALIFIER::PROP_NAME,                               \
                        sampled_image<Dimensions, AllocatorT>>                 \
      : std::true_type {};
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)
#include <sycl/properties/image_properties.def>

#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)                  \
  template <int Dimensions, typename AllocatorT>                               \
  struct is_property_of<NS_QUALIFIER::PROP_NAME,                               \
                        unsampled_image<Dimensions, AllocatorT>>               \
      : std::true_type {};
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)
#include <sycl/properties/image_properties.def>

} // namespace _V1
} // namespace sycl
