//==----------- queue_properties.hpp --- SYCL queue properties -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>  // for __SYCL2020_DEPRECATED
#include <sycl/detail/property_helper.hpp>     // for DataLessPropKind
#include <sycl/properties/property_traits.hpp> // for is_property_of

#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  namespace NS_QUALIFIER {                                                     \
  class PROP_NAME                                                              \
      : public sycl::detail::DataLessProperty<sycl::detail::ENUM_VAL> {};      \
  }

#include <sycl/properties/queue_properties.def>

namespace property ::queue {
namespace __SYCL2020_DEPRECATED(
    "use 'sycl::ext::oneapi::cuda::property::queue' instead") cuda {
class use_default_stream
    : public ::sycl::ext::oneapi::cuda::property::queue::use_default_stream {};
// clang-format off
} // namespace cuda
// clang-format on
} // namespace property::queue

namespace ext::intel::property::queue {
class compute_index : public sycl::detail::PropertyWithData<
                          sycl::detail::PropWithDataKind::QueueComputeIndex> {
public:
  compute_index(int idx) : idx(idx) {}
  int get_index() { return idx; }

private:
  int idx;
};
} // namespace ext::intel::property::queue

// Queue property trait specializations.
class queue;

#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)                  \
  template <>                                                                  \
  struct is_property_of<NS_QUALIFIER::PROP_NAME, queue> : std::true_type {};
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)

#include <sycl/properties/queue_properties.def>

} // namespace _V1
} // namespace sycl
