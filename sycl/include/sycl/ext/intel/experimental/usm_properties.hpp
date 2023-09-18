//===- usm_properties.hpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/property_helper.hpp>     // for PropWithDataKind, Dat...
#include <sycl/properties/property_traits.hpp> // for is_property

#include <stdint.h>    // for uint64_t
#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
namespace ext {

namespace oneapi::property::usm {
class device_read_only
    : public sycl::detail::DataLessProperty<sycl::detail::DeviceReadOnly> {
public:
  device_read_only() = default;
};
} // namespace oneapi::property::usm

namespace intel::experimental::property::usm {

class buffer_location
    : public sycl::detail::PropertyWithData<
          sycl::detail::PropWithDataKind::AccPropBufferLocation> {
public:
  buffer_location(uint64_t Location) : MLocation(Location) {}
  uint64_t get_buffer_location() const { return MLocation; }

private:
  uint64_t MLocation;
};

} // namespace intel::experimental::property::usm
} // namespace ext

template <>
struct is_property<ext::oneapi::property::usm::device_read_only>
    : std::true_type {};

template <>
struct is_property<ext::intel::experimental::property::usm::buffer_location>
    : std::true_type {};

} // namespace _V1
} // namespace sycl
