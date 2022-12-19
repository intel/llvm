#pragma once

#include <sycl/context.hpp>
#include <sycl/detail/property_helper.hpp>
#include <sycl/properties/property_traits.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
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

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
