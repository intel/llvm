//==------------------- device.cpp -----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/device_impl.hpp>
#include <CL/sycl/detail/force_device.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>

namespace cl {
namespace sycl {
namespace detail {
void force_type(info::device_type &t, const info::device_type &ft) {
  if (t == info::device_type::all) {
    t = ft;
  } else if (ft != info::device_type::all && t != ft) {
    throw cl::sycl::invalid_parameter_error("No device of forced type.");
  }
}
} // namespace detail

device::device() : impl(std::make_shared<detail::device_host>()) {}

device::device(cl_device_id deviceId)
    : impl(std::make_shared<detail::device_impl_pi>(
      detail::pi::cast<detail::RT::PiDevice>(deviceId))) {}

device::device(const device_selector &deviceSelector) {
  *this = deviceSelector.select_device();
}

vector_class<device> device::get_devices(info::device_type deviceType) {
  vector_class<device> devices;
  info::device_type forced_type = detail::get_forced_type();
  // Exclude devices which do not match requested device type
  if (detail::match_types(deviceType, forced_type)) {
    detail::force_type(deviceType, forced_type);
    for (const auto &plt : platform::get_platforms()) {
      vector_class<device> found_devices(plt.get_devices(deviceType));
      if (!found_devices.empty())
        devices.insert(devices.end(), found_devices.begin(),
                       found_devices.end());
    }
  }
  return devices;
}

} // namespace sycl
} // namespace cl
