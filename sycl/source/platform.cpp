//==----------- platform.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/force_device.hpp>
#include <CL/sycl/detail/platform_impl.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/platform.hpp>

namespace cl {
namespace sycl {

platform::platform() : impl(std::make_shared<detail::platform_impl_host>()) {}

platform::platform(cl_platform_id platform_id)
    : impl(std::make_shared<detail::platform_impl_pi>(
             detail::pi::cast<detail::RT::PiPlatform>(platform_id))) {}

platform::platform(const device_selector &dev_selector) {
  *this = dev_selector.select_device().get_platform();
}

vector_class<device> platform::get_devices(info::device_type dev_type) const {
  return impl->get_devices(dev_type);
}

vector_class<platform> platform::get_platforms() {

  vector_class<platform> platforms =
    detail::platform_impl_pi::get_platforms();

  // Add host device platform if required
  info::device_type forced_type = detail::get_forced_type();
  if (detail::match_types(forced_type, info::device_type::host))
    platforms.push_back(platform());

  return platforms;
}

} // namespace sycl
} // namespace cl
