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
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>

namespace cl {
namespace sycl {

platform::platform() : impl(std::make_shared<detail::platform_impl>()) {}

platform::platform(cl_platform_id PlatformId)
    : impl(std::make_shared<detail::platform_impl>(
          detail::pi::cast<detail::RT::PiPlatform>(PlatformId))) {}

platform::platform(const device_selector &dev_selector) {
  *this = dev_selector.select_device().get_platform();
}

cl_platform_id platform::get() const { return impl->get(); }

bool platform::has_extension(const string_class &ExtensionName) const {
  return impl->has_extension(ExtensionName);
}

bool platform::is_host() const { return impl->is_host(); }

vector_class<device> platform::get_devices(info::device_type DeviceType) const {
  return impl->get_devices(DeviceType);
}

vector_class<platform> platform::get_platforms() {

  vector_class<platform> platforms = detail::platform_impl::get_platforms();

  // Add host device platform if required
  info::device_type forced_type = detail::get_forced_type();
  if (detail::match_types(forced_type, info::device_type::host))
    platforms.push_back(platform());

  return platforms;
}

template <info::platform param>
typename info::param_traits<info::platform, param>::return_type
platform::get_info() const {
  return impl->get_info<param>();
}

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
  template ret_type platform::get_info<info::param_type::param>() const;

#include <CL/sycl/info/platform_traits.def>

#undef PARAM_TRAITS_SPEC

} // namespace sycl
} // namespace cl
