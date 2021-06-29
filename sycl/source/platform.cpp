//==----------- platform.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>
#include <detail/backend_impl.hpp>
#include <detail/force_device.hpp>
#include <detail/platform_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

platform::platform() : impl(detail::platform_impl::getHostPlatformImpl()) {}

platform::platform(cl_platform_id PlatformId) {
  impl = detail::platform_impl::getOrMakePlatformImpl(
      detail::pi::cast<detail::RT::PiPlatform>(PlatformId),
      detail::RT::getPlugin<backend::opencl>());
}

platform::platform(const device_selector &dev_selector) {
  *this = dev_selector.select_device().get_platform();
}

cl_platform_id platform::get() const { return impl->get(); }

bool platform::has_extension(const std::string &ExtensionName) const {
  return impl->has_extension(ExtensionName);
}

bool platform::is_host() const { return impl->is_host(); }

std::vector<device> platform::get_devices(info::device_type DeviceType) const {
  return impl->get_devices(DeviceType);
}

std::vector<platform> platform::get_platforms() {
  return detail::platform_impl::get_platforms();
}

backend platform::get_backend() const noexcept { return getImplBackend(impl); }

template <info::platform param>
typename info::param_traits<info::platform, param>::return_type
platform::get_info() const {
  return impl->get_info<param>();
}

pi_native_handle platform::getNative() const { return impl->getNative(); }

bool platform::has(aspect Aspect) const { return impl->has(Aspect); }

#define __SYCL_PARAM_TRAITS_SPEC(param_type, param, ret_type)                  \
  template __SYCL_EXPORT ret_type                                              \
  platform::get_info<info::param_type::param>() const;

#include <CL/sycl/info/platform_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
