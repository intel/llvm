//==----------- platform.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/backend_impl.hpp>
#include <detail/config.hpp>
#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>
#include <sycl/device.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/platform.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

platform::platform() : platform(default_selector_v) {}

platform::platform(cl_platform_id PlatformId) {
  impl = detail::platform_impl::getOrMakePlatformImpl(
      detail::pi::cast<detail::RT::PiPlatform>(PlatformId),
      detail::RT::getPlugin<backend::opencl>());
}

// protected constructor for internal use
platform::platform(const device &Device) { *this = Device.get_platform(); }

platform::platform(const device_selector &dev_selector) {
  *this = dev_selector.select_device().get_platform();
}

cl_platform_id platform::get() const { return impl->get(); }

bool platform::has_extension(const std::string &ExtensionName) const {
  return impl->has_extension(ExtensionName);
}

bool platform::is_host() const {
  bool IsHost = impl->is_host();
  assert(!IsHost &&
         "platform::is_host should not be called in implementation.");
  return IsHost;
}

std::vector<device> platform::get_devices(info::device_type DeviceType) const {
  return impl->get_devices(DeviceType);
}

std::vector<platform> platform::get_platforms() {
  return detail::platform_impl::get_platforms();
}

backend platform::get_backend() const noexcept { return getImplBackend(impl); }

template <typename Param>
typename detail::is_platform_info_desc<Param>::return_type
platform::get_info() const {
  return impl->get_info<Param>();
}

pi_native_handle platform::getNative() const { return impl->getNative(); }

bool platform::has(aspect Aspect) const { return impl->has(Aspect); }

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template __SYCL_EXPORT ReturnT platform::get_info<info::platform::Desc>()    \
      const;

#include <sycl/info/platform_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

context platform::ext_oneapi_get_default_context() const {
  if (!detail::SYCLConfig<detail::SYCL_ENABLE_DEFAULT_CONTEXTS>::get())
    throw std::runtime_error("SYCL default contexts are not enabled");

  // Keeping the default context for platforms in the global cache to avoid
  // shared_ptr based circular dependency between platform and context classes
  std::unordered_map<detail::PlatformImplPtr, detail::ContextImplPtr>
      &PlatformToDefaultContextCache =
          detail::GlobalHandler::instance().getPlatformToDefaultContextCache();

  std::lock_guard<std::mutex> Lock{
      detail::GlobalHandler::instance()
          .getPlatformToDefaultContextCacheMutex()};

  auto It = PlatformToDefaultContextCache.find(impl);
  if (PlatformToDefaultContextCache.end() == It)
    std::tie(It, std::ignore) = PlatformToDefaultContextCache.insert(
        {impl, detail::getSyclObjImpl(context{get_devices()})});

  return detail::createSyclObjFromImpl<context>(It->second);
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
