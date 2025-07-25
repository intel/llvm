//==----------- platform.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/backend_impl.hpp>
#include <detail/config.hpp>
#include <detail/context_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/ur.hpp>
#include <sycl/context.hpp>
#include <sycl/device.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/image.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/platform.hpp>

namespace sycl {
inline namespace _V1 {

platform::platform() : platform(default_selector_v) {}

platform::platform(cl_platform_id PlatformId) {
  detail::adapter_impl &Adapter =
      sycl::detail::ur::getAdapter<backend::opencl>();
  ur_platform_handle_t UrPlatform = nullptr;
  Adapter.call<detail::UrApiKind::urPlatformCreateWithNativeHandle>(
      detail::ur::cast<ur_native_handle_t>(PlatformId), Adapter.getUrAdapter(),
      /* pProperties = */ nullptr, &UrPlatform);
  impl = detail::platform_impl::getOrMakePlatformImpl(UrPlatform, Adapter)
             .shared_from_this();
}

// protected constructor for internal use
platform::platform(const device &Device) { *this = Device.get_platform(); }

platform::platform(const device_selector &dev_selector) {
  *this = dev_selector.select_device().get_platform();
}

cl_platform_id platform::get() const { return impl->get(); }

bool platform::has_extension(detail::string_view ExtName) const {
  return impl->has_extension(std::string(std::string_view(ExtName)));
}

std::vector<device> platform::get_devices(info::device_type DeviceType) const {
  return impl->get_devices(DeviceType);
}

std::vector<platform> platform::get_platforms() {
  return detail::platform_impl::get_platforms();
}

backend platform::get_backend() const noexcept { return impl->getBackend(); }

template <typename Param>
detail::ABINeutralT_t<
    typename detail::is_platform_info_desc<Param>::return_type>
platform::get_info_impl() const {
  return detail::convert_to_abi_neutral(impl->template get_info<Param>());
}

ur_native_handle_t platform::getNative() const { return impl->getNative(); }

bool platform::has(aspect Aspect) const { return impl->has(Aspect); }

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template __SYCL_EXPORT detail::ABINeutralT_t<ReturnT>                        \
  platform::get_info_impl<info::platform::Desc>() const;

#include <sycl/info/platform_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

template <typename Param>
typename detail::is_backend_info_desc<Param>::return_type
platform::get_backend_info() const {
  return impl->get_backend_info<Param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, Picode)              \
  template __SYCL_EXPORT ReturnT                                               \
  platform::get_backend_info<info::DescType::Desc>() const;

#include <sycl/info/sycl_backend_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

context platform::khr_get_default_context() const {
  return detail::createSyclObjFromImpl<context>(
      impl->khr_get_default_context());
}

context platform::ext_oneapi_get_default_context() const {
  if (!detail::SYCLConfig<detail::SYCL_ENABLE_DEFAULT_CONTEXTS>::get())
    throw std::runtime_error("SYCL default contexts are not enabled");

  return khr_get_default_context();
}

std::vector<device> platform::ext_oneapi_get_composite_devices() const {
  // Only GPU architectures can be composite devices.
  auto GPUDevices = get_devices(info::device_type::gpu);
  // Using ZE_FLAT_DEVICE_HIERARCHY=COMBINED, we receive tiles as devices, which
  // are component devices. Thus, we need to get the composite device for each
  // of the component devices, and filter out duplicates.
  std::vector<device> Composites;
  std::vector<device> Result;
  for (auto &Dev : GPUDevices) {
    if (!Dev.has(sycl::aspect::ext_oneapi_is_component))
      continue;

    auto Composite = Dev.get_info<
        sycl::ext::oneapi::experimental::info::device::composite_device>();
    if (std::find(Composites.begin(), Composites.end(), Composite) ==
        Composites.end())
      Composites.push_back(Composite);
  }
  for (const auto &Composite : Composites) {
    auto Components = Composite.get_info<
        sycl::ext::oneapi::experimental::info::device::component_devices>();
    // Checking whether Components are GPU device is not enough, we need to
    // check if they are in the list of available devices returned by
    // `get_devices()`, because we cannot return a Composite device unless all
    // of its components are available too.
    size_t ComponentsFound = std::count_if(
        Components.begin(), Components.end(), [&](const device &d) {
          return std::find(GPUDevices.begin(), GPUDevices.end(), d) !=
                 GPUDevices.end();
        });
    if (ComponentsFound == Components.size())
      Result.push_back(Composite);
  }
  return Result;
}

namespace detail {

void enable_ext_oneapi_default_context(bool Val) {
  const char *StringVal = Val ? "1" : "0";
  detail::SYCLConfig<detail::SYCL_ENABLE_DEFAULT_CONTEXTS>::resetWithValue(
      StringVal);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
