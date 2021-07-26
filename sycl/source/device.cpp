//==------------------- device.cpp -----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/device_filter.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <detail/backend_impl.hpp>
#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <detail/force_device.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
void force_type(info::device_type &t, const info::device_type &ft) {
  if (t == info::device_type::all) {
    t = ft;
  } else if (ft != info::device_type::all && t != ft) {
    throw cl::sycl::invalid_parameter_error("No device of forced type.",
                                            PI_INVALID_OPERATION);
  }
}
} // namespace detail

device::device() : impl(detail::device_impl::getHostDeviceImpl()) {}

device::device(cl_device_id DeviceId) {
  // The implementation constructor takes ownership of the native handle so we
  // must retain it in order to adhere to SYCL 1.2.1 spec (Rev6, section 4.3.1.)
  detail::RT::PiDevice Device;
  auto Plugin = detail::RT::getPlugin<backend::opencl>();
  Plugin.call<detail::PiApiKind::piextDeviceCreateWithNativeHandle>(
      detail::pi::cast<pi_native_handle>(DeviceId), nullptr, &Device);
  auto Platform =
      detail::platform_impl::getPlatformFromPiDevice(Device, Plugin);
  impl = Platform->getOrMakeDeviceImpl(Device, Platform);
  clRetainDevice(DeviceId);
}

device::device(const device_selector &deviceSelector) {
  *this = deviceSelector.select_device();
}

std::vector<device> device::get_devices(info::device_type deviceType) {
  std::vector<device> devices;
  detail::device_filter_list *FilterList =
      detail::SYCLConfig<detail::SYCL_DEVICE_FILTER>::get();
  // Host device availability should depend on the forced type
  bool includeHost = false;
  // If SYCL_DEVICE_FILTER is set, we don't automatically include it.
  // We will check if host devices are specified in the filter below.
  if (FilterList) {
    if (deviceType != info::device_type::host &&
        deviceType != info::device_type::all)
      includeHost = false;
    else
      includeHost = FilterList->containsHost();
  } else {
    includeHost = detail::match_types(deviceType, info::device_type::host);
  }
  info::device_type forced_type = detail::get_forced_type();
  // Exclude devices which do not match requested device type
  if (detail::match_types(deviceType, forced_type)) {
    detail::force_type(deviceType, forced_type);
    for (const auto &plt : platform::get_platforms()) {
      // If SYCL_BE is set then skip platforms which doesn't have specified
      // backend.
      backend *ForcedBackend = detail::SYCLConfig<detail::SYCL_BE>::get();
      if (ForcedBackend)
        if (!plt.is_host() && plt.get_backend() != *ForcedBackend)
          continue;
      // If SYCL_DEVICE_FILTER is set, skip platforms that is incompatible
      // with the filter specification.
      if (FilterList && !FilterList->backendCompatible(plt.get_backend()))
        continue;

      if (includeHost && plt.is_host()) {
        std::vector<device> host_device(
            plt.get_devices(info::device_type::host));
        if (!host_device.empty())
          devices.insert(devices.end(), host_device.begin(), host_device.end());
      } else {
        std::vector<device> found_devices(plt.get_devices(deviceType));
        if (!found_devices.empty())
          devices.insert(devices.end(), found_devices.begin(),
                         found_devices.end());
      }
    }
  }
  return devices;
}

cl_device_id device::get() const { return impl->get(); }

bool device::is_host() const { return impl->is_host(); }

bool device::is_cpu() const { return impl->is_cpu(); }

bool device::is_gpu() const { return impl->is_gpu(); }

bool device::is_accelerator() const { return impl->is_accelerator(); }

platform device::get_platform() const { return impl->get_platform(); }

template <info::partition_property prop>
std::vector<device> device::create_sub_devices(size_t ComputeUnits) const {
  return impl->create_sub_devices(ComputeUnits);
}

template __SYCL_EXPORT std::vector<device>
device::create_sub_devices<info::partition_property::partition_equally>(
    size_t ComputeUnits) const;

template <info::partition_property prop>
std::vector<device>
device::create_sub_devices(const std::vector<size_t> &Counts) const {
  return impl->create_sub_devices(Counts);
}

template __SYCL_EXPORT std::vector<device>
device::create_sub_devices<info::partition_property::partition_by_counts>(
    const std::vector<size_t> &Counts) const;

template <info::partition_property prop>
std::vector<device> device::create_sub_devices(
    info::partition_affinity_domain AffinityDomain) const {
  return impl->create_sub_devices(AffinityDomain);
}

template __SYCL_EXPORT std::vector<device> device::create_sub_devices<
    info::partition_property::partition_by_affinity_domain>(
    info::partition_affinity_domain AffinityDomain) const;

bool device::has_extension(const std::string &extension_name) const {
  return impl->has_extension(extension_name);
}

template <info::device param>
typename info::param_traits<info::device, param>::return_type
device::get_info() const {
  return impl->template get_info<param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(param_type, param, ret_type)                  \
  template __SYCL_EXPORT ret_type device::get_info<info::param_type::param>()  \
      const;

#include <CL/sycl/info/device_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

backend device::get_backend() const noexcept { return getImplBackend(impl); }

pi_native_handle device::getNative() const { return impl->getNative(); }

bool device::has(aspect Aspect) const { return impl->has(Aspect); }

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
