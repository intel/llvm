//==------------------- device.cpp -----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/backend_impl.hpp>
#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <sycl/detail/device_filter.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/device.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/info/info_desc.hpp>

#include <algorithm>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
void force_type(info::device_type &t, const info::device_type &ft) {
  if (t == info::device_type::all) {
    t = ft;
  } else if (ft != info::device_type::all && t != ft) {
    throw sycl::invalid_parameter_error("No device of forced type.",
                                        PI_ERROR_INVALID_OPERATION);
  }
}
} // namespace detail

device::device() : device(default_selector_v) {}

device::device(cl_device_id DeviceId) {
  // The implementation constructor takes ownership of the native handle so we
  // must retain it in order to adhere to SYCL 1.2.1 spec (Rev6, section 4.3.1.)
  sycl::detail::pi::PiDevice Device;
  auto Plugin = sycl::detail::pi::getPlugin<backend::opencl>();
  Plugin->call<detail::PiApiKind::piextDeviceCreateWithNativeHandle>(
      detail::pi::cast<pi_native_handle>(DeviceId), nullptr, &Device);
  auto Platform =
      detail::platform_impl::getPlatformFromPiDevice(Device, Plugin);
  impl = Platform->getOrMakeDeviceImpl(Device, Platform);
  Plugin->call<detail::PiApiKind::piDeviceRetain>(impl->getHandleRef());
}

device::device(const device_selector &deviceSelector) {
  *this = deviceSelector.select_device();
}

std::vector<device> device::get_devices(info::device_type deviceType) {
  std::vector<device> devices;
  detail::device_filter_list *FilterList =
      detail::SYCLConfig<detail::SYCL_DEVICE_FILTER>::get();
  detail::ods_target_list *OdsTargetList =
      detail::SYCLConfig<detail::ONEAPI_DEVICE_SELECTOR>::get();

  auto thePlatforms = platform::get_platforms();
  for (const auto &plt : thePlatforms) {
    // If SYCL_DEVICE_FILTER is set, skip platforms that is incompatible
    // with the filter specification.
    backend platformBackend = plt.get_backend();
    if (FilterList && !FilterList->backendCompatible(platformBackend))
      continue;
    if (OdsTargetList && !OdsTargetList->backendCompatible(platformBackend))
      continue;

    std::vector<device> found_devices(plt.get_devices(deviceType));
    if (!found_devices.empty())
      devices.insert(devices.end(), found_devices.begin(), found_devices.end());
  }

  return devices;
}

cl_device_id device::get() const { return impl->get(); }

bool device::is_host() const {
  bool IsHost = impl->is_host();
  assert(!IsHost && "device::is_host should not be called in implementation.");
  return IsHost;
}

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

template <info::partition_property prop>
std::vector<device> device::create_sub_devices() const {
  return impl->create_sub_devices();
}

template __SYCL_EXPORT std::vector<device> device::create_sub_devices<
    info::partition_property::ext_intel_partition_by_cslice>() const;

bool device::has_extension(const std::string &extension_name) const {
  return impl->has_extension(extension_name);
}

template <typename Param>
typename detail::is_device_info_desc<Param>::return_type
device::get_info() const {
  return impl->template get_info<Param>();
}

// Explicit override. Not fulfilled by #include device_traits.def below.
template <>
__SYCL_EXPORT device device::get_info<info::device::parent_device>() const {
  // With ONEAPI_DEVICE_SELECTOR the impl.MRootDevice is preset and may be
  // overridden (ie it may be nullptr on a sub-device) The PI of the sub-devices
  // have parents, but we don't want to return them. They must pretend to be
  // parentless root devices.
  if (impl->isRootDevice())
    throw invalid_object_error(
        "No parent for device because it is not a subdevice",
        PI_ERROR_INVALID_DEVICE);
  else
    return impl->template get_info<info::device::parent_device>();
}

template <>
__SYCL_EXPORT std::vector<sycl::aspect>
device::get_info<info::device::aspects>() const {
  std::vector<sycl::aspect> DeviceAspects{
#define __SYCL_ASPECT(ASPECT, ID) aspect::ASPECT,
#include <sycl/info/aspects.def>
#undef __SYCL_ASPECT
  };

  auto UnsupportedAspects = std::remove_if(
      DeviceAspects.begin(), DeviceAspects.end(), [&](aspect Aspect) {
        try {
          return !impl->has(Aspect);
        } catch (const runtime_error &ex) {
          if (ex.get_cl_code() == PI_ERROR_INVALID_DEVICE)
            return true;
          throw;
        }
      });

  DeviceAspects.erase(UnsupportedAspects, DeviceAspects.end());

  return DeviceAspects;
}

template <>
__SYCL_EXPORT bool device::get_info<info::device::image_support>() const {
  // Explicit specialization is needed due to the class of info handle. The
  // implementation is done in get_device_info_impl.
  return impl->template get_info<info::device::image_support>();
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template __SYCL_EXPORT ReturnT device::get_info<info::device::Desc>() const;

#define __SYCL_PARAM_TRAITS_SPEC_SPECIALIZED(DescType, Desc, ReturnT, PiCode)

#include <sycl/info/device_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC_SPECIALIZED
#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(Namespace, DescType, Desc, ReturnT, PiCode)   \
  template __SYCL_EXPORT ReturnT                                               \
  device::get_info<Namespace::info::DescType::Desc>() const;

#include <sycl/info/ext_codeplay_device_traits.def>
#include <sycl/info/ext_intel_device_traits.def>
#include <sycl/info/ext_oneapi_device_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

backend device::get_backend() const noexcept { return impl->getBackend(); }

pi_native_handle device::getNative() const { return impl->getNative(); }

bool device::has(aspect Aspect) const { return impl->has(Aspect); }

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
