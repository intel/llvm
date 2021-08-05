//==----------------- device_impl.cpp - SYCL device ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/device.hpp>
#include <detail/device_impl.hpp>
#include <detail/platform_impl.hpp>

#include <algorithm>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

device_impl::device_impl()
    : MIsHostDevice(true), MPlatform(platform_impl::getHostPlatformImpl()) {}

device_impl::device_impl(pi_native_handle InteropDeviceHandle,
                         const plugin &Plugin)
    : device_impl(InteropDeviceHandle, nullptr, nullptr, Plugin) {}

device_impl::device_impl(RT::PiDevice Device, PlatformImplPtr Platform)
    : device_impl(reinterpret_cast<pi_native_handle>(nullptr), Device, Platform,
                  Platform->getPlugin()) {}

device_impl::device_impl(RT::PiDevice Device, const plugin &Plugin)
    : device_impl(reinterpret_cast<pi_native_handle>(nullptr), Device, nullptr,
                  Plugin) {}

device_impl::device_impl(pi_native_handle InteropDeviceHandle,
                         RT::PiDevice Device, PlatformImplPtr Platform,
                         const plugin &Plugin)
    : MDevice(Device), MIsHostDevice(false) {

  bool InteroperabilityConstructor = false;
  if (Device == nullptr) {
    assert(InteropDeviceHandle);
    // Get PI device from the raw device handle.
    // NOTE: this is for OpenCL interop only (and should go away).
    // With SYCL-2020 BE generalization "make" functions are used instead.
    Plugin.call<PiApiKind::piextDeviceCreateWithNativeHandle>(
        InteropDeviceHandle, nullptr, &MDevice);
    InteroperabilityConstructor = true;
  }

  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piDeviceGetInfo>(
      MDevice, PI_DEVICE_INFO_TYPE, sizeof(RT::PiDeviceType), &MType, nullptr);

  RT::PiDevice parent = nullptr;
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piDeviceGetInfo>(MDevice, PI_DEVICE_INFO_PARENT_DEVICE,
                                          sizeof(RT::PiDevice), &parent,
                                          nullptr);

  MIsRootDevice = (nullptr == parent);
  if (!InteroperabilityConstructor) {
    // TODO catch an exception and put it to list of asynchronous exceptions
    // Interoperability Constructor already calls DeviceRetain in
    // piextDeviceFromNative.
    Plugin.call<PiApiKind::piDeviceRetain>(MDevice);
  }

  // set MPlatform
  if (!Platform) {
    Platform = platform_impl::getPlatformFromPiDevice(MDevice, Plugin);
  }
  MPlatform = Platform;
}

device_impl::~device_impl() {
  if (!MIsHostDevice) {
    // TODO catch an exception and put it to list of asynchronous exceptions
    const detail::plugin &Plugin = getPlugin();
    RT::PiResult Err = Plugin.call_nocheck<PiApiKind::piDeviceRelease>(MDevice);
    __SYCL_CHECK_OCL_CODE_NO_EXC(Err);
  }
}

bool device_impl::is_affinity_supported(
    info::partition_affinity_domain AffinityDomain) const {
  auto SupportedDomains = get_info<info::device::partition_affinity_domains>();
  return std::find(SupportedDomains.begin(), SupportedDomains.end(),
                   AffinityDomain) != SupportedDomains.end();
}

cl_device_id device_impl::get() const {
  if (MIsHostDevice) {
    throw invalid_object_error(
        "This instance of device doesn't support OpenCL interoperability.",
        PI_INVALID_DEVICE);
  }
  // TODO catch an exception and put it to list of asynchronous exceptions
  getPlugin().call<PiApiKind::piDeviceRetain>(MDevice);
  return pi::cast<cl_device_id>(getNative());
}

platform device_impl::get_platform() const {
  return createSyclObjFromImpl<platform>(MPlatform);
}

bool device_impl::has_extension(const std::string &ExtensionName) const {
  if (MIsHostDevice)
    // TODO: implement extension management for host device;
    return false;

  std::string AllExtensionNames =
      get_device_info<std::string, info::device::extensions>::get(
          this->getHandleRef(), this->getPlugin());
  return (AllExtensionNames.find(ExtensionName) != std::string::npos);
}

bool device_impl::is_partition_supported(info::partition_property Prop) const {
  auto SupportedProperties = get_info<info::device::partition_properties>();
  return std::find(SupportedProperties.begin(), SupportedProperties.end(),
                   Prop) != SupportedProperties.end();
}

std::vector<device>
device_impl::create_sub_devices(const cl_device_partition_property *Properties,
                                size_t SubDevicesCount) const {

  std::vector<RT::PiDevice> SubDevices(SubDevicesCount);
  pi_uint32 ReturnedSubDevices = 0;
  const detail::plugin &Plugin = getPlugin();
  Plugin.call<PiApiKind::piDevicePartition>(MDevice, Properties,
                                            SubDevicesCount, SubDevices.data(),
                                            &ReturnedSubDevices);
  // TODO: check that returned number of sub-devices matches what was
  // requested, otherwise this walk below is wrong.
  //
  // TODO: Need to describe the subdevice model. Some sub_device management
  // may be necessary. What happens if create_sub_devices is called multiple
  // times with the same arguments?
  //
  std::vector<device> res;
  std::for_each(SubDevices.begin(), SubDevices.end(),
                [&res, this](const RT::PiDevice &a_pi_device) {
                  device sycl_device = detail::createSyclObjFromImpl<device>(
                      MPlatform->getOrMakeDeviceImpl(a_pi_device, MPlatform));
                  res.push_back(sycl_device);
                });
  return res;
}

std::vector<device> device_impl::create_sub_devices(size_t ComputeUnits) const {

  if (MIsHostDevice)
    // TODO: implement host device partitioning
    throw runtime_error(
        "Partitioning to subdevices of the host device is not implemented yet",
        PI_INVALID_DEVICE);

  if (!is_partition_supported(info::partition_property::partition_equally)) {
    throw cl::sycl::feature_not_supported();
  }
  size_t SubDevicesCount =
      get_info<info::device::max_compute_units>() / ComputeUnits;
  const cl_device_partition_property Properties[3] = {
      CL_DEVICE_PARTITION_EQUALLY, (cl_device_partition_property)ComputeUnits,
      0};
  return create_sub_devices(Properties, SubDevicesCount);
}

std::vector<device>
device_impl::create_sub_devices(const std::vector<size_t> &Counts) const {

  if (MIsHostDevice)
    // TODO: implement host device partitioning
    throw runtime_error(
        "Partitioning to subdevices of the host device is not implemented yet",
        PI_INVALID_DEVICE);

  if (!is_partition_supported(info::partition_property::partition_by_counts)) {
    throw cl::sycl::feature_not_supported();
  }
  static const cl_device_partition_property P[] = {
      CL_DEVICE_PARTITION_BY_COUNTS, CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0};
  std::vector<cl_device_partition_property> Properties(P, P + 3);
  Properties.insert(Properties.begin() + 1, Counts.begin(), Counts.end());
  return create_sub_devices(Properties.data(), Counts.size());
}

std::vector<device> device_impl::create_sub_devices(
    info::partition_affinity_domain AffinityDomain) const {

  if (MIsHostDevice)
    // TODO: implement host device partitioning
    throw runtime_error(
        "Partitioning to subdevices of the host device is not implemented yet",
        PI_INVALID_DEVICE);

  if (!is_partition_supported(
          info::partition_property::partition_by_affinity_domain) ||
      !is_affinity_supported(AffinityDomain)) {
    throw cl::sycl::feature_not_supported();
  }
  const pi_device_partition_property Properties[3] = {
      PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
      (pi_device_partition_property)AffinityDomain, 0};
  size_t SubDevicesCount = get_info<info::device::partition_max_sub_devices>();
  return create_sub_devices(Properties, SubDevicesCount);
}

pi_native_handle device_impl::getNative() const {
  auto Plugin = getPlugin();
  if (Plugin.getBackend() == backend::opencl)
    Plugin.call<PiApiKind::piDeviceRetain>(getHandleRef());
  pi_native_handle Handle;
  Plugin.call<PiApiKind::piextDeviceGetNativeHandle>(getHandleRef(), &Handle);
  return Handle;
}

bool device_impl::has(aspect Aspect) const {
  size_t return_size = 0;
  pi_device_type device_type;

  switch (Aspect) {
  case aspect::host:
    return is_host();
  case aspect::cpu:
    return is_cpu();
  case aspect::gpu:
    return is_gpu();
  case aspect::accelerator:
    return is_accelerator();
  case aspect::custom:
    return false;
  case aspect::fp16:
    return has_extension("cl_khr_fp16");
  case aspect::fp64:
    return has_extension("cl_khr_fp64");
  case aspect::int64_base_atomics:
    return has_extension("cl_khr_int64_base_atomics");
  case aspect::int64_extended_atomics:
    return has_extension("cl_khr_int64_extended_atomics");
  case aspect::atomic64:
    return get_info<info::device::atomic64>();
  case aspect::image:
    return get_info<info::device::image_support>();
  case aspect::online_compiler:
    return get_info<info::device::is_compiler_available>();
  case aspect::online_linker:
    return get_info<info::device::is_linker_available>();
  case aspect::queue_profiling:
    return get_info<info::device::queue_profiling>();
  case aspect::usm_device_allocations:
    return get_info<info::device::usm_device_allocations>();
  case aspect::usm_host_allocations:
    return get_info<info::device::usm_host_allocations>();
  case aspect::usm_atomic_host_allocations:
    return is_host() ||
           (get_device_info<
                pi_usm_capabilities,
                info::device::usm_host_allocations>::get(MDevice, getPlugin()) &
            PI_USM_ATOMIC_ACCESS);
  case aspect::usm_shared_allocations:
    return get_info<info::device::usm_shared_allocations>();
  case aspect::usm_atomic_shared_allocations:
    return is_host() ||
           (get_device_info<
                pi_usm_capabilities,
                info::device::usm_shared_allocations>::get(MDevice,
                                                           getPlugin()) &
            PI_USM_ATOMIC_ACCESS);
  case aspect::usm_restricted_shared_allocations:
    return get_info<info::device::usm_restricted_shared_allocations>();
  case aspect::usm_system_allocations:
    return get_info<info::device::usm_system_allocations>();
  case aspect::ext_intel_pci_address:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_PCI_ADDRESS, sizeof(pi_device_type),
               &device_type, &return_size) == PI_SUCCESS;
  case aspect::ext_intel_gpu_eu_count:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_GPU_EU_COUNT, sizeof(pi_device_type),
               &device_type, &return_size) == PI_SUCCESS;
  case aspect::ext_intel_gpu_eu_simd_width:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH,
               sizeof(pi_device_type), &device_type,
               &return_size) == PI_SUCCESS;
  case aspect::ext_intel_gpu_slices:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_GPU_SLICES, sizeof(pi_device_type),
               &device_type, &return_size) == PI_SUCCESS;
  case aspect::ext_intel_gpu_subslices_per_slice:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE,
               sizeof(pi_device_type), &device_type,
               &return_size) == PI_SUCCESS;
  case aspect::ext_intel_gpu_eu_count_per_subslice:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE,
               sizeof(pi_device_type), &device_type,
               &return_size) == PI_SUCCESS;
  case aspect::ext_intel_device_info_uuid: {
    auto Result = getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
        MDevice, PI_DEVICE_INFO_UUID, 0, nullptr, &return_size);
    if (Result != PI_SUCCESS) {
      return false;
    }

    assert(return_size <= 16);
    unsigned char UUID[16];

    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_UUID, 16 * sizeof(unsigned char), UUID,
               nullptr) == PI_SUCCESS;
  }
  case aspect::ext_intel_max_mem_bandwidth:
    // currently not supported
    return false;
  case aspect::ext_oneapi_srgb:
    return get_info<info::device::ext_oneapi_srgb>();

  default:
    throw runtime_error("This device aspect has not been implemented yet.",
                        PI_INVALID_DEVICE);
  }
}

std::shared_ptr<device_impl> device_impl::getHostDeviceImpl() {
  static std::shared_ptr<device_impl> HostImpl =
      std::make_shared<device_impl>();

  return HostImpl;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
