//==----------------- device_impl.cpp - SYCL device ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/platform_impl.hpp>
#include <sycl/device.hpp>

#include <algorithm>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

device_impl::device_impl()
    : MIsHostDevice(true), MPlatform(platform_impl::getHostPlatformImpl()),
      // assert is natively supported by host
      MIsAssertFailSupported(true) {}

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
    : MDevice(Device), MIsHostDevice(false),
      MDeviceHostBaseTime(std::make_pair(0, 0)) {

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

  // No need to set MRootDevice when MAlwaysRootDevice is true
  if ((Platform == nullptr) || !Platform->MAlwaysRootDevice) {
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        MDevice, PI_DEVICE_INFO_PARENT_DEVICE, sizeof(RT::PiDevice),
        &MRootDevice, nullptr);
  }

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

  MIsAssertFailSupported =
      has_extension(PI_DEVICE_INFO_EXTENSION_DEVICELIB_ASSERT);
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
        PI_ERROR_INVALID_DEVICE);
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

  std::string AllExtensionNames = get_device_info_string(
      this->getHandleRef(), PiInfoCode<info::device::extensions>::value,
      this->getPlugin());
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
  Plugin.call<sycl::errc::invalid, PiApiKind::piDevicePartition>(
      MDevice, Properties, SubDevicesCount, SubDevices.data(),
      &ReturnedSubDevices);
  if (ReturnedSubDevices != SubDevicesCount) {
    throw sycl::exception(
        errc::invalid,
        "Could not partition to the specified number of sub-devices");
  }
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
  assert(!MIsHostDevice && "Partitioning is not supported on host.");

  if (!is_partition_supported(info::partition_property::partition_equally)) {
    throw sycl::feature_not_supported(
        "Device does not support "
        "sycl::info::partition_property::partition_equally.",
        PI_ERROR_INVALID_OPERATION);
  }
  // If count exceeds the total number of compute units in the device, an
  // exception with the errc::invalid error code must be thrown.
  auto MaxComputeUnits = get_info<info::device::max_compute_units>();
  if (ComputeUnits > MaxComputeUnits)
    throw sycl::exception(errc::invalid,
                          "Total counts exceed max compute units");

  size_t SubDevicesCount = MaxComputeUnits / ComputeUnits;
  const pi_device_partition_property Properties[3] = {
      PI_DEVICE_PARTITION_EQUALLY, (pi_device_partition_property)ComputeUnits,
      0};
  return create_sub_devices(Properties, SubDevicesCount);
}

std::vector<device>
device_impl::create_sub_devices(const std::vector<size_t> &Counts) const {
  assert(!MIsHostDevice && "Partitioning is not supported on host.");

  if (!is_partition_supported(info::partition_property::partition_by_counts)) {
    throw sycl::feature_not_supported(
        "Device does not support "
        "sycl::info::partition_property::partition_by_counts.",
        PI_ERROR_INVALID_OPERATION);
  }
  static const pi_device_partition_property P[] = {
      PI_DEVICE_PARTITION_BY_COUNTS, PI_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0};
  std::vector<pi_device_partition_property> Properties(P, P + 3);

  // Fill the properties vector with counts and validate it
  auto It = Properties.begin() + 1;
  size_t TotalCounts = 0;
  size_t NonZeroCounts = 0;
  for (auto Count : Counts) {
    TotalCounts += Count;
    NonZeroCounts += (Count != 0) ? 1 : 0;
    It = Properties.insert(It, Count);
  }

  // If the number of non-zero values in counts exceeds the device’s maximum
  // number of sub devices (as returned by info::device::
  // partition_max_sub_devices) an exception with the errc::invalid
  // error code must be thrown.
  if (NonZeroCounts > get_info<info::device::partition_max_sub_devices>())
    throw sycl::exception(errc::invalid,
                          "Total non-zero counts exceed max sub-devices");

  // If the total of all the values in the counts vector exceeds the total
  // number of compute units in the device (as returned by
  // info::device::max_compute_units), an exception with the errc::invalid
  // error code must be thrown.
  if (TotalCounts > get_info<info::device::max_compute_units>())
    throw sycl::exception(errc::invalid,
                          "Total counts exceed max compute units");

  return create_sub_devices(Properties.data(), Counts.size());
}

std::vector<device> device_impl::create_sub_devices(
    info::partition_affinity_domain AffinityDomain) const {
  assert(!MIsHostDevice && "Partitioning is not supported on host.");

  if (!is_partition_supported(
          info::partition_property::partition_by_affinity_domain)) {
    throw sycl::feature_not_supported(
        "Device does not support "
        "sycl::info::partition_property::partition_by_affinity_domain.",
        PI_ERROR_INVALID_OPERATION);
  }
  if (!is_affinity_supported(AffinityDomain)) {
    throw sycl::feature_not_supported(
        "Device does not support " + affinityDomainToString(AffinityDomain) +
            ".",
        PI_ERROR_INVALID_VALUE);
  }
  const pi_device_partition_property Properties[3] = {
      PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
      (pi_device_partition_property)AffinityDomain, 0};

  pi_uint32 SubDevicesCount = 0;
  const detail::plugin &Plugin = getPlugin();
  Plugin.call<sycl::errc::invalid, PiApiKind::piDevicePartition>(
      MDevice, Properties, 0, nullptr, &SubDevicesCount);

  return create_sub_devices(Properties, SubDevicesCount);
}

std::vector<device> device_impl::create_sub_devices() const {
  assert(!MIsHostDevice && "Partitioning is not supported on host.");

  if (!is_partition_supported(
          info::partition_property::ext_intel_partition_by_cslice)) {
    throw sycl::feature_not_supported(
        "Device does not support "
        "sycl::info::partition_property::ext_intel_partition_by_cslice.",
        PI_ERROR_INVALID_OPERATION);
  }

  const pi_device_partition_property Properties[2] = {
      PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE, 0};

  pi_uint32 SubDevicesCount = 0;
  const detail::plugin &Plugin = getPlugin();
  Plugin.call<sycl::errc::invalid, PiApiKind::piDevicePartition>(
      MDevice, Properties, 0, nullptr, &SubDevicesCount);

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
  case aspect::ext_oneapi_bfloat16_math_functions:
    return get_info<info::device::ext_oneapi_bfloat16_math_functions>();
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
           (get_device_info_impl<
                pi_usm_capabilities,
                info::device::usm_host_allocations>::get(MDevice, getPlugin()) &
            PI_USM_CONCURRENT_ATOMIC_ACCESS);
  case aspect::usm_shared_allocations:
    return get_info<info::device::usm_shared_allocations>();
  case aspect::usm_atomic_shared_allocations:
    return is_host() ||
           (get_device_info_impl<
                pi_usm_capabilities,
                info::device::usm_shared_allocations>::get(MDevice,
                                                           getPlugin()) &
            PI_USM_CONCURRENT_ATOMIC_ACCESS);
  case aspect::usm_restricted_shared_allocations:
    return get_info<info::device::usm_restricted_shared_allocations>();
  case aspect::usm_system_allocations:
    return get_info<info::device::usm_system_allocations>();
  case aspect::ext_intel_device_id:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_DEVICE_ID, 0, nullptr, &return_size) ==
           PI_SUCCESS;
  case aspect::ext_intel_pci_address:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_PCI_ADDRESS, 0, nullptr, &return_size) ==
           PI_SUCCESS;
  case aspect::ext_intel_gpu_eu_count:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_GPU_EU_COUNT, 0, nullptr,
               &return_size) == PI_SUCCESS;
  case aspect::ext_intel_gpu_eu_simd_width:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH, 0, nullptr,
               &return_size) == PI_SUCCESS;
  case aspect::ext_intel_gpu_slices:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_GPU_SLICES, 0, nullptr, &return_size) ==
           PI_SUCCESS;
  case aspect::ext_intel_gpu_subslices_per_slice:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE, 0, nullptr,
               &return_size) == PI_SUCCESS;
  case aspect::ext_intel_gpu_eu_count_per_subslice:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE, 0, nullptr,
               &return_size) == PI_SUCCESS;
  case aspect::ext_intel_gpu_hw_threads_per_eu:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_DEVICE_INFO_GPU_HW_THREADS_PER_EU, 0, nullptr,
               &return_size) == PI_SUCCESS;
  case aspect::ext_intel_free_memory:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_EXT_INTEL_DEVICE_INFO_FREE_MEMORY, 0, nullptr,
               &return_size) == PI_SUCCESS;
  case aspect::ext_intel_memory_clock_rate:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_EXT_INTEL_DEVICE_INFO_MEMORY_CLOCK_RATE, 0, nullptr,
               &return_size) == PI_SUCCESS;
  case aspect::ext_intel_memory_bus_width:
    return getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
               MDevice, PI_EXT_INTEL_DEVICE_INFO_MEMORY_BUS_WIDTH, 0, nullptr,
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
  case aspect::ext_oneapi_native_assert:
    return isAssertFailSupported();
  case aspect::ext_oneapi_cuda_async_barrier: {
    int async_barrier_supported;
    bool call_successful =
        getPlugin().call_nocheck<detail::PiApiKind::piDeviceGetInfo>(
            MDevice, PI_EXT_ONEAPI_DEVICE_INFO_CUDA_ASYNC_BARRIER, sizeof(int),
            &async_barrier_supported, nullptr) == PI_SUCCESS;
    return call_successful && async_barrier_supported;
  }
  default:
    throw runtime_error("This device aspect has not been implemented yet.",
                        PI_ERROR_INVALID_DEVICE);
  }
}

std::shared_ptr<device_impl> device_impl::getHostDeviceImpl() {
  static std::shared_ptr<device_impl> HostImpl =
      std::make_shared<device_impl>();

  return HostImpl;
}

bool device_impl::isAssertFailSupported() const {
  return MIsAssertFailSupported;
}

std::string device_impl::getDeviceName() const {
  std::call_once(MDeviceNameFlag,
                 [this]() { MDeviceName = get_info<info::device::name>(); });

  return MDeviceName;
}

/* On first call this function queries for device timestamp
   along with host synchronized timestamp
   and stores it in memeber varaible deviceTimePair.
   Subsequent calls to this function would just retrieve the host timestamp ,
   compute difference against the host timestamp in deviceTimePair
   and calculate the device timestamp based on the difference.
   deviceTimePair is refreshed with new device and host timestamp after a
   certain interval (determined by timeTillRefresh) to account for clock drift
   between host and device.
*/

uint64_t device_impl::getCurrentDeviceTime() {
  // To account for potential clock drift between host clock and device clock.
  // The value set is arbitrary: 200 seconds
  constexpr uint64_t timeTillRefresh = 200e9;

  uint64_t hostTime;
  if (MIsHostDevice) {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
        .count();
  }
  auto plugin = getPlugin();
  RT::PiResult result =
      plugin.call_nocheck<detail::PiApiKind::piGetDeviceAndHostTimer>(
          MDevice, nullptr, &hostTime);
  plugin.checkPiResult(result == PI_ERROR_INVALID_OPERATION ? PI_SUCCESS
                                                            : result);

  if (result == PI_ERROR_INVALID_OPERATION) {
    std::string errorMsg{};
    char *p;
    plugin.call_nocheck<detail::PiApiKind::piPluginGetLastError>(&p);
    while (*p != '\0') {
      errorMsg += *p;
      p++;
    }
    throw sycl::feature_not_supported(
        "Device and/or backend does not support querying timestamp: " +
            errorMsg,
        result);
  }
  uint64_t diff = hostTime - MDeviceHostBaseTime.second;

  if (diff > timeTillRefresh || diff <= 0) {
    plugin.call<detail::PiApiKind::piGetDeviceAndHostTimer>(
        MDevice, &MDeviceHostBaseTime.first, &MDeviceHostBaseTime.second);
    diff = 0;
  }

  return MDeviceHostBaseTime.first + diff;
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
