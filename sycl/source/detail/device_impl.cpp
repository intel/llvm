//==----------------- device_impl.cpp - SYCL device ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/jit_compiler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/ur_info_code.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/device.hpp>

#include <algorithm>

namespace sycl {
inline namespace _V1 {
namespace detail {

ext::oneapi::experimental::architecture device_impl::get_architecture() const {
  using oneapi_exp_arch = sycl::ext::oneapi::experimental::architecture;

  // Only for NVIDIA and AMD GPU architectures
  constexpr std::pair<const char *, oneapi_exp_arch>
      NvidiaAmdGPUArchitectures[] = {
          {"5.0", oneapi_exp_arch::nvidia_gpu_sm_50},
          {"5.2", oneapi_exp_arch::nvidia_gpu_sm_52},
          {"5.3", oneapi_exp_arch::nvidia_gpu_sm_53},
          {"6.0", oneapi_exp_arch::nvidia_gpu_sm_60},
          {"6.1", oneapi_exp_arch::nvidia_gpu_sm_61},
          {"6.2", oneapi_exp_arch::nvidia_gpu_sm_62},
          {"7.0", oneapi_exp_arch::nvidia_gpu_sm_70},
          {"7.2", oneapi_exp_arch::nvidia_gpu_sm_72},
          {"7.5", oneapi_exp_arch::nvidia_gpu_sm_75},
          {"8.0", oneapi_exp_arch::nvidia_gpu_sm_80},
          {"8.6", oneapi_exp_arch::nvidia_gpu_sm_86},
          {"8.7", oneapi_exp_arch::nvidia_gpu_sm_87},
          {"8.9", oneapi_exp_arch::nvidia_gpu_sm_89},
          {"9.0", oneapi_exp_arch::nvidia_gpu_sm_90},
          {"gfx701", oneapi_exp_arch::amd_gpu_gfx701},
          {"gfx702", oneapi_exp_arch::amd_gpu_gfx702},
          {"gfx703", oneapi_exp_arch::amd_gpu_gfx703},
          {"gfx704", oneapi_exp_arch::amd_gpu_gfx704},
          {"gfx705", oneapi_exp_arch::amd_gpu_gfx705},
          {"gfx801", oneapi_exp_arch::amd_gpu_gfx801},
          {"gfx802", oneapi_exp_arch::amd_gpu_gfx802},
          {"gfx803", oneapi_exp_arch::amd_gpu_gfx803},
          {"gfx805", oneapi_exp_arch::amd_gpu_gfx805},
          {"gfx810", oneapi_exp_arch::amd_gpu_gfx810},
          {"gfx900", oneapi_exp_arch::amd_gpu_gfx900},
          {"gfx902", oneapi_exp_arch::amd_gpu_gfx902},
          {"gfx904", oneapi_exp_arch::amd_gpu_gfx904},
          {"gfx906", oneapi_exp_arch::amd_gpu_gfx906},
          {"gfx908", oneapi_exp_arch::amd_gpu_gfx908},
          {"gfx909", oneapi_exp_arch::amd_gpu_gfx909},
          {"gfx90a", oneapi_exp_arch::amd_gpu_gfx90a},
          {"gfx90c", oneapi_exp_arch::amd_gpu_gfx90c},
          {"gfx940", oneapi_exp_arch::amd_gpu_gfx940},
          {"gfx941", oneapi_exp_arch::amd_gpu_gfx941},
          {"gfx942", oneapi_exp_arch::amd_gpu_gfx942},
          {"gfx1010", oneapi_exp_arch::amd_gpu_gfx1010},
          {"gfx1011", oneapi_exp_arch::amd_gpu_gfx1011},
          {"gfx1012", oneapi_exp_arch::amd_gpu_gfx1012},
          {"gfx1013", oneapi_exp_arch::amd_gpu_gfx1013},
          {"gfx1030", oneapi_exp_arch::amd_gpu_gfx1030},
          {"gfx1031", oneapi_exp_arch::amd_gpu_gfx1031},
          {"gfx1032", oneapi_exp_arch::amd_gpu_gfx1032},
          {"gfx1033", oneapi_exp_arch::amd_gpu_gfx1033},
          {"gfx1034", oneapi_exp_arch::amd_gpu_gfx1034},
          {"gfx1035", oneapi_exp_arch::amd_gpu_gfx1035},
          {"gfx1036", oneapi_exp_arch::amd_gpu_gfx1036},
          {"gfx1100", oneapi_exp_arch::amd_gpu_gfx1100},
          {"gfx1101", oneapi_exp_arch::amd_gpu_gfx1101},
          {"gfx1102", oneapi_exp_arch::amd_gpu_gfx1102},
          {"gfx1103", oneapi_exp_arch::amd_gpu_gfx1103},
          {"gfx1150", oneapi_exp_arch::amd_gpu_gfx1150},
          {"gfx1151", oneapi_exp_arch::amd_gpu_gfx1151},
          {"gfx1200", oneapi_exp_arch::amd_gpu_gfx1200},
          {"gfx1201", oneapi_exp_arch::amd_gpu_gfx1201},
      };

  // Only for Intel GPU architectures
  constexpr std::pair<const int, oneapi_exp_arch> IntelGPUArchitectures[] = {
      {0x02000000, oneapi_exp_arch::intel_gpu_bdw},
      {0x02400009, oneapi_exp_arch::intel_gpu_skl},
      {0x02404009, oneapi_exp_arch::intel_gpu_kbl},
      {0x02408009, oneapi_exp_arch::intel_gpu_cfl},
      {0x0240c000, oneapi_exp_arch::intel_gpu_apl},
      {0x02410000, oneapi_exp_arch::intel_gpu_glk},
      {0x02414000, oneapi_exp_arch::intel_gpu_whl},
      {0x02418000, oneapi_exp_arch::intel_gpu_aml},
      {0x0241c000, oneapi_exp_arch::intel_gpu_cml},
      {0x02c00000, oneapi_exp_arch::intel_gpu_icllp},
      {0x02c08000, oneapi_exp_arch::intel_gpu_ehl},
      {0x03000000, oneapi_exp_arch::intel_gpu_tgllp},
      {0x03004000, oneapi_exp_arch::intel_gpu_rkl},
      {0x03008000, oneapi_exp_arch::intel_gpu_adl_s},
      {0x0300c000, oneapi_exp_arch::intel_gpu_adl_p},
      {0x03010000, oneapi_exp_arch::intel_gpu_adl_n},
      {0x03028000, oneapi_exp_arch::intel_gpu_dg1},
      {0x030dc000, oneapi_exp_arch::intel_gpu_acm_g10}, // A0
      {0x030dc001, oneapi_exp_arch::intel_gpu_acm_g10}, // A1
      {0x030dc004, oneapi_exp_arch::intel_gpu_acm_g10}, // B0
      {0x030dc008, oneapi_exp_arch::intel_gpu_acm_g10}, // C0
      {0x030e0000, oneapi_exp_arch::intel_gpu_acm_g11}, // A0
      {0x030e0004, oneapi_exp_arch::intel_gpu_acm_g11}, // B0
      {0x030e0005, oneapi_exp_arch::intel_gpu_acm_g11}, // B1
      {0x030e4000, oneapi_exp_arch::intel_gpu_acm_g12}, // A0
      {0x030f0000, oneapi_exp_arch::intel_gpu_pvc},     // XL-A0
      {0x030f0001, oneapi_exp_arch::intel_gpu_pvc},     // XL-AOP
      {0x030f0003, oneapi_exp_arch::intel_gpu_pvc},     // XT-A0
      {0x030f0005, oneapi_exp_arch::intel_gpu_pvc},     // XT-B0
      {0x030f0006, oneapi_exp_arch::intel_gpu_pvc},     // XT-B1
      {0x030f0007, oneapi_exp_arch::intel_gpu_pvc},     // XT-C0
      {0x030f4007, oneapi_exp_arch::intel_gpu_pvc_vg},  // C0
      {0x03118000, oneapi_exp_arch::intel_gpu_mtl_u},   // A0
      {0x03118004, oneapi_exp_arch::intel_gpu_mtl_u},   // B0
      {0x0311c000, oneapi_exp_arch::intel_gpu_mtl_h},   // A0
      {0x0311c004, oneapi_exp_arch::intel_gpu_mtl_h},   // B0
      {0x03128000, oneapi_exp_arch::intel_gpu_arl_h},   // A0
      {0x03128004, oneapi_exp_arch::intel_gpu_arl_h},   // B0
      {0x05004000, oneapi_exp_arch::intel_gpu_bmg_g21}, // A0
      {0x05004001, oneapi_exp_arch::intel_gpu_bmg_g21}, // A1
      {0x05004004, oneapi_exp_arch::intel_gpu_bmg_g21}, // B0
      {0x05010000, oneapi_exp_arch::intel_gpu_lnl_m},   // A0
      {0x05010001, oneapi_exp_arch::intel_gpu_lnl_m},   // A1
      {0x05010004, oneapi_exp_arch::intel_gpu_lnl_m},   // B0
      {0x07800000, oneapi_exp_arch::intel_gpu_ptl_h},   // A0
      {0x07800004, oneapi_exp_arch::intel_gpu_ptl_h},   // B0
      {0x07804000, oneapi_exp_arch::intel_gpu_ptl_u},   // A0
      {0x07804001, oneapi_exp_arch::intel_gpu_ptl_u},   // A1
  };

  // Only for Intel CPU architectures
  constexpr std::pair<const int, oneapi_exp_arch> IntelCPUArchitectures[] = {
      {8, oneapi_exp_arch::intel_cpu_spr},
      {9, oneapi_exp_arch::intel_cpu_gnr},
      {10, oneapi_exp_arch::intel_cpu_dmr},
  };
  backend CurrentBackend = getBackend();
  auto LookupIPVersion = [&, this](auto &ArchList)
      -> std::optional<ext::oneapi::experimental::architecture> {
    auto DeviceIp = get_info_impl_nocheck<UR_DEVICE_INFO_IP_VERSION>();
    if (!DeviceIp.has_val()) {
      ur_result_t Err = DeviceIp.error();
      if (Err == UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION) {
        // Not all devices support this device info query
        return std::nullopt;
      }
      getAdapter()->checkUrResult(Err);
    }

    auto Val = static_cast<int>(DeviceIp.value());
    for (const auto &Item : ArchList) {
      if (Item.first == Val)
        return Item.second;
    }
    return std::nullopt;
  };

  if (is_gpu() && (backend::ext_oneapi_level_zero == CurrentBackend ||
                   backend::opencl == CurrentBackend)) {
    return LookupIPVersion(IntelGPUArchitectures)
        .value_or(ext::oneapi::experimental::architecture::unknown);
  } else if (is_gpu() && (backend::ext_oneapi_cuda == CurrentBackend ||
                          backend::ext_oneapi_hip == CurrentBackend)) {
    auto MapArchIDToArchName = [&](const char *arch) {
      for (const auto &Item : NvidiaAmdGPUArchitectures) {
        if (std::string_view(Item.first) == arch)
          return Item.second;
      }
      return ext::oneapi::experimental::architecture::unknown;
    };
    std::string DeviceArch =
        get_info_impl<UrInfoCode<info::device::version>::value>();
    std::string_view DeviceArchSubstr =
        std::string_view{DeviceArch}.substr(0, DeviceArch.find(":"));
    return MapArchIDToArchName(DeviceArchSubstr.data());
  } else if (is_cpu() && backend::opencl == CurrentBackend) {
    return LookupIPVersion(IntelCPUArchitectures)
        .value_or(ext::oneapi::experimental::architecture::x86_64);
  } // else is not needed
  // TODO: add support of other architectures by extending with else if
  return ext::oneapi::experimental::architecture::unknown;
}
/// Constructs a SYCL device instance using the provided
/// UR device instance.
device_impl::device_impl(ur_device_handle_t Device, platform_impl &Platform,
                         device_impl::private_tag)
    : MDevice(Device), MPlatform(Platform.shared_from_this()),
      // TODO catch an exception and put it to list of asynchronous exceptions
      // for the field initializers below:
      MType(get_info_impl<UR_DEVICE_INFO_TYPE>()),
      // No need to set MRootDevice when MAlwaysRootDevice is true
      MRootDevice(Platform.MAlwaysRootDevice
                      ? nullptr
                      : get_info_impl<UR_DEVICE_INFO_PARENT_DEVICE>()),
      MUseNativeAssert(get_info_impl<UR_DEVICE_INFO_USE_NATIVE_ASSERT>()),
      MExtensions([this]() {
        auto Extensions =
            split_string(get_info_impl<UR_DEVICE_INFO_EXTENSIONS>(), ' ');
        std::sort(Extensions.begin(), Extensions.end());
        return Extensions;
      }()),
      MDeviceArch(get_architecture()),
      MDeviceName(get_info_impl<UR_DEVICE_INFO_NAME>()) {
  // TODO catch an exception and put it to list of asynchronous exceptions
  // Interoperability Constructor already calls DeviceRetain in
  // urDeviceCreateWithNativeHandle.
  getAdapter()->call<UrApiKind::urDeviceRetain>(MDevice);
}

device_impl::~device_impl() {
  try {
    // TODO catch an exception and put it to list of asynchronous exceptions
    const AdapterPtr &Adapter = getAdapter();
    ur_result_t Err =
        Adapter->call_nocheck<UrApiKind::urDeviceRelease>(MDevice);
    __SYCL_CHECK_UR_CODE_NO_EXC(Err);
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~device_impl", e);
  }
}

bool device_impl::is_affinity_supported(
    info::partition_affinity_domain AffinityDomain) const {
  auto SupportedDomains = get_info<info::device::partition_affinity_domains>();
  return std::find(SupportedDomains.begin(), SupportedDomains.end(),
                   AffinityDomain) != SupportedDomains.end();
}

cl_device_id device_impl::get() const {
  // TODO catch an exception and put it to list of asynchronous exceptions
  __SYCL_OCL_CALL(clRetainDevice, ur::cast<cl_device_id>(getNative()));
  return ur::cast<cl_device_id>(getNative());
}

platform device_impl::get_platform() const {
  return createSyclObjFromImpl<platform>(MPlatform);
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
template <>
typename info::platform::version::return_type
device_impl::get_backend_info<info::platform::version>() const {
  if (getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::platform::version info descriptor can "
                          "only be queried with an OpenCL backend");
  }
  return get_platform().get_info<info::platform::version>();
}
#endif

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
template <>
typename info::device::version::return_type
device_impl::get_backend_info<info::device::version>() const {
  if (getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::device::version info descriptor can only "
                          "be queried with an OpenCL backend");
  }
  return get_info<info::device::version>();
}
#endif

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
template <>
typename info::device::backend_version::return_type
device_impl::get_backend_info<info::device::backend_version>() const {
  if (getBackend() != backend::ext_oneapi_level_zero) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::device::backend_version info descriptor "
                          "can only be queried with a Level Zero backend");
  }
  return "";
  // Currently The Level Zero backend does not define the value of this
  // information descriptor and implementations are encouraged to return the
  // empty string as per specification.
}
#endif

bool device_impl::has_extension(std::string_view ExtensionName) const {
  return std::find(MExtensions.begin(), MExtensions.end(), ExtensionName) !=
         MExtensions.end();
}

bool device_impl::is_partition_supported(info::partition_property Prop) const {
  auto SupportedProperties = get_info<info::device::partition_properties>();
  return std::find(SupportedProperties.begin(), SupportedProperties.end(),
                   Prop) != SupportedProperties.end();
}

std::vector<device> device_impl::create_sub_devices(
    const ur_device_partition_properties_t *Properties,
    size_t SubDevicesCount) const {
  std::vector<ur_device_handle_t> SubDevices(SubDevicesCount);
  uint32_t ReturnedSubDevices = 0;
  const AdapterPtr &Adapter = getAdapter();
  Adapter->call<sycl::errc::invalid, UrApiKind::urDevicePartition>(
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
                [&res, this](const ur_device_handle_t &a_ur_device) {
                  device sycl_device = detail::createSyclObjFromImpl<device>(
                      MPlatform->getOrMakeDeviceImpl(a_ur_device));
                  res.push_back(sycl_device);
                });
  return res;
}

std::vector<device> device_impl::create_sub_devices(size_t ComputeUnits) const {
  if (!is_partition_supported(info::partition_property::partition_equally)) {
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Device does not support "
                          "sycl::info::partition_property::partition_equally.");
  }
  // If count exceeds the total number of compute units in the device, an
  // exception with the errc::invalid error code must be thrown.
  auto MaxComputeUnits = get_info<info::device::max_compute_units>();
  if (ComputeUnits > MaxComputeUnits)
    throw sycl::exception(errc::invalid,
                          "Total counts exceed max compute units");

  size_t SubDevicesCount = MaxComputeUnits / ComputeUnits;

  ur_device_partition_property_t Prop{};
  Prop.type = UR_DEVICE_PARTITION_EQUALLY;
  Prop.value.count = static_cast<uint32_t>(ComputeUnits);

  ur_device_partition_properties_t Properties{};
  Properties.stype = UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES;
  Properties.PropCount = 1;
  Properties.pProperties = &Prop;

  return create_sub_devices(&Properties, SubDevicesCount);
}

std::vector<device>
device_impl::create_sub_devices(const std::vector<size_t> &Counts) const {
  if (!is_partition_supported(info::partition_property::partition_by_counts)) {
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "Device does not support "
        "sycl::info::partition_property::partition_by_counts.");
  }

  std::vector<ur_device_partition_property_t> Props{};

  // Fill the properties vector with counts and validate it
  size_t TotalCounts = 0;
  size_t NonZeroCounts = 0;
  for (auto Count : Counts) {
    TotalCounts += Count;
    NonZeroCounts += (Count != 0) ? 1 : 0;
    Props.push_back(ur_device_partition_property_t{
        UR_DEVICE_PARTITION_BY_COUNTS, {static_cast<uint32_t>(Count)}});
  }

  ur_device_partition_properties_t Properties{};
  Properties.stype = UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES;
  Properties.pProperties = Props.data();
  Properties.PropCount = Props.size();

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

  return create_sub_devices(&Properties, Counts.size());
}

static inline std::string
affinityDomainToString(info::partition_affinity_domain AffinityDomain) {
  switch (AffinityDomain) {
#define __SYCL_AFFINITY_DOMAIN_STRING_CASE(DOMAIN)                             \
  case DOMAIN:                                                                 \
    return #DOMAIN;

    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::numa)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L4_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L3_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L2_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L1_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::next_partitionable)
#undef __SYCL_AFFINITY_DOMAIN_STRING_CASE
  default:
    assert(false && "Missing case for affinity domain.");
    return "unknown";
  }
}

std::vector<device> device_impl::create_sub_devices(
    info::partition_affinity_domain AffinityDomain) const {
  if (!is_partition_supported(
          info::partition_property::partition_by_affinity_domain)) {
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "Device does not support "
        "sycl::info::partition_property::partition_by_affinity_domain.");
  }
  if (!is_affinity_supported(AffinityDomain)) {
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Device does not support " +
                              affinityDomainToString(AffinityDomain) + ".");
  }

  ur_device_partition_property_t Prop;
  Prop.type = UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
  Prop.value.affinity_domain =
      static_cast<ur_device_affinity_domain_flags_t>(AffinityDomain);

  ur_device_partition_properties_t Properties{};
  Properties.stype = UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES;
  Properties.PropCount = 1;
  Properties.pProperties = &Prop;

  uint32_t SubDevicesCount = 0;
  const AdapterPtr &Adapter = getAdapter();
  Adapter->call<sycl::errc::invalid, UrApiKind::urDevicePartition>(
      MDevice, &Properties, 0, nullptr, &SubDevicesCount);

  return create_sub_devices(&Properties, SubDevicesCount);
}

std::vector<device> device_impl::create_sub_devices() const {
  if (!is_partition_supported(
          info::partition_property::ext_intel_partition_by_cslice)) {
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "Device does not support "
        "sycl::info::partition_property::ext_intel_partition_by_cslice.");
  }

  ur_device_partition_property_t Prop;
  Prop.type = UR_DEVICE_PARTITION_BY_CSLICE;

  ur_device_partition_properties_t Properties{};
  Properties.stype = UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES;
  Properties.pProperties = &Prop;
  Properties.PropCount = 1;

  uint32_t SubDevicesCount = 0;
  const AdapterPtr &Adapter = getAdapter();
  Adapter->call<UrApiKind::urDevicePartition>(MDevice, &Properties, 0, nullptr,
                                              &SubDevicesCount);

  return create_sub_devices(&Properties, SubDevicesCount);
}

ur_native_handle_t device_impl::getNative() const {
  auto Adapter = getAdapter();
  ur_native_handle_t Handle;
  Adapter->call<UrApiKind::urDeviceGetNativeHandle>(getHandleRef(), &Handle);
  if (getBackend() == backend::opencl) {
    __SYCL_OCL_CALL(clRetainDevice, ur::cast<cl_device_id>(Handle));
  }
  return Handle;
}

bool device_impl::has(aspect Aspect) const {
  switch (Aspect) {
  case aspect::host:
    // Deprecated
    return false;
  case aspect::cpu:
    return is_cpu();
  case aspect::gpu:
    return is_gpu();
  case aspect::accelerator:
    return is_accelerator();
  case aspect::custom:
    return false;
  // TODO: Implement this for FPGA emulator.
  case aspect::emulated:
    return false;
  case aspect::host_debuggable:
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
  case aspect::ext_intel_mem_channel:
    return get_info<info::device::ext_intel_mem_channel>();
  case aspect::ext_oneapi_cuda_cluster_group:
    return get_info<info::device::ext_oneapi_cuda_cluster_group>();
  case aspect::usm_atomic_host_allocations:
    return (get_info_impl<UR_DEVICE_INFO_USM_HOST_SUPPORT>() &
            UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS);
  case aspect::usm_shared_allocations:
    return get_info<info::device::usm_shared_allocations>();
  case aspect::usm_atomic_shared_allocations:
    return (get_info_impl<UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT>() &
            UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS);
  case aspect::usm_restricted_shared_allocations:
    return get_info<info::device::usm_restricted_shared_allocations>();
  case aspect::usm_system_allocations:
    return get_info<info::device::usm_system_allocations>();
  case aspect::ext_intel_device_id:
    return has_info_desc(UR_DEVICE_INFO_DEVICE_ID);
  case aspect::ext_intel_pci_address:
    return has_info_desc(UR_DEVICE_INFO_PCI_ADDRESS);
  case aspect::ext_intel_gpu_eu_count:
    return has_info_desc(UR_DEVICE_INFO_GPU_EU_COUNT);
  case aspect::ext_intel_gpu_eu_simd_width:
    return has_info_desc(UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH);
  case aspect::ext_intel_gpu_slices:
    return has_info_desc(UR_DEVICE_INFO_GPU_EU_SLICES);
  case aspect::ext_intel_gpu_subslices_per_slice:
    return has_info_desc(UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE);
  case aspect::ext_intel_gpu_eu_count_per_subslice:
    return has_info_desc(UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE);
  case aspect::ext_intel_gpu_hw_threads_per_eu:
    return has_info_desc(UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU);
  case aspect::ext_intel_free_memory:
    return has_info_desc(UR_DEVICE_INFO_GLOBAL_MEM_FREE);
  case aspect::ext_intel_memory_clock_rate:
    return has_info_desc(UR_DEVICE_INFO_MEMORY_CLOCK_RATE);
  case aspect::ext_intel_memory_bus_width:
    return has_info_desc(UR_DEVICE_INFO_MEMORY_BUS_WIDTH);
  case aspect::ext_intel_device_info_uuid:
    return has_info_desc(UR_DEVICE_INFO_UUID);
  case aspect::ext_intel_max_mem_bandwidth:
    // currently not supported
    return false;
  case aspect::ext_intel_current_clock_throttle_reasons:
    return has_info_desc(UR_DEVICE_INFO_CURRENT_CLOCK_THROTTLE_REASONS);
  case aspect::ext_intel_fan_speed:
    return has_info_desc(UR_DEVICE_INFO_FAN_SPEED);
  case aspect::ext_intel_power_limits:
    return has_info_desc(UR_DEVICE_INFO_MIN_POWER_LIMIT) &&
           has_info_desc(UR_DEVICE_INFO_MAX_POWER_LIMIT);
  case aspect::ext_oneapi_srgb:
    return get_info<info::device::ext_oneapi_srgb>();
  case aspect::ext_oneapi_native_assert:
    return MUseNativeAssert;
  case aspect::ext_oneapi_cuda_async_barrier: {
    return get_info_impl_nocheck<UR_DEVICE_INFO_ASYNC_BARRIER>().value_or(0);
  }
  case aspect::ext_intel_legacy_image: {
    return get_info_impl_nocheck<UR_DEVICE_INFO_IMAGE_SUPPORT>().value_or(0);
  }
  case aspect::ext_oneapi_bindless_images: {
    return get_info_impl_nocheck<UR_DEVICE_INFO_BINDLESS_IMAGES_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_bindless_images_shared_usm: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_BINDLESS_IMAGES_SHARED_USM_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_bindless_images_1d_usm: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_BINDLESS_IMAGES_1D_USM_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_bindless_images_2d_usm: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_BINDLESS_IMAGES_2D_USM_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_external_memory_import: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_EXTERNAL_MEMORY_IMPORT_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_external_semaphore_import: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_EXTERNAL_SEMAPHORE_IMPORT_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_mipmap: {
    return get_info_impl_nocheck<UR_DEVICE_INFO_MIPMAP_SUPPORT_EXP>().value_or(
        0);
  }
  case aspect::ext_oneapi_mipmap_anisotropy: {
    return get_info_impl_nocheck<UR_DEVICE_INFO_MIPMAP_ANISOTROPY_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_mipmap_level_reference: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_MIPMAP_LEVEL_REFERENCE_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_bindless_sampled_image_fetch_1d_usm: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_USM_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_bindless_sampled_image_fetch_1d: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_bindless_sampled_image_fetch_2d_usm: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_USM_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_bindless_sampled_image_fetch_2d: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_bindless_sampled_image_fetch_3d: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_3D_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_bindless_images_gather: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_BINDLESS_IMAGES_GATHER_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_cubemap: {
    return get_info_impl_nocheck<UR_DEVICE_INFO_CUBEMAP_SUPPORT_EXP>().value_or(
        0);
  }
  case aspect::ext_oneapi_cubemap_seamless_filtering: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_CUBEMAP_SEAMLESS_FILTERING_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_image_array: {
    return get_info_impl_nocheck<UR_DEVICE_INFO_IMAGE_ARRAY_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_unique_addressing_per_dim: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_BINDLESS_UNIQUE_ADDRESSING_PER_DIM_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_bindless_images_sample_1d_usm: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_BINDLESS_SAMPLE_1D_USM_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_bindless_images_sample_2d_usm: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_BINDLESS_SAMPLE_2D_USM_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_intel_esimd: {
    return get_info_impl_nocheck<UR_DEVICE_INFO_ESIMD_SUPPORT>().value_or(0);
  }
  case aspect::ext_oneapi_ballot_group:
  case aspect::ext_oneapi_fixed_size_group:
  case aspect::ext_oneapi_opportunistic_group: {
    return (this->getBackend() == backend::ext_oneapi_level_zero) ||
           (this->getBackend() == backend::opencl) ||
           (this->getBackend() == backend::ext_oneapi_cuda);
  }
  case aspect::ext_oneapi_tangle_group: {
    // TODO: tangle_group is not currently supported for CUDA devices. Add when
    //       implemented.
    return (this->getBackend() == backend::ext_oneapi_level_zero) ||
           (this->getBackend() == backend::opencl);
  }
  case aspect::ext_intel_matrix: {
    using arch = sycl::ext::oneapi::experimental::architecture;
    const arch supported_archs[] = {
        arch::intel_cpu_spr,     arch::intel_cpu_gnr,
        arch::intel_cpu_dmr,     arch::intel_gpu_pvc,
        arch::intel_gpu_dg2_g10, arch::intel_gpu_dg2_g11,
        arch::intel_gpu_dg2_g12, arch::intel_gpu_bmg_g21,
        arch::intel_gpu_lnl_m,   arch::intel_gpu_arl_h,
        arch::intel_gpu_ptl_h,   arch::intel_gpu_ptl_u,
    };
    try {
      return std::any_of(
          std::begin(supported_archs), std::end(supported_archs),
          [=](const arch a) { return this->extOneapiArchitectureIs(a); });
    } catch (const sycl::exception &) {
      // If we're here it means the device does not support architecture
      // querying
      return false;
    }
  }
  case aspect::ext_oneapi_is_composite: {
    auto components = get_info<
        sycl::ext::oneapi::experimental::info::device::component_devices>();
    // Any device with ext_oneapi_is_composite aspect will have at least two
    // constituent component devices.
    return components.size() >= 2;
  }
  case aspect::ext_oneapi_is_component: {
    return get_info_impl_nocheck<UR_DEVICE_INFO_COMPOSITE_DEVICE>().value_or(
               nullptr) != nullptr;
  }
  case aspect::ext_oneapi_graph: {
    ur_device_command_buffer_update_capability_flags_t UpdateCapabilities;
    bool CallSuccessful =
        getAdapter()->call_nocheck<UrApiKind::urDeviceGetInfo>(
            MDevice, UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP,
            sizeof(UpdateCapabilities), &UpdateCapabilities,
            nullptr) == UR_RESULT_SUCCESS;
    if (!CallSuccessful) {
      return false;
    }

    /* The kernel handle update capability is not yet required for the
     * ext_oneapi_graph aspect */
    ur_device_command_buffer_update_capability_flags_t RequiredCapabilities =
        UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_ARGUMENTS |
        UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_LOCAL_WORK_SIZE |
        UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_SIZE |
        UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_OFFSET |
        UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_HANDLE;

    return has(aspect::ext_oneapi_limited_graph) &&
           (UpdateCapabilities & RequiredCapabilities) == RequiredCapabilities;
  }
  case aspect::ext_oneapi_limited_graph: {
    bool SupportsCommandBuffers = false;
    bool CallSuccessful =
        getAdapter()->call_nocheck<UrApiKind::urDeviceGetInfo>(
            MDevice, UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP,
            sizeof(SupportsCommandBuffers), &SupportsCommandBuffers,
            nullptr) == UR_RESULT_SUCCESS;
    if (!CallSuccessful) {
      return false;
    }

    return SupportsCommandBuffers;
  }
  case aspect::ext_oneapi_private_alloca: {
    // Extension only supported on SPIR-V targets.
    backend be = getBackend();
    return be == sycl::backend::ext_oneapi_level_zero ||
           be == sycl::backend::opencl;
  }
  case aspect::ext_oneapi_queue_profiling_tag: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_TIMESTAMP_RECORDING_SUPPORT_EXP>()
        .value_or(0);
  }
  case aspect::ext_oneapi_virtual_mem: {
    return get_info_impl_nocheck<UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT>()
        .value_or(0);
  }
  case aspect::ext_intel_fpga_task_sequence: {
    return is_accelerator();
  }
  case aspect::ext_oneapi_atomic16: {
    // Likely L0 doesn't check it properly. Need to double-check.
    return has_extension("cl_ext_float_atomics");
  }
  case aspect::ext_oneapi_virtual_functions: {
    // TODO: move to UR like e.g. aspect::ext_oneapi_virtual_mem
    backend BE = getBackend();
    bool isCompatibleBE = BE == sycl::backend::ext_oneapi_level_zero ||
                          BE == sycl::backend::opencl;
    return (is_cpu() || is_gpu()) && isCompatibleBE;
  }
  case aspect::ext_intel_spill_memory_size: {
    backend BE = getBackend();
    bool isCompatibleBE = BE == sycl::backend::ext_oneapi_level_zero;
    return is_gpu() && isCompatibleBE;
  }
  case aspect::ext_oneapi_async_memory_alloc: {
    return get_info_impl_nocheck<
               UR_DEVICE_INFO_ASYNC_USM_ALLOCATIONS_SUPPORT_EXP>()
        .value_or(0);
  }
  }

  return false; // This device aspect has not been implemented yet.
}

// On the first call this function queries for device timestamp
// along with host synchronized timestamp and stores it in member variable
// MDeviceHostBaseTime. Subsequent calls to this function would just retrieve
// the host timestamp, compute difference against the host timestamp in
// MDeviceHostBaseTime and calculate the device timestamp based on the
// difference.
//
// The MDeviceHostBaseTime is refreshed with new device and host timestamp
// after a certain interval (determined by TimeTillRefresh) to account for
// clock drift between host and device.
//
uint64_t device_impl::getCurrentDeviceTime() {
  using namespace std::chrono;
  uint64_t HostTime =
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
          .count();

  // To account for potential clock drift between host clock and device clock.
  // The value set is arbitrary: 200 seconds
  constexpr uint64_t TimeTillRefresh = 200e9;
  assert(HostTime >= MDeviceHostBaseTime.second);
  uint64_t Diff = HostTime - MDeviceHostBaseTime.second;

  // If getCurrentDeviceTime is called for the first time or we have to refresh.
  if (!MDeviceHostBaseTime.second || Diff > TimeTillRefresh) {
    const auto &Adapter = getAdapter();
    auto Result = Adapter->call_nocheck<UrApiKind::urDeviceGetGlobalTimestamps>(
        MDevice, &MDeviceHostBaseTime.first, &MDeviceHostBaseTime.second);
    // We have to remember base host timestamp right after UR call and it is
    // going to be used for calculation of the device timestamp at the next
    // getCurrentDeviceTime() call. We need to do it here because getAdapter()
    // and urDeviceGetGlobalTimestamps calls may take significant amount of
    // time, for example on the first call to getAdapter adapters may need to be
    // initialized. If we use timestamp from the beginning of the function then
    // the difference between host timestamps of the current
    // getCurrentDeviceTime and the next getCurrentDeviceTime will be incorrect
    // because it will include execution time of the code before we get device
    // timestamp from urDeviceGetGlobalTimestamps.
    HostTime =
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
            .count();
    if (Result == UR_RESULT_ERROR_INVALID_OPERATION) {
      // NOTE(UR port): Removed the call to GetLastError because  we shouldn't
      // be calling it after ERROR_INVALID_OPERATION: there is no
      // adapter-specific error.
      throw detail::set_ur_error(
          sycl::exception(
              make_error_code(errc::feature_not_supported),
              "Device and/or backend does not support querying timestamp."),
          UR_RESULT_ERROR_INVALID_OPERATION);
    } else {
      Adapter->checkUrResult<errc::feature_not_supported>(Result);
    }
    // Until next sync we will compute device time based on the host time
    // returned in HostTime, so make this our base host time.
    MDeviceHostBaseTime.second = HostTime;
    Diff = 0;
  }
  return MDeviceHostBaseTime.first + Diff;
}

bool device_impl::extOneapiCanBuild(
    ext::oneapi::experimental::source_language Language) {
  try {
    // Get the shared_ptr to this object from the platform that owns it.
    device_impl &Self = MPlatform->getOrMakeDeviceImpl(MDevice);
    return sycl::ext::oneapi::experimental::detail::
        is_source_kernel_bundle_supported(Language,
                                          std::vector<device_impl *>{&Self});

  } catch (sycl::exception &) {
    return false;
  }
}

bool device_impl::extOneapiCanCompile(
    ext::oneapi::experimental::source_language Language) {
  try {
    // Currently only SYCL language is supported for compiling.
    device_impl &Self = MPlatform->getOrMakeDeviceImpl(MDevice);
    return Language == ext::oneapi::experimental::source_language::sycl &&
           sycl::ext::oneapi::experimental::detail::
               is_source_kernel_bundle_supported(
                   Language, std::vector<device_impl *>{&Self});
  } catch (sycl::exception &) {
    return false;
  }
}

// Returns the strongest guarantee that can be provided by the host device for
// threads created at threadScope from a coordination scope given by
// coordinationScope
sycl::ext::oneapi::experimental::forward_progress_guarantee
device_impl::getHostProgressGuarantee(
    ext::oneapi::experimental::execution_scope,
    ext::oneapi::experimental::execution_scope) {
  return sycl::ext::oneapi::experimental::forward_progress_guarantee::
      weakly_parallel;
}

// Returns the strongest progress guarantee that can be provided by this device
// for threads created at threadScope from the coordination scope given by
// coordinationScope.
sycl::ext::oneapi::experimental::forward_progress_guarantee
device_impl::getProgressGuarantee(
    ext::oneapi::experimental::execution_scope threadScope,
    ext::oneapi::experimental::execution_scope coordinationScope) const {
  using forward_progress_guarantee =
      ext::oneapi::experimental::forward_progress_guarantee;
  using execution_scope = ext::oneapi::experimental::execution_scope;
  const int executionScopeSize = 4;
  (void)coordinationScope;
  int threadScopeNum = static_cast<int>(threadScope);
  // we get the immediate progress guarantee that is provided by each scope
  // between root_group and threadScope and then return the weakest of these.
  // Counterintuitively, this corresponds to taking the max of the enum values
  // because of how the forward_progress_guarantee enum values are declared.
  int guaranteeNum = static_cast<int>(
      getImmediateProgressGuarantee(execution_scope::root_group));
  for (int currentScope = executionScopeSize - 2; currentScope > threadScopeNum;
       --currentScope) {
    guaranteeNum = std::max(guaranteeNum,
                            static_cast<int>(getImmediateProgressGuarantee(
                                static_cast<execution_scope>(currentScope))));
  }
  return static_cast<forward_progress_guarantee>(guaranteeNum);
}

bool device_impl::supportsForwardProgress(
    ext::oneapi::experimental::forward_progress_guarantee guarantee,
    ext::oneapi::experimental::execution_scope threadScope,
    ext::oneapi::experimental::execution_scope coordinationScope) const {
  auto guarantees = getProgressGuaranteesUpTo(
      getProgressGuarantee(threadScope, coordinationScope));
  return std::find(guarantees.begin(), guarantees.end(), guarantee) !=
         guarantees.end();
}

// Returns the progress guarantee provided for a coordination scope
// given by coordination_scope for threads created at a scope
// immediately below coordination_scope. For example, for root_group
// coordination scope it returns the progress guarantee provided
// at root_group for threads created at work_group.
ext::oneapi::experimental::forward_progress_guarantee
device_impl::getImmediateProgressGuarantee(
    ext::oneapi::experimental::execution_scope coordination_scope) const {
  using forward_progress_guarantee =
      ext::oneapi::experimental::forward_progress_guarantee;
  using execution_scope = ext::oneapi::experimental::execution_scope;
  if (is_cpu() && getBackend() == backend::opencl) {
    switch (coordination_scope) {
    case execution_scope::root_group:
      return forward_progress_guarantee::parallel;
    case execution_scope::work_group:
    case execution_scope::sub_group:
      return forward_progress_guarantee::weakly_parallel;
    default:
      throw sycl::exception(sycl::errc::invalid,
                            "Work item is not a valid coordination scope!");
    }
  } else if (is_gpu() && getBackend() == backend::ext_oneapi_level_zero) {
    switch (coordination_scope) {
    case execution_scope::root_group:
    case execution_scope::work_group:
      return forward_progress_guarantee::concurrent;
    case execution_scope::sub_group:
      return forward_progress_guarantee::weakly_parallel;
    default:
      throw sycl::exception(sycl::errc::invalid,
                            "Work item is not a valid coordination scope!");
    }
  }
  return forward_progress_guarantee::weakly_parallel;
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
#define EXPORT_GET_INFO(PARAM)                                                 \
  template <>                                                                  \
  __SYCL_EXPORT PARAM::return_type device_impl::get_info<PARAM>() const {      \
    return get_info_abi_workaround<PARAM>();                                   \
  }

// clang-format off
EXPORT_GET_INFO(ext::intel::info::device::device_id)
EXPORT_GET_INFO(ext::intel::info::device::pci_address)
EXPORT_GET_INFO(ext::intel::info::device::gpu_eu_count)
EXPORT_GET_INFO(ext::intel::info::device::gpu_eu_simd_width)
EXPORT_GET_INFO(ext::intel::info::device::gpu_slices)
EXPORT_GET_INFO(ext::intel::info::device::gpu_subslices_per_slice)
EXPORT_GET_INFO(ext::intel::info::device::gpu_eu_count_per_subslice)
EXPORT_GET_INFO(ext::intel::info::device::gpu_hw_threads_per_eu)
EXPORT_GET_INFO(ext::intel::info::device::max_mem_bandwidth)
EXPORT_GET_INFO(ext::intel::info::device::uuid)
EXPORT_GET_INFO(ext::intel::info::device::free_memory)
EXPORT_GET_INFO(ext::intel::info::device::memory_clock_rate)
EXPORT_GET_INFO(ext::intel::info::device::memory_bus_width)
EXPORT_GET_INFO(ext::intel::info::device::max_compute_queue_indices)
EXPORT_GET_INFO(ext::intel::esimd::info::device::has_2d_block_io_support)
EXPORT_GET_INFO(ext::intel::info::device::current_clock_throttle_reasons)
EXPORT_GET_INFO(ext::intel::info::device::fan_speed)
EXPORT_GET_INFO(ext::intel::info::device::min_power_limit)
EXPORT_GET_INFO(ext::intel::info::device::max_power_limit)

EXPORT_GET_INFO(ext::codeplay::experimental::info::device::supports_fusion)
EXPORT_GET_INFO(ext::codeplay::experimental::info::device::max_registers_per_work_group)

EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_global_work_groups)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_work_groups<1>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_work_groups<2>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_work_groups<3>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::work_group_progress_capabilities<ext::oneapi::experimental::execution_scope::root_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::sub_group_progress_capabilities<ext::oneapi::experimental::execution_scope::root_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::sub_group_progress_capabilities<ext::oneapi::experimental::execution_scope::work_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::work_item_progress_capabilities<ext::oneapi::experimental::execution_scope::root_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::work_item_progress_capabilities<ext::oneapi::experimental::execution_scope::work_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::work_item_progress_capabilities<ext::oneapi::experimental::execution_scope::sub_group>)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::architecture)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::matrix_combinations)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::image_row_pitch_align)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_image_linear_row_pitch)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_image_linear_width)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::max_image_linear_height)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::mipmap_max_anisotropy)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::component_devices)
EXPORT_GET_INFO(ext::oneapi::experimental::info::device::composite_device)
EXPORT_GET_INFO(ext::oneapi::info::device::num_compute_units)
// clang-format on

#undef EXPORT_GET_INFO
#endif

} // namespace detail
} // namespace _V1
} // namespace sycl
