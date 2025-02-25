//===--------- device.cpp - Level Zero Adapter ----------------------------===//
//
// Copyright (C) 2023-2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "device.hpp"
#include "adapter.hpp"
#include "logger/ur_logger.hpp"
#include "ur_interface_loader.hpp"
#include "ur_level_zero.hpp"
#include "ur_util.hpp"
#include <algorithm>
#include <climits>
#include <optional>

// UR_L0_USE_COPY_ENGINE can be set to an integer value, or
// a pair of integer values of the form "lower_index:upper_index".
// Here, the indices point to copy engines in a list of all available copy
// engines.
// This functions returns this pair of indices.
// If the user specifies only a single integer, a value of 0 indicates that
// the copy engines will not be used at all. A value of 1 indicates that all
// available copy engines can be used.
const std::pair<int, int>
getRangeOfAllowedCopyEngines(const ur_device_handle_t &Device) {
  const char *UrRet = std::getenv("UR_L0_USE_COPY_ENGINE");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE");
  static const char *EnvVar = UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  // If the environment variable is not set, no copy engines are used when
  // immediate commandlists are being used. For standard commandlists all are
  // used.
  if (!EnvVar) {
    if (Device->ImmCommandListUsed)
      return std::pair<int, int>(0, 0); // Only main copy engine will be used.
    return std::pair<int, int>(0, INT_MAX); // All copy engines will be used.
  }
  std::string CopyEngineRange = EnvVar;
  // Environment variable can be a single integer or a pair of integers
  // separated by ":"
  auto pos = CopyEngineRange.find(":");
  if (pos == std::string::npos) {
    bool UseCopyEngine = (std::stoi(CopyEngineRange) != 0);
    if (UseCopyEngine)
      return std::pair<int, int>(0, INT_MAX); // All copy engines can be used.
    return std::pair<int, int>(-1, -1);       // No copy engines will be used.
  }
  int LowerCopyEngineIndex = std::stoi(CopyEngineRange.substr(0, pos));
  int UpperCopyEngineIndex = std::stoi(CopyEngineRange.substr(pos + 1));
  if ((LowerCopyEngineIndex > UpperCopyEngineIndex) ||
      (LowerCopyEngineIndex < -1) || (UpperCopyEngineIndex < -1)) {
    logger::error("UR_L0_LEVEL_ZERO_USE_COPY_ENGINE: invalid value provided, "
                  "default set.");
    LowerCopyEngineIndex = 0;
    UpperCopyEngineIndex = INT_MAX;
  }
  return std::pair<int, int>(LowerCopyEngineIndex, UpperCopyEngineIndex);
}

namespace ur::level_zero {

ur_result_t urDeviceGet(
    /// [in] handle of the platform instance
    ur_platform_handle_t Platform,
    /// [in] the type of the devices.
    ur_device_type_t DeviceType,
    /// [in] the number of devices to be added to phDevices. If phDevices in not
    /// NULL then NumEntries should be greater than zero, otherwise
    /// ::UR_RESULT_ERROR_INVALID_SIZE, will be returned.
    uint32_t NumEntries,
    /// [out][optional][range(0, NumEntries)] array of handle of devices. If
    /// NumEntries is less than the number of devices available, then
    /// platform shall only retrieve that number of devices.
    ur_device_handle_t *Devices,
    /// [out][optional] pointer to the number of devices. pNumDevices will be
    /// updated with the total number of devices available.
    uint32_t *NumDevices) {

  auto Res = Platform->populateDeviceCacheIfNeeded();
  if (Res != UR_RESULT_SUCCESS) {
    return Res;
  }

  // Filter available devices based on input DeviceType.
  std::vector<ur_device_handle_t> MatchedDevices;
  std::shared_lock<ur_shared_mutex> Lock(Platform->URDevicesCacheMutex);
  // We need to filter out composite devices when
  // ZE_FLAT_DEVICE_HIERARCHY=COMBINED. We can know if we are in combined
  // mode depending on the return value of zeDeviceGetRootDevice:
  //   - If COMPOSITE, L0 returns cards as devices. Since we filter out
  //     subdevices early, zeDeviceGetRootDevice must return nullptr, because we
  //     only query for root-devices and they don't have any device higher up in
  //     the hierarchy.
  //   - If FLAT,  according to L0 spec, zeDeviceGetRootDevice always returns
  //     nullptr in this mode.
  //   - If COMBINED, L0 returns tiles as devices, and zeDeviceGetRootdevice
  //     returns the card containing a given tile.
  bool isCombinedMode =
      std::any_of(Platform->URDevicesCache.begin(),
                  Platform->URDevicesCache.end(), [](const auto &D) {
                    if (D->isSubDevice())
                      return false;
                    ze_device_handle_t RootDev = nullptr;
                    // Query Root Device for root-devices.
                    // We cannot use ZE2UR_CALL because under some circumstances
                    // this call may return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE,
                    // and ZE2UR_CALL will abort because it's not
                    // UR_RESULT_SUCCESS. Instead, we use ZE_CALL_NOCHECK and we
                    // check manually that the result is either
                    // ZE_RESULT_SUCCESS or ZE_RESULT_ERROR_UNSUPPORTED_FEATURE.
                    auto errc = ZE_CALL_NOCHECK(zeDeviceGetRootDevice,
                                                (D->ZeDevice, &RootDev));
                    return (errc == ZE_RESULT_SUCCESS && RootDev != nullptr);
                  });
  for (auto &D : Platform->URDevicesCache) {
    // Only ever return root-devices from urDeviceGet, but the
    // devices cache also keeps sub-devices.
    if (D->isSubDevice())
      continue;

    bool Matched = false;
    switch (DeviceType) {
    case UR_DEVICE_TYPE_ALL:
      Matched = true;
      break;
    case UR_DEVICE_TYPE_GPU:
    case UR_DEVICE_TYPE_DEFAULT:
      Matched = (D->ZeDeviceProperties->type == ZE_DEVICE_TYPE_GPU);
      break;
    case UR_DEVICE_TYPE_CPU:
      Matched = (D->ZeDeviceProperties->type == ZE_DEVICE_TYPE_CPU);
      break;
    case UR_DEVICE_TYPE_FPGA:
      Matched = D->ZeDeviceProperties->type == ZE_DEVICE_TYPE_FPGA;
      break;
    case UR_DEVICE_TYPE_MCA:
      Matched = D->ZeDeviceProperties->type == ZE_DEVICE_TYPE_MCA;
      break;
    default:
      Matched = false;
      logger::warning("Unknown device type");
      break;
    }

    if (Matched) {
      bool isComposite =
          isCombinedMode && (D->ZeDeviceProperties->flags &
                             ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE) == 0;
      if (!isComposite)
        MatchedDevices.push_back(D.get());
    }
  }

  uint32_t ZeDeviceCount = MatchedDevices.size();

  auto N = (std::min)(ZeDeviceCount, NumEntries);
  if (Devices)
    std::copy_n(MatchedDevices.begin(), N, Devices);

  if (NumDevices) {
    if (*NumDevices == 0)
      *NumDevices = ZeDeviceCount;
    else
      *NumDevices = N;
  }

  return UR_RESULT_SUCCESS;
}

uint64_t calculateGlobalMemSize(ur_device_handle_t Device) {
  // Cache GlobalMemSize
  Device->ZeGlobalMemSize.Compute =
      [Device](struct ze_global_memsize &GlobalMemSize) {
        for (const auto &ZeDeviceMemoryExtProperty :
             Device->ZeDeviceMemoryProperties->second) {
          GlobalMemSize.value += ZeDeviceMemoryExtProperty.physicalSize;
        }
        if (GlobalMemSize.value == 0) {
          for (const auto &ZeDeviceMemoryProperty :
               Device->ZeDeviceMemoryProperties->first) {
            GlobalMemSize.value += ZeDeviceMemoryProperty.totalSize;
          }
        }
      };
  return Device->ZeGlobalMemSize.get().value;
}

ur_result_t urDeviceGetInfo(
    /// [in] handle of the device instance
    ur_device_handle_t Device,
    /// [in] type of the info to retrieve
    ur_device_info_t ParamName,
    /// [in] the number of bytes pointed to by ParamValue.
    size_t propSize,
    /// [out][optional] array of bytes holding the info. If propSize is not
    /// equal to or greater than the real number of bytes needed to return the
    /// info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    /// pDeviceInfo is not used.
    void *ParamValue,
    /// [out][optional] pointer to the actual size in bytes of the queried
    /// infoType.
    size_t *pSize) {
  UrReturnHelper ReturnValue(propSize, ParamValue, pSize);

  ze_device_handle_t ZeDevice = Device->ZeDevice;

  switch ((int)ParamName) {
  case UR_DEVICE_INFO_TYPE: {
    switch (Device->ZeDeviceProperties->type) {
    case ZE_DEVICE_TYPE_GPU:
      return ReturnValue(UR_DEVICE_TYPE_GPU);
    case ZE_DEVICE_TYPE_CPU:
      return ReturnValue(UR_DEVICE_TYPE_CPU);
    case ZE_DEVICE_TYPE_FPGA:
      return ReturnValue(UR_DEVICE_TYPE_FPGA);
    default:
      logger::error("This device type is not supported");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  }
  case UR_DEVICE_INFO_PARENT_DEVICE:
    return ReturnValue(Device->RootDevice);
  case UR_DEVICE_INFO_PLATFORM:
    return ReturnValue(Device->Platform);
  case UR_DEVICE_INFO_VENDOR_ID:
    return ReturnValue(uint32_t{Device->ZeDeviceProperties->vendorId});
  case UR_DEVICE_INFO_UUID: {
    // Intel extension for device UUID. This returns the UUID as
    // std::array<std::byte, 16>. For details about this extension,
    // see sycl/doc/extensions/supported/sycl_ext_intel_device_info.md.
    const auto &UUID = Device->ZeDeviceProperties->uuid.id;
    return ReturnValue(UUID, sizeof(UUID));
  }
  case UR_DEVICE_INFO_ATOMIC_64:
    return ReturnValue(
        static_cast<ur_bool_t>(Device->ZeDeviceModuleProperties->flags &
                               ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS));
  case UR_DEVICE_INFO_EXTENSIONS: {
    // Convention adopted from OpenCL:
    //     "Returns a space separated list of extension names (the extension
    // names themselves do not contain any spaces) supported by the device."
    //
    // TODO: Use proper mechanism to get this information from Level Zero after
    // it is added to Level Zero.
    // Hardcoding the few we know are supported by the current hardware.
    //
    //
    std::string SupportedExtensions;

    // cl_khr_il_program - OpenCL 2.0 KHR extension for SPIR-V support. Core
    //   feature in >OpenCL 2.1
    // cl_khr_subgroups - Extension adds support for implementation-controlled
    //   subgroups.
    // cl_intel_subgroups - Extension adds subgroup features, defined by Intel.
    // cl_intel_subgroups_short - Extension adds subgroup functions described in
    //   the cl_intel_subgroups extension to support 16-bit integer data types
    //   for performance.
    // cl_intel_required_subgroup_size - Extension to allow programmers to
    //   optionally specify the required subgroup size for a kernel function.
    // cl_khr_fp16 - Optional half floating-point support.
    // cl_khr_fp64 - Support for double floating-point precision.
    // cl_khr_int64_base_atomics, cl_khr_int64_extended_atomics - Optional
    //   extensions that implement atomic operations on 64-bit signed and
    //   unsigned integers to locations in __global and __local memory.
    // cl_khr_3d_image_writes - Extension to enable writes to 3D image memory
    //   objects.
    //
    // Hardcoding some extensions we know are supported by all Level Zero
    // devices.
    SupportedExtensions += (ZE_SUPPORTED_EXTENSIONS);
    if (Device->ZeDeviceModuleProperties->flags & ZE_DEVICE_MODULE_FLAG_FP16)
      SupportedExtensions += ("cl_khr_fp16 ");
    if (Device->ZeDeviceModuleProperties->flags & ZE_DEVICE_MODULE_FLAG_FP64)
      SupportedExtensions += ("cl_khr_fp64 ");
    if (Device->ZeDeviceModuleProperties->flags &
        ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS)
      // int64AtomicsSupported indicates support for both.
      SupportedExtensions +=
          ("cl_khr_int64_base_atomics cl_khr_int64_extended_atomics ");
    if (Device->ZeDeviceImageProperties->maxImageDims3D > 0)
      // Supports reading and writing of images.
      SupportedExtensions += ("cl_khr_3d_image_writes ");

    // L0 does not tell us if bfloat16 is supported.
    // For now, assume ATS and PVC support it.
    // TODO: change the way we detect bfloat16 support.
    if ((Device->ZeDeviceProperties->deviceId & 0xfff) == 0x201 ||
        (Device->ZeDeviceProperties->deviceId & 0xff0) == 0xbd0)
      SupportedExtensions += ("cl_intel_bfloat16_conversions ");

    // Return supported for the UR command-buffer experimental feature
    SupportedExtensions += ("ur_exp_command_buffer ");
    // Return supported for the UR multi-device compile experimental feature
    SupportedExtensions += ("ur_exp_multi_device_compile ");
    SupportedExtensions += ("ur_exp_usm_p2p ");

    return ReturnValue(SupportedExtensions.c_str());
  }
  case UR_DEVICE_INFO_NAME:
    return ReturnValue(Device->ZeDeviceProperties->name);
  // zeModuleCreate allows using root device module for sub-devices:
  // > The application must only use the module for the device, or its
  // > sub-devices, which was provided during creation.
  case UR_DEVICE_INFO_BUILD_ON_SUBDEVICE:
    return ReturnValue(ur_bool_t{0});
  case UR_DEVICE_INFO_COMPILER_AVAILABLE:
    return ReturnValue(static_cast<ur_bool_t>(true));
  case UR_DEVICE_INFO_LINKER_AVAILABLE:
    return ReturnValue(static_cast<ur_bool_t>(true));
  case UR_DEVICE_INFO_MAX_COMPUTE_UNITS: {
    uint32_t MaxComputeUnits =
        Device->ZeDeviceProperties->numEUsPerSubslice *
        Device->ZeDeviceProperties->numSubslicesPerSlice *
        Device->ZeDeviceProperties->numSlices;

    bool RepresentsCSlice =
        Device->QueueGroup[ur_device_handle_t_::queue_group_info_t::Compute]
            .ZeIndex >= 0;
    if (RepresentsCSlice)
      MaxComputeUnits /= Device->RootDevice->SubDevices.size();

    return ReturnValue(uint32_t{MaxComputeUnits});
  }
  case UR_DEVICE_INFO_NUM_COMPUTE_UNITS: {
    uint32_t NumComputeUnits =
        Device->ZeDeviceProperties->numSubslicesPerSlice *
        Device->ZeDeviceProperties->numSlices;
    return ReturnValue(uint32_t{NumComputeUnits});
  }
  case UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS:
    // Level Zero spec defines only three dimensions
    return ReturnValue(uint32_t{3});
  case UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
    return ReturnValue(
        uint64_t{Device->ZeDeviceComputeProperties->maxTotalGroupSize});
  case UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    struct {
      size_t Arr[3];
    } MaxGroupSize = {{Device->ZeDeviceComputeProperties->maxGroupSizeX,
                       Device->ZeDeviceComputeProperties->maxGroupSizeY,
                       Device->ZeDeviceComputeProperties->maxGroupSizeZ}};
    return ReturnValue(MaxGroupSize);
  }
  case UR_DEVICE_INFO_MAX_WORK_GROUPS_3D: {
    struct {
      size_t Arr[3];
    } MaxGroupCounts = {{Device->ZeDeviceComputeProperties->maxGroupCountX,
                         Device->ZeDeviceComputeProperties->maxGroupCountY,
                         Device->ZeDeviceComputeProperties->maxGroupCountZ}};
    return ReturnValue(MaxGroupCounts);
  }
  case UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY:
    return ReturnValue(uint32_t{Device->ZeDeviceProperties->coreClockRate});
  case UR_DEVICE_INFO_ADDRESS_BITS: {
    // TODO: To confirm with spec.
    return ReturnValue(uint32_t{64});
  }
  case UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE:
    // if the user wishes to allocate large allocations on a system that usually
    // does not allow that allocation size, then we return the max global mem
    // size as the limit.
    if (Device->useRelaxedAllocationLimits()) {
      return ReturnValue(uint64_t{calculateGlobalMemSize(Device)});
    } else {
      return ReturnValue(uint64_t{Device->ZeDeviceProperties->maxMemAllocSize});
    }
  case UR_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    // Support to read physicalSize depends on kernel,
    // so fallback into reading totalSize if physicalSize
    // is not available.
    uint64_t GlobalMemSize = calculateGlobalMemSize(Device);
    return ReturnValue(uint64_t{GlobalMemSize});
  }
  case UR_DEVICE_INFO_LOCAL_MEM_SIZE:
    return ReturnValue(
        uint64_t{Device->ZeDeviceComputeProperties->maxSharedLocalMemory});
  case UR_DEVICE_INFO_IMAGE_SUPPORTED:
    return ReturnValue(Device->ZeDeviceImageProperties->maxImageDims1D > 0);
  case UR_DEVICE_INFO_HOST_UNIFIED_MEMORY:
    return ReturnValue(
        static_cast<ur_bool_t>((Device->ZeDeviceProperties->flags &
                                ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0));
  case UR_DEVICE_INFO_AVAILABLE:
    return ReturnValue(static_cast<ur_bool_t>(ZeDevice ? true : false));
  case UR_DEVICE_INFO_VENDOR:
    // TODO: Level-Zero does not return vendor's name at the moment
    // only the ID.
    return ReturnValue("Intel(R) Corporation");
  case UR_DEVICE_INFO_DRIVER_VERSION:
  case UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION:
    return ReturnValue(Device->Platform->ZeDriverVersion.c_str());
  case UR_DEVICE_INFO_VERSION: {
    // from compute-runtime/shared/source/helpers/hw_ip_version.h
    typedef struct {
      uint32_t revision : 6;
      uint32_t reserved : 8;
      uint32_t release : 8;
      uint32_t architecture : 10;
    } version_components_t;
    typedef struct {
      union {
        uint32_t value;
        version_components_t components;
      };
    } ipVersion_t;
    ipVersion_t IpVersion;
    IpVersion.value = Device->ZeDeviceIpVersionExt->ipVersion;
    std::stringstream S;
    S << IpVersion.components.architecture << "."
      << IpVersion.components.release << "." << IpVersion.components.revision;
    return ReturnValue(S.str().c_str());
  }
  case UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES: {
    auto Res = Device->Platform->populateDeviceCacheIfNeeded();
    if (Res != UR_RESULT_SUCCESS) {
      return Res;
    }
    return ReturnValue((uint32_t)Device->SubDevices.size());
  }
  case UR_DEVICE_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{Device->RefCount.load()});
  case UR_DEVICE_INFO_SUPPORTED_PARTITIONS: {
    // SYCL spec says: if this SYCL device cannot be partitioned into at least
    // two sub devices then the returned vector must be empty.
    auto Res = Device->Platform->populateDeviceCacheIfNeeded();
    if (Res != UR_RESULT_SUCCESS) {
      return Res;
    }

    uint32_t ZeSubDeviceCount = Device->SubDevices.size();
    if (pSize && ZeSubDeviceCount < 2) {
      *pSize = 0;
      return UR_RESULT_SUCCESS;
    }
    bool PartitionedByCSlice = Device->SubDevices[0]->isCCS();

    auto ReturnHelper = [&](auto... Partitions) {
      struct {
        ur_device_partition_t Arr[sizeof...(Partitions)];
      } PartitionProperties = {{Partitions...}};
      return ReturnValue(PartitionProperties);
    };

    if (ExposeCSliceInAffinityPartitioning) {
      if (PartitionedByCSlice)
        return ReturnHelper(UR_DEVICE_PARTITION_BY_CSLICE,
                            UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN);

      else
        return ReturnHelper(UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN);
    } else {
      return ReturnHelper(PartitionedByCSlice
                              ? UR_DEVICE_PARTITION_BY_CSLICE
                              : UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN);
    }
    break;
  }
  case UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN:
    return ReturnValue(ur_device_affinity_domain_flag_t(
        UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA |
        UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE));
  case UR_DEVICE_INFO_PARTITION_TYPE: {
    // For root-device there is no partitioning to report.
    if (Device->SubDeviceCreationProperty == std::nullopt ||
        !Device->isSubDevice()) {
      if (pSize)
        *pSize = 0;
      return UR_RESULT_SUCCESS;
    }

    if (Device->isCCS()) {
      ur_device_partition_property_t cslice{};
      cslice.type = UR_DEVICE_PARTITION_BY_CSLICE;

      return ReturnValue(cslice);
    }

    return ReturnValue(*Device->SubDeviceCreationProperty);
  }
  // Everything under here is not supported yet
  case UR_EXT_DEVICE_INFO_OPENCL_C_VERSION:
    return ReturnValue("");
  case UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC:
    return ReturnValue(static_cast<ur_bool_t>(true));
  case UR_DEVICE_INFO_PRINTF_BUFFER_SIZE:
    return ReturnValue(
        size_t{Device->ZeDeviceModuleProperties->printfBufferSize});
  case UR_DEVICE_INFO_PROFILE:
    return ReturnValue("FULL_PROFILE");
  case UR_DEVICE_INFO_BUILT_IN_KERNELS:
    // TODO: To find out correct value
    return ReturnValue("");
  case UR_DEVICE_INFO_LOW_POWER_EVENTS_EXP:
    return ReturnValue(static_cast<ur_bool_t>(true));
  case UR_DEVICE_INFO_QUEUE_PROPERTIES:
    return ReturnValue(
        ur_queue_flag_t(UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                        UR_QUEUE_FLAG_PROFILING_ENABLE));
  case UR_DEVICE_INFO_EXECUTION_CAPABILITIES:
    return ReturnValue(ur_device_exec_capability_flag_t{
        UR_DEVICE_EXEC_CAPABILITY_FLAG_NATIVE_KERNEL});
  case UR_DEVICE_INFO_ENDIAN_LITTLE:
    return ReturnValue(static_cast<ur_bool_t>(true));
  case UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT:
    return ReturnValue(static_cast<ur_bool_t>(
        Device->ZeDeviceProperties->flags & ZE_DEVICE_PROPERTY_FLAG_ECC));
  case UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION:
    return ReturnValue(
        static_cast<size_t>(Device->ZeDeviceProperties->timerResolution));
  case UR_DEVICE_INFO_LOCAL_MEM_TYPE:
    return ReturnValue(UR_DEVICE_LOCAL_MEM_TYPE_LOCAL);
  case UR_DEVICE_INFO_MAX_CONSTANT_ARGS:
    return ReturnValue(uint32_t{64});
  case UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE:
    return ReturnValue(
        uint64_t{Device->ZeDeviceImageProperties->maxImageBufferSize});
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE:
    return ReturnValue(UR_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE);
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE:
    return ReturnValue(
        // TODO[1.0]: how to query cache line-size?
        uint32_t{1});
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE:
    return ReturnValue(uint64_t{Device->ZeDeviceCacheProperties->cacheSize});
  case UR_DEVICE_INFO_IP_VERSION:
    return ReturnValue(uint32_t{Device->ZeDeviceIpVersionExt->ipVersion});
  case UR_DEVICE_INFO_MAX_PARAMETER_SIZE:
    return ReturnValue(
        size_t{Device->ZeDeviceModuleProperties->maxArgumentsSize});
  case UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN:
    // SYCL/OpenCL spec is vague on what this means exactly, but seems to
    // be for "alignment requirement (in bits) for sub-buffer offsets."
    // An OpenCL implementation returns 8*128, but Level Zero can do just 8,
    // meaning unaligned access for values of types larger than 8 bits.
    return ReturnValue(uint32_t{8});
  case UR_DEVICE_INFO_MAX_SAMPLERS:
    return ReturnValue(uint32_t{Device->ZeDeviceImageProperties->maxSamplers});
  case UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS:
    return ReturnValue(
        uint32_t{Device->ZeDeviceImageProperties->maxReadImageArgs});
  case UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS:
    return ReturnValue(
        uint32_t{Device->ZeDeviceImageProperties->maxWriteImageArgs});
  case UR_DEVICE_INFO_SINGLE_FP_CONFIG: {
    ur_device_fp_capability_flags_t SingleFPValue = 0;
    ze_device_fp_flags_t ZeSingleFPCapabilities =
        Device->ZeDeviceModuleProperties->fp32flags;
    if (ZE_DEVICE_FP_FLAG_DENORM & ZeSingleFPCapabilities) {
      SingleFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_DENORM;
    }
    if (ZE_DEVICE_FP_FLAG_INF_NAN & ZeSingleFPCapabilities) {
      SingleFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST & ZeSingleFPCapabilities) {
      SingleFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO & ZeSingleFPCapabilities) {
      SingleFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_INF & ZeSingleFPCapabilities) {
      SingleFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF;
    }
    if (ZE_DEVICE_FP_FLAG_FMA & ZeSingleFPCapabilities) {
      SingleFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_FMA;
    }
    if (ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT & ZeSingleFPCapabilities) {
      SingleFPValue |=
          UR_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    }
    return ReturnValue(SingleFPValue);
  }
  case UR_DEVICE_INFO_HALF_FP_CONFIG: {
    ur_device_fp_capability_flags_t HalfFPValue = 0;
    ze_device_fp_flags_t ZeHalfFPCapabilities =
        Device->ZeDeviceModuleProperties->fp16flags;
    if (ZE_DEVICE_FP_FLAG_DENORM & ZeHalfFPCapabilities) {
      HalfFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_DENORM;
    }
    if (ZE_DEVICE_FP_FLAG_INF_NAN & ZeHalfFPCapabilities) {
      HalfFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST & ZeHalfFPCapabilities) {
      HalfFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO & ZeHalfFPCapabilities) {
      HalfFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_INF & ZeHalfFPCapabilities) {
      HalfFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF;
    }
    if (ZE_DEVICE_FP_FLAG_FMA & ZeHalfFPCapabilities) {
      HalfFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_FMA;
    }
    if (ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT & ZeHalfFPCapabilities) {
      HalfFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    }
    return ReturnValue(HalfFPValue);
  }
  case UR_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    ur_device_fp_capability_flags_t DoubleFPValue = 0;
    ze_device_fp_flags_t ZeDoubleFPCapabilities =
        Device->ZeDeviceModuleProperties->fp64flags;
    if (ZE_DEVICE_FP_FLAG_DENORM & ZeDoubleFPCapabilities) {
      DoubleFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_DENORM;
    }
    if (ZE_DEVICE_FP_FLAG_INF_NAN & ZeDoubleFPCapabilities) {
      DoubleFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST & ZeDoubleFPCapabilities) {
      DoubleFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO & ZeDoubleFPCapabilities) {
      DoubleFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_INF & ZeDoubleFPCapabilities) {
      DoubleFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF;
    }
    if (ZE_DEVICE_FP_FLAG_FMA & ZeDoubleFPCapabilities) {
      DoubleFPValue |= UR_DEVICE_FP_CAPABILITY_FLAG_FMA;
    }
    if (ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT & ZeDoubleFPCapabilities) {
      DoubleFPValue |=
          UR_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    }
    return ReturnValue(DoubleFPValue);
  }
  case UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
    return ReturnValue(size_t{Device->ZeDeviceImageProperties->maxImageDims2D});
  case UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
    return ReturnValue(size_t{Device->ZeDeviceImageProperties->maxImageDims2D});
  case UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH:
    return ReturnValue(size_t{Device->ZeDeviceImageProperties->maxImageDims3D});
  case UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT:
    return ReturnValue(size_t{Device->ZeDeviceImageProperties->maxImageDims3D});
  case UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH:
    return ReturnValue(size_t{Device->ZeDeviceImageProperties->maxImageDims3D});
  case UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE:
    return ReturnValue(
        size_t{Device->ZeDeviceImageProperties->maxImageBufferSize});
  case UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE:
    return ReturnValue(
        size_t{Device->ZeDeviceImageProperties->maxImageArraySlices});
  // Handle SIMD widths.
  // TODO: can we do better than this?
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 1);
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 2);
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 4);
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 8);
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 4);
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE:
    // Must return 0 for *vector_width_double* if the device does not have fp64.
    if (!(Device->ZeDeviceModuleProperties->flags & ZE_DEVICE_MODULE_FLAG_FP64))
      return ReturnValue(uint32_t{0});
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 8);
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF:
    // Must return 0 for *vector_width_half* if the device does not have fp16.
    if (!(Device->ZeDeviceModuleProperties->flags & ZE_DEVICE_MODULE_FLAG_FP16))
      return ReturnValue(uint32_t{0});
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 2);
  case UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS: {
    // Max_num_sub_Groups = maxTotalGroupSize/min(set of subGroupSizes);
    uint32_t MinSubGroupSize =
        Device->ZeDeviceComputeProperties->subGroupSizes[0];
    for (uint32_t I = 1;
         I < Device->ZeDeviceComputeProperties->numSubGroupSizes; I++) {
      if (MinSubGroupSize > Device->ZeDeviceComputeProperties->subGroupSizes[I])
        MinSubGroupSize = Device->ZeDeviceComputeProperties->subGroupSizes[I];
    }
    return ReturnValue(Device->ZeDeviceComputeProperties->maxTotalGroupSize /
                       MinSubGroupSize);
  }
  case UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {
    // TODO: Not supported yet. Needs to be updated after support is added.
    return ReturnValue(static_cast<ur_bool_t>(false));
  }
  case UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL: {
    return ReturnValue(Device->ZeDeviceComputeProperties->subGroupSizes,
                       Device->ZeDeviceComputeProperties->numSubGroupSizes);
  }
  case UR_DEVICE_INFO_IL_VERSION: {
    // Set to a space separated list of IL version strings of the form
    // <IL_Prefix>_<Major_version>.<Minor_version>.
    // "SPIR-V" is a required IL prefix when cl_khr_il_progam extension is
    // reported.
    uint32_t SpirvVersion =
        Device->ZeDeviceModuleProperties->spirvVersionSupported;
    uint32_t SpirvVersionMajor = ZE_MAJOR_VERSION(SpirvVersion);
    uint32_t SpirvVersionMinor = ZE_MINOR_VERSION(SpirvVersion);

    char SpirvVersionString[50];
    int Len = sprintf(SpirvVersionString, "SPIR-V_%d.%d ", SpirvVersionMajor,
                      SpirvVersionMinor);
    // returned string to contain only len number of characters.
    std::string ILVersion(SpirvVersionString, Len);
    return ReturnValue(ILVersion.c_str());
  }
  case UR_DEVICE_INFO_USM_HOST_SUPPORT:
  case UR_DEVICE_INFO_USM_DEVICE_SUPPORT:
  case UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
  case UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT:
  case UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    auto MapCaps = [](const ze_memory_access_cap_flags_t &ZeCapabilities) {
      ur_device_usm_access_capability_flags_t Capabilities = 0;
      if (ZeCapabilities & ZE_MEMORY_ACCESS_CAP_FLAG_RW)
        Capabilities |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS;
      if (ZeCapabilities & ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC)
        Capabilities |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS;
      if (ZeCapabilities & ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT)
        Capabilities |= UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS;
      if (ZeCapabilities & ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC)
        Capabilities |=
            UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS;
      return Capabilities;
    };
    auto &Props = Device->ZeDeviceMemoryAccessProperties;
    switch (ParamName) {
    case UR_DEVICE_INFO_USM_HOST_SUPPORT:
      return ReturnValue(MapCaps(Props->hostAllocCapabilities));
    case UR_DEVICE_INFO_USM_DEVICE_SUPPORT:
      return ReturnValue(MapCaps(Props->deviceAllocCapabilities));
    case UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
      return ReturnValue(MapCaps(Props->sharedSingleDeviceAllocCapabilities));
    case UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT:
      return ReturnValue(MapCaps(Props->sharedCrossDeviceAllocCapabilities));
    case UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT:
      return ReturnValue(MapCaps(Props->sharedSystemAllocCapabilities));
    default:
      die("urDeviceGetInfo: unexpected ParamName.");
    }
  }

    // intel extensions for GPU information
  case UR_DEVICE_INFO_DEVICE_ID:
    return ReturnValue(uint32_t{Device->ZeDeviceProperties->deviceId});
  case UR_DEVICE_INFO_PCI_ADDRESS: {
    ze_pci_address_ext_t PciAddr{};
    ZeStruct<ze_pci_ext_properties_t> ZeDevicePciProperties;
    ZeDevicePciProperties.address = PciAddr;
    ZE2UR_CALL(zeDevicePciGetPropertiesExt, (ZeDevice, &ZeDevicePciProperties));
    constexpr size_t AddressBufferSize = 13;
    char AddressBuffer[AddressBufferSize];
    std::snprintf(AddressBuffer, AddressBufferSize, "%04x:%02x:%02x.%01x",
                  ZeDevicePciProperties.address.domain,
                  ZeDevicePciProperties.address.bus,
                  ZeDevicePciProperties.address.device,
                  ZeDevicePciProperties.address.function);
    return ReturnValue(AddressBuffer);
  }

  case UR_DEVICE_INFO_GLOBAL_MEM_FREE: {
    bool SysManEnv = getenv_tobool("ZES_ENABLE_SYSMAN", false);
    if ((Device->Platform->ZedeviceToZesDeviceMap.size() == 0) && !SysManEnv) {
      logger::error("SysMan support is unavailable on this system. Please "
                    "check your level zero driver installation.");
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }
    // Calculate the global memory size as the max limit that can be reported as
    // "free" memory for the user to allocate.
    uint64_t GlobalMemSize = calculateGlobalMemSize(Device);
    // Only report device memory which zeMemAllocDevice can allocate from.
    // Currently this is only the one enumerated with ordinal 0.
    uint64_t FreeMemory = 0;
    uint32_t MemCount = 0;

    zes_device_handle_t ZesDevice = Device->ZeDevice;
    struct ur_zes_device_handle_data_t ZesDeviceData = {};
    // If legacy sysman is enabled thru the environment variable, then zesInit
    // will fail, but sysman is still usable so go the legacy route.
    if (!SysManEnv) {
      auto It = Device->Platform->ZedeviceToZesDeviceMap.find(Device->ZeDevice);
      if (It == Device->Platform->ZedeviceToZesDeviceMap.end()) {
        // no matching device
        return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
      } else {
        ZesDeviceData =
            Device->Platform->ZedeviceToZesDeviceMap[Device->ZeDevice];
        ZesDevice = ZesDeviceData.ZesDevice;
      }
    }

    ZE2UR_CALL(zesDeviceEnumMemoryModules, (ZesDevice, &MemCount, nullptr));
    if (MemCount != 0) {
      std::vector<zes_mem_handle_t> ZesMemHandles(MemCount);
      ZE2UR_CALL(zesDeviceEnumMemoryModules,
                 (ZesDevice, &MemCount, ZesMemHandles.data()));
      for (auto &ZesMemHandle : ZesMemHandles) {
        ZesStruct<zes_mem_properties_t> ZesMemProperties;
        ZE2UR_CALL(zesMemoryGetProperties, (ZesMemHandle, &ZesMemProperties));
        // For root-device report memory from all memory modules since that
        // is what totally available in the default implicit scaling mode.
        // For sub-devices only report memory local to them.
        if (SysManEnv) {
          if (!Device->isSubDevice() ||
              Device->ZeDeviceProperties->subdeviceId ==
                  ZesMemProperties.subdeviceId) {

            ZesStruct<zes_mem_state_t> ZesMemState;
            ZE2UR_CALL(zesMemoryGetState, (ZesMemHandle, &ZesMemState));
            FreeMemory += ZesMemState.free;
          }
        } else {
          if (ZesDeviceData.SubDeviceId == ZesMemProperties.subdeviceId ||
              !ZesDeviceData.SubDevice) {
            ZesStruct<zes_mem_state_t> ZesMemState;
            ZE2UR_CALL(zesMemoryGetState, (ZesMemHandle, &ZesMemState));
            FreeMemory += ZesMemState.free;
          }
        }
      }
    }
    if (MemCount > 0) {
      return ReturnValue(std::min(GlobalMemSize, FreeMemory));
    } else {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }
  }
  case UR_DEVICE_INFO_MEMORY_CLOCK_RATE: {
    // If there are not any memory modules then return 0.
    if (Device->ZeDeviceMemoryProperties->first.empty())
      return ReturnValue(uint32_t{0});

    // If there are multiple memory modules on the device then we have to report
    // the value of the slowest memory.
    auto Comp = [](const ze_device_memory_properties_t &A,
                   const ze_device_memory_properties_t &B) -> bool {
      return A.maxClockRate < B.maxClockRate;
    };
    auto MinIt =
        std::min_element(Device->ZeDeviceMemoryProperties->first.begin(),
                         Device->ZeDeviceMemoryProperties->first.end(), Comp);
    return ReturnValue(uint32_t{MinIt->maxClockRate});
  }
  case UR_DEVICE_INFO_MEMORY_BUS_WIDTH: {
    // If there are not any memory modules then return 0.
    if (Device->ZeDeviceMemoryProperties->first.empty())
      return ReturnValue(uint32_t{0});

    // If there are multiple memory modules on the device then we have to report
    // the value of the slowest memory.
    auto Comp = [](const ze_device_memory_properties_t &A,
                   const ze_device_memory_properties_t &B) -> bool {
      return A.maxBusWidth < B.maxBusWidth;
    };
    auto MinIt =
        std::min_element(Device->ZeDeviceMemoryProperties->first.begin(),
                         Device->ZeDeviceMemoryProperties->first.end(), Comp);
    return ReturnValue(uint32_t{MinIt->maxBusWidth});
  }
  case UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES: {
    if (Device->QueueGroup[ur_device_handle_t_::queue_group_info_t::Compute]
            .ZeIndex >= 0)
      // Sub-sub-device represents a particular compute index already.
      return ReturnValue(int32_t{1});

    auto ZeDeviceNumIndices =
        Device->QueueGroup[ur_device_handle_t_::queue_group_info_t::Compute]
            .ZeProperties.numQueues;
    return ReturnValue(int32_t(ZeDeviceNumIndices));
  } break;
  case UR_DEVICE_INFO_GPU_EU_COUNT: {
    if (Device->Platform->ZeDriverEuCountExtensionFound) {
      ze_device_properties_t DeviceProp = {};
      DeviceProp.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      ze_eu_count_ext_t EuCountDesc = {};
      EuCountDesc.stype = ZE_STRUCTURE_TYPE_EU_COUNT_EXT;
      DeviceProp.pNext = (void *)&EuCountDesc;
      ZE2UR_CALL(zeDeviceGetProperties, (ZeDevice, &DeviceProp));
      if (EuCountDesc.numTotalEUs > 0) {
        return ReturnValue(uint32_t{EuCountDesc.numTotalEUs});
      }
    }

    uint32_t count = Device->ZeDeviceProperties->numEUsPerSubslice *
                     Device->ZeDeviceProperties->numSubslicesPerSlice *
                     Device->ZeDeviceProperties->numSlices;
    return ReturnValue(uint32_t{count});
  }
  case UR_DEVICE_INFO_GPU_EU_SLICES: {
    return ReturnValue(uint32_t{Device->ZeDeviceProperties->numSlices});
  }
  case UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
    return ReturnValue(
        uint32_t{Device->ZeDeviceProperties->physicalEUSimdWidth});
  case UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
    return ReturnValue(
        uint32_t{Device->ZeDeviceProperties->numSubslicesPerSlice});
  case UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
    return ReturnValue(uint32_t{Device->ZeDeviceProperties->numEUsPerSubslice});
  case UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
    return ReturnValue(uint32_t{Device->ZeDeviceProperties->numThreadsPerEU});
  case UR_DEVICE_INFO_BFLOAT16: {
    // bfloat16 math functions are not yet supported on Intel GPUs.
    return ReturnValue(ur_bool_t{false});
  }
  case UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES: {
    // There are no explicit restrictions in L0 programming guide, so assume all
    // are supported
    ur_memory_scope_capability_flags_t result =
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM;

    return ReturnValue(result);
  }
  case UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES: {
    // There are no explicit restrictions in L0 programming guide, so assume all
    // are supported
    ur_memory_order_capability_flags_t result =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;

    return ReturnValue(result);
  }
  case UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    // There are no explicit restrictions in L0 programming guide, so assume all
    // are supported
    ur_memory_scope_capability_flags_t result =
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM;

    return ReturnValue(result);
  }

  case UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: {
    ur_memory_order_capability_flags_t capabilities =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
    return ReturnValue(capabilities);
  }
  case UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT:
    return ReturnValue(ur_bool_t{false});
  case UR_DEVICE_INFO_IMAGE_SRGB:
    return ReturnValue(ur_bool_t{false});

  case UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES:
  case UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES: {
    ur_queue_flags_t queue_flags = 0;
    return ReturnValue(queue_flags);
  }
  case UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS: {
    return ReturnValue(static_cast<uint32_t>(
        0)); //__read_write attribute currently undefinde in opencl
  }
  case UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT: {
    return ReturnValue(static_cast<ur_bool_t>(true));
  }
  case UR_DEVICE_INFO_TIMESTAMP_RECORDING_SUPPORT_EXP: {
    return ReturnValue(static_cast<ur_bool_t>(true));
  }
  case UR_DEVICE_INFO_ENQUEUE_NATIVE_COMMAND_SUPPORT_EXP: {
    // L0 doesn't support enqueueing native work through the urNativeEnqueueExp
    return ReturnValue(static_cast<ur_bool_t>(false));
  }

  case UR_DEVICE_INFO_ESIMD_SUPPORT: {
    // ESIMD is only supported by Intel GPUs.
    ur_bool_t result = Device->ZeDeviceProperties->type == ZE_DEVICE_TYPE_GPU &&
                       Device->ZeDeviceProperties->vendorId == 0x8086;
    return ReturnValue(result);
  }

  case UR_DEVICE_INFO_COMPONENT_DEVICES: {
    ze_device_handle_t DevHandle = Device->ZeDevice;
    uint32_t SubDeviceCount = 0;
    // First call to get SubDeviceCount.
    ZE2UR_CALL(zeDeviceGetSubDevices, (DevHandle, &SubDeviceCount, nullptr));
    if (SubDeviceCount == 0)
      return ReturnValue(std::nullopt);

    std::vector<ze_device_handle_t> SubDevs(SubDeviceCount);
    // Second call to get the actual list of devices.
    ZE2UR_CALL(zeDeviceGetSubDevices,
               (DevHandle, &SubDeviceCount, SubDevs.data()));

    size_t SubDeviceCount_s{SubDeviceCount};
    auto ResSize =
        std::min(SubDeviceCount_s, propSize / sizeof(ur_device_handle_t));
    std::vector<ur_device_handle_t> Res;
    for (const auto &d : SubDevs) {
      // We can only reach this code if ZE_FLAT_DEVICE_HIERARCHY != FLAT,
      // because in flat mode we directly get tiles, and those don't have any
      // further divisions, so zeDeviceGetSubDevices always will return an empty
      // list. Thus, there's only two options left: (a) composite mode, and (b)
      // combined mode. In (b), zeDeviceGet returns tiles as devices, and those
      // are presented as root devices (i.e. isSubDevice() returns false). In
      // contrast, in (a), zeDeviceGet returns cards as devices, so tiles are
      // not root devices (i.e. isSubDevice() returns true). Since we only reach
      // this code if there are tiles returned by zeDeviceGetSubDevices, we
      // can know if we are in (a) or (b) by checking if a tile is root device
      // or not.
      ur_device_handle_t URDev = Device->Platform->getDeviceFromNativeHandle(d);
      if (URDev->isSubDevice()) {
        // We are in COMPOSITE mode, return an empty list.
        if (pSize) {
          *pSize = 0;
        }
        return UR_RESULT_SUCCESS;
      }

      Res.push_back(URDev);
    }
    if (pSize)
      *pSize = SubDeviceCount * sizeof(ur_device_handle_t);
    if (ParamValue) {
      return ReturnValue(Res.data(), ResSize);
    }
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_COMPOSITE_DEVICE: {
    ur_device_handle_t UrRootDev = nullptr;
    ze_device_handle_t DevHandle = Device->ZeDevice;
    ze_device_handle_t RootDev;
    // Query Root Device.
    auto errc = ZE_CALL_NOCHECK(zeDeviceGetRootDevice, (DevHandle, &RootDev));
    UrRootDev = Device->Platform->getDeviceFromNativeHandle(RootDev);
    if (errc != ZE_RESULT_SUCCESS &&
        errc != ZE_RESULT_ERROR_UNSUPPORTED_FEATURE)
      return ze2urResult(errc);
    return ReturnValue(UrRootDev);
  }
  case UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP:
    return ReturnValue(true);
  case UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP: {
    const auto ZeMutableCommandFlags =
        Device->ZeDeviceMutableCmdListsProperties->mutableCommandFlags;

    auto supportsFlags = [&](ze_mutable_command_exp_flags_t RequiredFlags) {
      return (ZeMutableCommandFlags & RequiredFlags) == RequiredFlags;
    };

    ur_device_command_buffer_update_capability_flags_t UpdateCapabilities = 0;
    if (supportsFlags(ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS)) {
      UpdateCapabilities |=
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_ARGUMENTS;
    }
    /* These capabilities are bundled together because, when the user updates
     * the global work-size, the implementation might have to generate a new
     * local work-size. This would require both mutable command flags to be set
     * even though only the global work-size was explicitly updated. */
    if (supportsFlags(ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT |
                      ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE)) {
      UpdateCapabilities |=
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_SIZE |
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_LOCAL_WORK_SIZE;
    }
    if (supportsFlags(ZE_MUTABLE_COMMAND_EXP_FLAG_GLOBAL_OFFSET)) {
      UpdateCapabilities |=
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_OFFSET;
    }
    if (supportsFlags(ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_INSTRUCTION)) {
      UpdateCapabilities |=
          UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_HANDLE;
    }
    return ReturnValue(UpdateCapabilities);
  }
  case UR_DEVICE_INFO_COMMAND_BUFFER_EVENT_SUPPORT_EXP:
    return ReturnValue(false);
  case UR_DEVICE_INFO_BINDLESS_IMAGES_SUPPORT_EXP: {
    return ReturnValue(Device->isIntelDG2OrNewer() &&
                       Device->ZeDeviceImageProperties->maxImageDims1D > 0 &&
                       Device->ZeDeviceImageProperties->maxImageDims2D > 0 &&
                       Device->ZeDeviceImageProperties->maxImageDims3D > 0);
  }
  case UR_DEVICE_INFO_BINDLESS_IMAGES_SHARED_USM_SUPPORT_EXP: {
    // On L0 bindless images can not be backed by shared (managed) USM.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_BINDLESS_IMAGES_1D_USM_SUPPORT_EXP: {
    return ReturnValue(Device->isIntelDG2OrNewer() &&
                       Device->ZeDeviceImageProperties->maxImageDims1D > 0);
  }
  case UR_DEVICE_INFO_BINDLESS_IMAGES_2D_USM_SUPPORT_EXP: {
    return ReturnValue(Device->isIntelDG2OrNewer() &&
                       Device->ZeDeviceImageProperties->maxImageDims2D > 0);
  }
  case UR_DEVICE_INFO_IMAGE_PITCH_ALIGN_EXP:
  case UR_DEVICE_INFO_MAX_IMAGE_LINEAR_WIDTH_EXP:
  case UR_DEVICE_INFO_MAX_IMAGE_LINEAR_HEIGHT_EXP:
  case UR_DEVICE_INFO_MAX_IMAGE_LINEAR_PITCH_EXP:
    logger::error("Unsupported ParamName in urGetDeviceInfo");
    logger::error("ParamName=%{}(0x{})", ParamName, logger::toHex(ParamName));
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  case UR_DEVICE_INFO_MIPMAP_SUPPORT_EXP: {
    // L0 does not support mipmaps.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_MIPMAP_ANISOTROPY_SUPPORT_EXP: {
    // L0 does not support anisotropic filtering.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_MIPMAP_MAX_ANISOTROPY_EXP:
    logger::error("Unsupported ParamName in urGetDeviceInfo");
    logger::error("ParamName=%{}(0x{})", ParamName, logger::toHex(ParamName));
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  case UR_DEVICE_INFO_MIPMAP_LEVEL_REFERENCE_SUPPORT_EXP: {
    // L0 does not support creation of images from individual mipmap levels.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_EXTERNAL_MEMORY_IMPORT_SUPPORT_EXP: {
    // L0 does not support importing external memory.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_EXTERNAL_SEMAPHORE_IMPORT_SUPPORT_EXP: {
    // L0 does not support importing external semaphores.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_CUBEMAP_SUPPORT_EXP: {
    // L0 does not support cubemaps.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_CUBEMAP_SEAMLESS_FILTERING_SUPPORT_EXP: {
    // L0 does not support cubemap seamless filtering.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_USM_EXP: {
    // L0 does not support fetching 1D USM sampled image data.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_EXP: {
    // L0 does not not support fetching 1D non-USM sampled image data.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_USM_EXP: {
    // L0 does not support fetching 2D USM sampled image data.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_EXP: {
    // L0 does not support fetching 2D non-USM sampled image data.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_3D_EXP: {
    // L0 does not support fetching 3D non-USM sampled image data.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_IMAGE_ARRAY_SUPPORT_EXP: {
    // L0 does not support image arrays
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_BINDLESS_UNIQUE_ADDRESSING_PER_DIM_EXP: {
    // L0 does not support unique addressing per dimension
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_BINDLESS_SAMPLE_1D_USM_EXP: {
    // L0 does not support sampling 1D USM sampled image data.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_BINDLESS_SAMPLE_2D_USM_EXP: {
    // L0 does not support sampling 1D USM sampled image data.
    return ReturnValue(false);
  }
  case UR_DEVICE_INFO_PROGRAM_SET_SPECIALIZATION_CONSTANTS:
    return ReturnValue(true);
  case UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS:
    return ReturnValue(false);
  case UR_DEVICE_INFO_GLOBAL_VARIABLE_SUPPORT:
    return ReturnValue(true);
  case UR_DEVICE_INFO_USM_POOL_SUPPORT:
    return ReturnValue(true);
  case UR_DEVICE_INFO_2D_BLOCK_ARRAY_CAPABILITIES_EXP: {
#ifdef ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_NAME
    const auto ZeDeviceBlockArrayFlags =
        Device->ZeDeviceBlockArrayProperties->flags;

    auto supportsFlags =
        [&](ze_intel_device_block_array_exp_flags_t RequiredFlags) {
          return (ZeDeviceBlockArrayFlags & RequiredFlags) == RequiredFlags;
        };

    ur_exp_device_2d_block_array_capability_flags_t BlockArrayCapabilities = 0;
    if (supportsFlags(ZE_INTEL_DEVICE_EXP_FLAG_2D_BLOCK_LOAD)) {
      BlockArrayCapabilities |=
          UR_EXP_DEVICE_2D_BLOCK_ARRAY_CAPABILITY_FLAG_LOAD;
    }
    if (supportsFlags(ZE_INTEL_DEVICE_EXP_FLAG_2D_BLOCK_STORE)) {
      BlockArrayCapabilities |=
          UR_EXP_DEVICE_2D_BLOCK_ARRAY_CAPABILITY_FLAG_STORE;
    }
    return ReturnValue(BlockArrayCapabilities);
#else
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
#endif
  }
  case UR_DEVICE_INFO_ASYNC_BARRIER:
    return ReturnValue(false);
  case UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORTED:
    return ReturnValue(false);
  default:
    logger::error("Unsupported ParamName in urGetDeviceInfo");
    logger::error("ParamNameParamName={}(0x{})", ParamName,
                  logger::toHex(ParamName));
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

bool CopyEngineRequested(const ur_device_handle_t &Device) {
  int LowerCopyQueueIndex = getRangeOfAllowedCopyEngines(Device).first;
  int UpperCopyQueueIndex = getRangeOfAllowedCopyEngines(Device).second;
  return ((LowerCopyQueueIndex != -1) || (UpperCopyQueueIndex != -1));
}

ur_result_t urDevicePartition(
    /// [in] handle of the device to partition.
    ur_device_handle_t Device,
    /// [in] Device partition properties.
    const ur_device_partition_properties_t *Properties,
    /// [in] the number of sub-devices.
    uint32_t NumDevices,
    /// [out][optional][range(0, NumDevices)] array of handle of devices. If
    /// NumDevices is less than the number of sub-devices available, then
    /// the function shall only retrieve that number of sub-devices.
    ur_device_handle_t *OutDevices,
    /// [out][optional] pointer to the number of sub-devices the device can be
    /// partitioned into according to the partitioning property.
    uint32_t *NumDevicesRet) {
  // Other partitioning ways are not supported by Level Zero
  UR_ASSERT(Properties->PropCount == 1, UR_RESULT_ERROR_INVALID_VALUE);
  if (Properties->pProperties->type == UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN) {
    if ((Properties->pProperties->value.affinity_domain !=
             UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE &&
         Properties->pProperties->value.affinity_domain !=
             UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA)) {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  } else if (Properties->pProperties->type == UR_DEVICE_PARTITION_BY_CSLICE) {
    if (Properties->pProperties->value.affinity_domain != 0) {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  } else {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  // Devices cache is normally created in piDevicesGet but still make
  // sure that cache is populated.
  //
  auto Res = Device->Platform->populateDeviceCacheIfNeeded();
  if (Res != UR_RESULT_SUCCESS) {
    return Res;
  }

  auto EffectiveNumDevices = [&]() -> decltype(Device->SubDevices.size()) {
    if (Device->SubDevices.size() == 0)
      return 0;

    // Sub-Sub-Devices are partitioned by CSlices, not by affinity domain.
    // However, if
    // UR_L0_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING overrides that
    // still expose CSlices in partitioning by affinity domain for compatibility
    // reasons.
    if (Properties->pProperties->type ==
            UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN &&
        !ExposeCSliceInAffinityPartitioning) {
      if (Device->isSubDevice()) {
        return 0;
      }
    }
    if (Properties->pProperties->type == UR_DEVICE_PARTITION_BY_CSLICE) {
      // Not a CSlice-based partitioning.
      if (!Device->SubDevices[0]->isCCS()) {
        return 0;
      }
    }

    return Device->SubDevices.size();
  }();

  // TODO: Consider support for partitioning to <= total sub-devices.
  // Currently supported partitioning (by affinity domain/numa) would always
  // partition to all sub-devices.
  //
  if (NumDevices != 0)
    UR_ASSERT(NumDevices == EffectiveNumDevices, UR_RESULT_ERROR_INVALID_VALUE);

  for (uint32_t I = 0; I < NumDevices; I++) {
    auto prop = Properties->pProperties[0];
    if (prop.type == UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN) {
      // In case the value is NEXT_PARTITIONABLE, we need to change it to the
      // chosen domain. This will always be NUMA since that's the only domain
      // supported by level zero.
      prop.value.affinity_domain = UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA;
    }
    Device->SubDevices[I]->SubDeviceCreationProperty = prop;

    OutDevices[I] = Device->SubDevices[I];
    // reusing the same pi_device needs to increment the reference count
    ur::level_zero::urDeviceRetain(OutDevices[I]);
  }

  if (NumDevicesRet) {
    *NumDevicesRet = EffectiveNumDevices;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urDeviceSelectBinary(
    /// [in] handle of the device to select binary for.
    ur_device_handle_t Device,
    /// [in] the array of binaries to select from.
    const ur_device_binary_t *Binaries,
    /// [in] the number of binaries passed in ppBinaries. Must greater than or
    /// equal to zero otherwise ::UR_RESULT_ERROR_INVALID_VALUE is returned.
    uint32_t NumBinaries,
    /// [out] the index of the selected binary in the input array of
    /// binaries. If a suitable binary was not found the function returns
    /// ${X}_INVALID_BINARY.
    uint32_t *SelectedBinary) {
  std::ignore = Device;
  // TODO: this is a bare-bones implementation for choosing a device image
  // that would be compatible with the targeted device. An AOT-compiled
  // image is preferred over SPIR-V for known devices (i.e. Intel devices)
  // The implementation makes no effort to differentiate between multiple images
  // for the given device, and simply picks the first one compatible.
  //
  // Real implementation will use the same mechanism OpenCL ICD dispatcher
  // uses. Something like:
  //   PI_VALIDATE_HANDLE_RETURN_HANDLE(ctx, PI_ERROR_INVALID_CONTEXT);
  //     return context->dispatch->piextDeviceSelectIR(
  //       ctx, images, num_images, selected_image);
  // where context->dispatch is set to the dispatch table provided by PI
  // plugin for platform/device the ctx was created for.

  // Look for GEN binary, which we known can only be handled by Level-Zero now.
  const char *BinaryTarget =
      UR_DEVICE_BINARY_TARGET_SPIRV64_GEN; // UR_DEVICE_BINARY_TARGET_SPIRV64_GEN;

  uint32_t *SelectedBinaryInd = SelectedBinary;

  // Find the appropriate device image, fallback to spirv if not found
  constexpr uint32_t InvalidInd = (std::numeric_limits<uint32_t>::max)();
  uint32_t Spirv = InvalidInd;

  for (uint32_t i = 0; i < NumBinaries; ++i) {
    if (strcmp(Binaries[i].pDeviceTargetSpec, BinaryTarget) == 0) {
      *SelectedBinaryInd = i;
      return UR_RESULT_SUCCESS;
    }
    if (strcmp(Binaries[i].pDeviceTargetSpec,
               UR_DEVICE_BINARY_TARGET_SPIRV64) == 0)
      Spirv = i;
  }
  // Points to a spirv image, if such indeed was found
  if ((*SelectedBinaryInd = Spirv) != InvalidInd)
    return UR_RESULT_SUCCESS;

  // No image can be loaded for the given device
  return UR_RESULT_ERROR_INVALID_BINARY;
}

ur_result_t urDeviceGetNativeHandle(
    /// [in] handle of the device.
    ur_device_handle_t Device,
    /// [out] a pointer to the native handle of the device.
    ur_native_handle_t *NativeDevice) {
  *NativeDevice = reinterpret_cast<ur_native_handle_t>(Device->ZeDevice);
  return UR_RESULT_SUCCESS;
}

ur_result_t urDeviceCreateWithNativeHandle(
    /// [in] the native handle of the device.
    ur_native_handle_t NativeDevice,
    /// [in] handle of the platform instance
    [[maybe_unused]] ur_adapter_handle_t Adapter,
    /// [in][optional] pointer to native device properties struct.
    [[maybe_unused]] const ur_device_native_properties_t *Properties,
    /// [out] pointer to the handle of the device object created.
    ur_device_handle_t *Device) {
  auto ZeDevice = ur_cast<ze_device_handle_t>(NativeDevice);

  // The SYCL spec requires that the set of devices must remain fixed for the
  // duration of the application's execution. We assume that we found all of the
  // Level Zero devices when we initialized the platforms/devices cache, so the
  // "NativeHandle" must already be in the cache. If it is not, this must not be
  // a valid Level Zero device.

  ur_device_handle_t Dev = nullptr;
  if (const auto *platforms = GlobalAdapter->PlatformCache->get_value()) {
    for (const auto &p : *platforms) {
      Dev = p->getDeviceFromNativeHandle(ZeDevice);
    }
  } else {
    return GlobalAdapter->PlatformCache->get_error();
  }

  if (Dev == nullptr)
    return UR_RESULT_ERROR_INVALID_VALUE;

  *Device = Dev;
  return UR_RESULT_SUCCESS;
}

ur_result_t urDeviceGetGlobalTimestamps(
    /// [in] handle of the device instance
    ur_device_handle_t Device,
    /// [out][optional] pointer to the Device's global timestamp that correlates
    /// with the Host's global timestamp value
    uint64_t *DeviceTimestamp,
    /// [out][optional] pointer to the Host's global timestamp that correlates
    /// with the Device's global timestamp value
    uint64_t *HostTimestamp) {
  const uint64_t &ZeTimerResolution =
      Device->ZeDeviceProperties->timerResolution;
  const uint64_t TimestampMaxCount = Device->getTimestampMask();
  uint64_t DeviceClockCount, Dummy;

  ZE2UR_CALL(zeDeviceGetGlobalTimestamps,
             (Device->ZeDevice,
              HostTimestamp == nullptr ? &Dummy : HostTimestamp,
              &DeviceClockCount));

  if (DeviceTimestamp != nullptr) {
    *DeviceTimestamp =
        (DeviceClockCount & TimestampMaxCount) * ZeTimerResolution;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urDeviceRetain(ur_device_handle_t Device) {
  // The root-device ref-count remains unchanged (always 1).
  if (Device->isSubDevice()) {
    Device->RefCount.increment();
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urDeviceRelease(ur_device_handle_t Device) {
  // Root devices are destroyed during the piTearDown process.
  if (Device->isSubDevice()) {
    if (Device->RefCount.decrementAndTest()) {
      delete Device;
    }
  }

  return UR_RESULT_SUCCESS;
}
} // namespace ur::level_zero

/**
 * @brief Determines the mode of immediate command lists to be used.
 *
 * This function checks environment variables and device properties to decide
 * the mode of immediate command lists. The mode can be influenced by the
 * following environment variables:
 * - `UR_L0_USE_IMMEDIATE_COMMANDLISTS`
 * - `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`
 *
 * If neither environment variable is set, the function defaults to using the
 * device's properties to determine the mode.
 *
 * @return The mode of immediate command lists, which can be one of the
 * following:
 * - `NotUsed`: Immediate command lists are not used.
 * - `PerQueue`: Immediate command lists are used per queue.
 * - `PerThreadPerQueue`: Immediate command lists are used per thread per queue.
 *
 * The decision process is as follows:
 * 1. If the environment variables are not set, the function checks if the
 * device is Intel DG2 or newer and if the driver version is supported. If both
 *    conditions are met, or if the device is PVC, it returns `PerQueue`.
 *    Otherwise, it returns `NotUsed`.
 * 2. If the environment variable is set, it returns the corresponding mode:
 *    - `0`: `NotUsed`
 *    - `1`: `PerQueue`
 *    - `2`: `PerThreadPerQueue`
 *    - Any other value: `NotUsed`
 */
ur_device_handle_t_::ImmCmdlistMode
ur_device_handle_t_::useImmediateCommandLists() {
  // If immediate commandlist setting is not explicitly set, then use the device
  // default.
  // TODO: confirm this is good once make_queue revert is added
  static const int ImmediateCommandlistsSetting = [] {
    const char *UrRet = std::getenv("UR_L0_USE_IMMEDIATE_COMMANDLISTS");
    const char *PiRet =
        std::getenv("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS");
    const char *ImmediateCommandlistsSettingStr =
        UrRet ? UrRet : (PiRet ? PiRet : nullptr);
    if (!ImmediateCommandlistsSettingStr)
      return -1;
    return std::atoi(ImmediateCommandlistsSettingStr);
  }();

  if (ImmediateCommandlistsSetting == -1) {
    bool isDG2OrNewer = this->isIntelDG2OrNewer();
    bool isDG2SupportedDriver =
        this->Platform->isDriverVersionNewerOrSimilar(1, 5, 30820);
    if ((isDG2SupportedDriver && isDG2OrNewer) || isPVC()) {
      return PerQueue;
    } else {
      return NotUsed;
    }
  }
  switch (ImmediateCommandlistsSetting) {
  case 0:
    return NotUsed;
  case 1:
    return PerQueue;
  case 2:
    return PerThreadPerQueue;
  default:
    return NotUsed;
  }
}

bool ur_device_handle_t_::useRelaxedAllocationLimits() {
  static const bool EnableRelaxedAllocationLimits = [] {
    auto UrRet = ur_getenv("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS");
    const bool RetVal = UrRet ? std::stoi(*UrRet) : 0;
    return RetVal;
  }();

  return EnableRelaxedAllocationLimits;
}

bool ur_device_handle_t_::useDriverInOrderLists() {
  // Use in-order lists implementation from L0 driver instead
  // of adapter's implementation.

  static const bool UseDriverInOrderLists = [&] {
    const char *UrRet = std::getenv("UR_L0_USE_DRIVER_INORDER_LISTS");
    // bool CompatibleDriver = this->Platform->isDriverVersionNewerOrSimilar(
    //     1, 3, L0_DRIVER_INORDER_MIN_VERSION);
    if (!UrRet)
      return false;
    return std::atoi(UrRet) != 0;
  }();

  return UseDriverInOrderLists;
}

bool ur_device_handle_t_::useDriverCounterBasedEvents() {
  // Use counter-based events implementation from L0 driver.

  static const bool DriverCounterBasedEventsEnabled = [] {
    const char *UrRet = std::getenv("UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS");
    if (!UrRet) {
      return true;
    }
    return std::atoi(UrRet) != 0;
  }();

  return DriverCounterBasedEventsEnabled;
}

ur_result_t ur_device_handle_t_::initialize(int SubSubDeviceOrdinal,
                                            int SubSubDeviceIndex) {
  // Maintain various device properties cache.
  // Note that we just describe here how to compute the data.
  // The real initialization is upon first access.
  //
  auto ZeDevice = this->ZeDevice;
  ZeDeviceProperties.Compute = [ZeDevice](ze_device_properties_t &Properties) {
    ZE_CALL_NOCHECK(zeDeviceGetProperties, (ZeDevice, &Properties));
  };

  ZeDeviceComputeProperties.Compute =
      [ZeDevice](ze_device_compute_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeDeviceGetComputeProperties, (ZeDevice, &Properties));
      };

  ZeDeviceIpVersionExt.Compute =
      [ZeDevice](ze_device_ip_version_ext_t &Properties) {
        ze_device_properties_t P;
        P.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        P.pNext = (void *)&Properties;
        ZE_CALL_NOCHECK(zeDeviceGetProperties, (ZeDevice, &P));
      };

  ZeDeviceImageProperties.Compute =
      [ZeDevice](ze_device_image_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeDeviceGetImageProperties, (ZeDevice, &Properties));
      };

  ZeDeviceModuleProperties.Compute =
      [ZeDevice](ze_device_module_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeDeviceGetModuleProperties, (ZeDevice, &Properties));
      };

  ZeDeviceMemoryProperties.Compute =
      [ZeDevice](
          std::pair<std::vector<ZeStruct<ze_device_memory_properties_t>>,
                    std::vector<ZeStruct<ze_device_memory_ext_properties_t>>>
              &Properties) {
        uint32_t Count = 0;
        ZE_CALL_NOCHECK(zeDeviceGetMemoryProperties,
                        (ZeDevice, &Count, nullptr));

        auto &PropertiesVector = Properties.first;
        auto &PropertiesExtVector = Properties.second;

        PropertiesVector.resize(Count);
        PropertiesExtVector.resize(Count);
        // Request for extended memory properties be read in
        for (uint32_t I = 0; I < Count; ++I)
          PropertiesVector[I].pNext = (void *)&PropertiesExtVector[I];

        ZE_CALL_NOCHECK(zeDeviceGetMemoryProperties,
                        (ZeDevice, &Count, PropertiesVector.data()));
      };

  ZeDeviceMemoryAccessProperties.Compute =
      [ZeDevice](ze_device_memory_access_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeDeviceGetMemoryAccessProperties,
                        (ZeDevice, &Properties));
      };

  ZeDeviceCacheProperties.Compute =
      [ZeDevice](ze_device_cache_properties_t &Properties) {
        // TODO: Since v1.0 there can be multiple cache properties.
        // For now remember the first one, if any.
        uint32_t Count = 0;
        ZE_CALL_NOCHECK(zeDeviceGetCacheProperties,
                        (ZeDevice, &Count, nullptr));
        if (Count > 0)
          Count = 1;
        ZE_CALL_NOCHECK(zeDeviceGetCacheProperties,
                        (ZeDevice, &Count, &Properties));
      };

  ZeDeviceMutableCmdListsProperties.Compute =
      [ZeDevice](
          ZeStruct<ze_mutable_command_list_exp_properties_t> &Properties) {
        ze_device_properties_t P;
        P.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        P.pNext = &Properties;
        ZE_CALL_NOCHECK(zeDeviceGetProperties, (ZeDevice, &P));
      };

#ifdef ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_NAME
  ZeDeviceBlockArrayProperties.Compute =
      [ZeDevice](
          ZeStruct<ze_intel_device_block_array_exp_properties_t> &Properties) {
        ze_device_properties_t P;
        P.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        P.pNext = &Properties;
        ZE_CALL_NOCHECK(zeDeviceGetProperties, (ZeDevice, &P));
      };
#endif // ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_NAME

  ImmCommandListUsed = this->useImmediateCommandLists();

  uint32_t numQueueGroups = 0;
  ZE2UR_CALL(zeDeviceGetCommandQueueGroupProperties,
             (ZeDevice, &numQueueGroups, nullptr));
  if (numQueueGroups == 0) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  logger::info(logger::LegacyMessage("NOTE: Number of queue groups = {}"),
               "Number of queue groups = {}", numQueueGroups);
  std::vector<ZeStruct<ze_command_queue_group_properties_t>>
      QueueGroupProperties(numQueueGroups);
  ZE2UR_CALL(zeDeviceGetCommandQueueGroupProperties,
             (ZeDevice, &numQueueGroups, QueueGroupProperties.data()));

  // Initialize ordinal and compute queue group properties
  for (uint32_t i = 0; i < numQueueGroups; i++) {
    if (QueueGroupProperties[i].flags &
        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      QueueGroup[ur_device_handle_t_::queue_group_info_t::Compute].ZeOrdinal =
          i;
      QueueGroup[ur_device_handle_t_::queue_group_info_t::Compute]
          .ZeProperties = QueueGroupProperties[i];
      break;
    }
  }

  // Reinitialize a sub-sub-device with its own ordinal, index.
  // Our sub-sub-device representation is currently [Level-Zero sub-device
  // handle + Level-Zero compute group/engine index]. Only the specified
  // index queue will be used to submit work to the sub-sub-device.
  if (SubSubDeviceOrdinal >= 0) {
    QueueGroup[ur_device_handle_t_::queue_group_info_t::Compute].ZeOrdinal =
        SubSubDeviceOrdinal;
    QueueGroup[ur_device_handle_t_::queue_group_info_t::Compute].ZeIndex =
        SubSubDeviceIndex;
  } else { // Proceed with initialization for root and sub-device
           // How is it possible that there are no "compute" capabilities?
    if (QueueGroup[ur_device_handle_t_::queue_group_info_t::Compute].ZeOrdinal <
        0) {
      return UR_RESULT_ERROR_UNKNOWN;
    }

    if (ur::level_zero::CopyEngineRequested((ur_device_handle_t)this)) {
      for (uint32_t i = 0; i < numQueueGroups; i++) {
        if (((QueueGroupProperties[i].flags &
              ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) == 0) &&
            (QueueGroupProperties[i].flags &
             ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)) {
          if (QueueGroupProperties[i].numQueues == 1) {
            QueueGroup[queue_group_info_t::MainCopy].ZeOrdinal = i;
            QueueGroup[queue_group_info_t::MainCopy].ZeProperties =
                QueueGroupProperties[i];
          } else {
            QueueGroup[queue_group_info_t::LinkCopy].ZeOrdinal = i;
            QueueGroup[queue_group_info_t::LinkCopy].ZeProperties =
                QueueGroupProperties[i];
            break;
          }
        }
      }
      if (QueueGroup[queue_group_info_t::MainCopy].ZeOrdinal < 0)
        logger::info(logger::LegacyMessage(
                         "NOTE: main blitter/copy engine is not available"),
                     "main blitter/copy engine is not available");
      else
        logger::info(logger::LegacyMessage(
                         "NOTE: main blitter/copy engine is available"),
                     "main blitter/copy engine is available");

      if (QueueGroup[queue_group_info_t::LinkCopy].ZeOrdinal < 0)
        logger::info(logger::LegacyMessage(
                         "NOTE: link blitter/copy engines are not available"),
                     "link blitter/copy engines are not available");
      else
        logger::info(logger::LegacyMessage(
                         "NOTE: link blitter/copy engines are available"),
                     "link blitter/copy engines are available");
    }
  }

  return UR_RESULT_SUCCESS;
}

void ZeDriverVersionStringExtension::setZeDriverVersionString(
    ur_platform_handle_t_ *Platform) {
  // Check if Intel Driver Version String is available. If yes, save the API
  // pointer. The pointer will be used when reading the Driver Version for
  // users.
  ze_driver_handle_t DriverHandle = Platform->ZeDriver;
  if (auto extension = Platform->zeDriverExtensionMap.find(
          "ZE_intel_get_driver_version_string");
      extension != Platform->zeDriverExtensionMap.end()) {
    if (ZE_CALL_NOCHECK(zeDriverGetExtensionFunctionAddress,
                        (DriverHandle, "zeIntelGetDriverVersionString",
                         reinterpret_cast<void **>(
                             &zeIntelGetDriverVersionStringPointer))) == 0) {
      // Intel Driver Version String is Supported by this Driver.
      Supported = true;
    }
  }
}

void ZeDriverVersionStringExtension::getDriverVersionString(
    ze_driver_handle_t DriverHandle, char *pDriverVersion,
    size_t *pVersionSize) {
  ZE_CALL_NOCHECK(zeIntelGetDriverVersionStringPointer,
                  (DriverHandle, pDriverVersion, pVersionSize));
}

void ZeUSMImportExtension::setZeUSMImport(ur_platform_handle_t_ *Platform) {
  // Check if USM hostptr import feature is available. If yes, save the API
  // pointers. The pointers will be used for both import/release of SYCL buffer
  // host ptr and the SYCL experimental APIs, prepare_for_device_copy and
  // release_from_device_copy.
  ze_driver_handle_t DriverHandle = Platform->ZeDriver;
  if (ZE_CALL_NOCHECK(
          zeDriverGetExtensionFunctionAddress,
          (DriverHandle, "zexDriverImportExternalPointer",
           reinterpret_cast<void **>(&zexDriverImportExternalPointer))) == 0) {
    ZE_CALL_NOCHECK(
        zeDriverGetExtensionFunctionAddress,
        (DriverHandle, "zexDriverReleaseImportedPointer",
         reinterpret_cast<void **>(&zexDriverReleaseImportedPointer)));
    // Hostptr import/release is supported by this platform.
    Supported = true;

    // Check if env var SYCL_USM_HOSTPTR_IMPORT has been set requesting
    // host ptr import during buffer creation.
    const char *USMHostPtrImportStr = std::getenv("SYCL_USM_HOSTPTR_IMPORT");
    if (!USMHostPtrImportStr || std::atoi(USMHostPtrImportStr) == 0)
      return;

    // Hostptr import/release is turned on because it has been requested
    // by the env var, and this platform supports the APIs.
    Enabled = true;
    // Hostptr import is only possible if piMemBufferCreate receives a
    // hostptr as an argument. The SYCL runtime passes a host ptr
    // only when SYCL_HOST_UNIFIED_MEMORY is enabled. Therefore we turn it on.
    setEnvVar("SYCL_HOST_UNIFIED_MEMORY", "1");
  }
}
void ZeUSMImportExtension::doZeUSMImport(ze_driver_handle_t DriverHandle,
                                         void *HostPtr, size_t Size) {
  ZE_CALL_NOCHECK(zexDriverImportExternalPointer,
                  (DriverHandle, HostPtr, Size));
}
void ZeUSMImportExtension::doZeUSMRelease(ze_driver_handle_t DriverHandle,
                                          void *HostPtr) {
  ZE_CALL_NOCHECK(zexDriverReleaseImportedPointer, (DriverHandle, HostPtr));
}
