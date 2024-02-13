//===--------- device.cpp - NATIVE CPU Adapter ----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur_api.h>

#include "platform.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGet(ur_platform_handle_t hPlatform,
                                                ur_device_type_t DeviceType,
                                                uint32_t NumEntries,
                                                ur_device_handle_t *phDevices,
                                                uint32_t *pNumDevices) {
  UR_ASSERT(hPlatform, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  const bool AskingForAll = DeviceType == UR_DEVICE_TYPE_ALL;
  const bool AskingForDefault = DeviceType == UR_DEVICE_TYPE_DEFAULT;
  const bool AskingForCPU = DeviceType == UR_DEVICE_TYPE_CPU;
  const bool ValidDeviceType = AskingForDefault || AskingForAll || AskingForCPU;
  uint32_t DeviceCount = ValidDeviceType ? 1 : 0;

  if (pNumDevices) {
    *pNumDevices = DeviceCount;
  }

  if (NumEntries == 0) {
    /// Runtime queries number of devices
    if (phDevices != nullptr) {
      if (PrintTrace) {
        std::cerr << "Invalid Arguments for urDevicesGet\n";
      }
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    return UR_RESULT_SUCCESS;
  }

  if (DeviceCount == 0) {
    /// No CPU entry to fill 'Device' array
    return UR_RESULT_SUCCESS;
  }

  if (phDevices) {
    phDevices[0] = &hPlatform->TheDevice;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetInfo(ur_device_handle_t hDevice,
                                                    ur_device_info_t propName,
                                                    size_t propSize,
                                                    void *pPropValue,
                                                    size_t *pPropSizeRet) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (static_cast<uint32_t>(propName)) {
  case UR_DEVICE_INFO_TYPE:
    return ReturnValue(UR_DEVICE_TYPE_CPU);
  case UR_DEVICE_INFO_PARENT_DEVICE:
    return ReturnValue(nullptr);
  case UR_DEVICE_INFO_PLATFORM:
    return ReturnValue(hDevice->Platform);
  case UR_DEVICE_INFO_NAME:
    return ReturnValue("SYCL Native CPU");
  case UR_DEVICE_INFO_IMAGE_SUPPORTED:
    return ReturnValue(bool{false});
  case UR_DEVICE_INFO_DRIVER_VERSION:
    return ReturnValue("0.0.0");
  case UR_DEVICE_INFO_VENDOR:
    return ReturnValue("Intel(R) Corporation");
  case UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION:
    // TODO : CHECK
    return ReturnValue("0.0.0");
  case UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
    return ReturnValue(size_t{8192});
  case UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
    return ReturnValue(size_t{8192});
  case UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE:
    return ReturnValue(size_t(65536 /*todo: min if aspect::image*/));
  case UR_DEVICE_INFO_MAX_SAMPLERS:
    return ReturnValue(uint32_t{16 /*todo: min if aspect::image*/});
  case UR_DEVICE_INFO_HOST_UNIFIED_MEMORY:
    return ReturnValue(bool{1});
  case UR_DEVICE_INFO_EXTENSIONS:
    // TODO : Populate return string accordingly - e.g. cl_khr_fp16,
    // cl_khr_fp64, cl_khr_int64_base_atomics,
    // cl_khr_int64_extended_atomics
    return ReturnValue("cl_khr_fp64 ");
  case UR_DEVICE_INFO_VERSION:
    return ReturnValue("0.1");
  case UR_DEVICE_INFO_COMPILER_AVAILABLE:
    return ReturnValue(bool{false});
  case UR_DEVICE_INFO_LINKER_AVAILABLE:
    return ReturnValue(bool{false});
  case UR_DEVICE_INFO_MAX_COMPUTE_UNITS:
    return ReturnValue(uint32_t{256});
  case UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES:
    return ReturnValue(uint32_t{0});
  case UR_DEVICE_INFO_SUPPORTED_PARTITIONS:
    // TODO: Ensure this property is tested correctly
    // Taken from CUDA ur adapter
    if (pPropSizeRet) {
      *pPropSizeRet = 0;
    }
    return UR_RESULT_SUCCESS;
  case UR_DEVICE_INFO_VENDOR_ID:
    // '0x8086' : 'Intel HD graphics vendor ID'
    return ReturnValue(uint32_t{0x8086});
  case UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
    return ReturnValue(size_t{256});
  case UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN:
    // Imported from level_zero
    return ReturnValue(uint32_t{8});
  case UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH:
  case UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT:
  case UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH:
    // Default minimum values required by the SYCL specification.
    return ReturnValue(size_t{2048});
  case UR_DEVICE_INFO_HALF_FP_CONFIG: {
    // todo:
    ur_device_fp_capability_flags_t HalfFPValue = 0;
    return ReturnValue(HalfFPValue);
  }
  case UR_DEVICE_INFO_SINGLE_FP_CONFIG: {
    // todo
    ur_device_fp_capability_flags_t SingleFPValue = 0;
    return ReturnValue(SingleFPValue);
  }
  case UR_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    ur_device_fp_capability_flags_t DoubleFPValue = 0;
    return ReturnValue(DoubleFPValue);
  }
  case UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS:
    return ReturnValue(uint32_t{3});
  case UR_DEVICE_INFO_PARTITION_TYPE:
    return ReturnValue(ur_device_partition_property_t{});
  case UR_EXT_DEVICE_INFO_OPENCL_C_VERSION:
    return ReturnValue("");
  case UR_DEVICE_INFO_QUEUE_PROPERTIES:
    return ReturnValue(
        ur_queue_flag_t(UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                        UR_QUEUE_FLAG_PROFILING_ENABLE));
  case UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    struct {
      size_t Arr[3];
    } MaxGroupSize = {{256, 256, 1}};
    return ReturnValue(MaxGroupSize);
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE:
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF:
    return ReturnValue(uint32_t{1});

  // Imported from level_zero
  case UR_DEVICE_INFO_USM_HOST_SUPPORT:
  case UR_DEVICE_INFO_USM_DEVICE_SUPPORT:
  case UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
  case UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT:
  case UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    uint64_t Supported = 0;
    // TODO[1.0]: how to query for USM support now?
    if (true) {
      // TODO: Use ze_memory_access_capabilities_t
      Supported = UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS |
                  UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS |
                  UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS |
                  UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS;
    }
    return ReturnValue(Supported);
  }
  case UR_DEVICE_INFO_ADDRESS_BITS:
    return ReturnValue(
        uint32_t{sizeof(void *) * std::numeric_limits<unsigned char>::digits});
  case UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY:
    return ReturnValue(uint32_t{1000});
  case UR_DEVICE_INFO_ENDIAN_LITTLE:
    return ReturnValue(bool{true});
  case UR_DEVICE_INFO_AVAILABLE:
    return ReturnValue(bool{true});
  case UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS:
  case UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS:
    /// TODO : Check
    return ReturnValue(uint32_t{0});
  case UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE:
    /// TODO : Check
    return ReturnValue(size_t{0});
  case UR_DEVICE_INFO_MAX_PARAMETER_SIZE:
    /// TODO : Check
    return ReturnValue(size_t{32});
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE:
    return ReturnValue(UR_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE);
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE:
    // TODO : CHECK
    return ReturnValue(uint32_t{64});
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE:
    // TODO : CHECK
    return ReturnValue(uint64_t{0});
  case UR_DEVICE_INFO_GLOBAL_MEM_SIZE:
    // TODO : CHECK
    return ReturnValue(uint64_t{0});
  case UR_DEVICE_INFO_LOCAL_MEM_SIZE:
    // TODO : CHECK
    return ReturnValue(uint64_t{0});
  case UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE:
    // TODO : CHECK
    return ReturnValue(uint64_t{0});
  case UR_DEVICE_INFO_MAX_CONSTANT_ARGS:
    // TODO : CHECK
    return ReturnValue(uint32_t{64});
  case UR_DEVICE_INFO_LOCAL_MEM_TYPE:
    // TODO : CHECK
    return ReturnValue(UR_DEVICE_LOCAL_MEM_TYPE_LOCAL);
  case UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT:
    return ReturnValue(bool{false});
  case UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION:
    // TODO : CHECK
    return ReturnValue(size_t{0});
  case UR_DEVICE_INFO_BUILT_IN_KERNELS:
    // TODO : CHECK
    return ReturnValue("");
  case UR_DEVICE_INFO_PRINTF_BUFFER_SIZE:
    // TODO : CHECK
    return ReturnValue(size_t{1024});
  case UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC:
    return ReturnValue(bool{false});
  case UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN:
    return ReturnValue(ur_device_affinity_domain_flags_t{0});
  case UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE:
    // TODO : CHECK
    return ReturnValue(uint64_t{0});
  case UR_DEVICE_INFO_EXECUTION_CAPABILITIES:
    // TODO : CHECK
    return ReturnValue(ur_device_exec_capability_flags_t{
        UR_DEVICE_EXEC_CAPABILITY_FLAG_KERNEL});
  case UR_DEVICE_INFO_PROFILE:
    return ReturnValue("FULL_PROFILE");
  case UR_DEVICE_INFO_REFERENCE_COUNT:
    // TODO : CHECK
    return ReturnValue(uint32_t{0});
  case UR_DEVICE_INFO_BUILD_ON_SUBDEVICE:
    return ReturnValue(bool{0});
  case UR_DEVICE_INFO_ATOMIC_64:
    return ReturnValue(bool{0});
  case UR_DEVICE_INFO_BFLOAT16:
    return ReturnValue(bool{0});
  case UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT:
    return ReturnValue(bool{0});
  case UR_DEVICE_INFO_IMAGE_SRGB:
    return ReturnValue(bool{0});
  case UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL:
    return ReturnValue(uint32_t{1});
  case UR_DEVICE_INFO_GPU_EU_COUNT:
  case UR_DEVICE_INFO_PCI_ADDRESS:
  case UR_DEVICE_INFO_GPU_EU_SLICES:
  case UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
  case UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
  case UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
  case UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
  case UR_DEVICE_INFO_UUID:
  case UR_DEVICE_INFO_DEVICE_ID:
  case UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS:
  case UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS:
  case UR_DEVICE_INFO_IL_VERSION:
  case UR_DEVICE_INFO_MAX_WORK_GROUPS_3D:
  case UR_DEVICE_INFO_MEMORY_CLOCK_RATE:
  case UR_DEVICE_INFO_MEMORY_BUS_WIDTH:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: {
    ur_memory_order_capability_flags_t Capabilities =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL;
    return ReturnValue(Capabilities);
  }
  case UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES: {
    uint64_t Capabilities = UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
                            UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
                            UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP |
                            UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE;
    return ReturnValue(Capabilities);
  }
  case UR_DEVICE_INFO_ESIMD_SUPPORT:
    return ReturnValue(false);
  case UR_DEVICE_INFO_COMPONENT_DEVICES:
  case UR_DEVICE_INFO_COMPOSITE_DEVICE:
    // These two are exclusive of L0.
    return ReturnValue(0);

    CASE_UR_UNSUPPORTED(UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH);
  case UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT:
    return ReturnValue(false);

  case UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP:
  case UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_SUPPORT_EXP:
    return ReturnValue(false);

  default:
    DIE_NO_IMPLEMENTATION;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceRetain(ur_device_handle_t hDevice) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE)

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urDeviceRelease(ur_device_handle_t hDevice) {
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE)

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDevicePartition(
    ur_device_handle_t hDevice,
    const ur_device_partition_properties_t *pProperties, uint32_t NumDevices,
    ur_device_handle_t *phSubDevices, uint32_t *pNumDevicesRet) {
  std::ignore = hDevice;
  std::ignore = NumDevices;
  std::ignore = pProperties;
  std::ignore = phSubDevices;
  std::ignore = pNumDevicesRet;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetNativeHandle(
    ur_device_handle_t hDevice, ur_native_handle_t *phNativeDevice) {
  std::ignore = hDevice;
  std::ignore = phNativeDevice;

  DIE_NO_IMPLEMENTATION
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceCreateWithNativeHandle(
    ur_native_handle_t hNativeDevice, ur_platform_handle_t hPlatform,
    const ur_device_native_properties_t *pProperties,
    ur_device_handle_t *phDevice) {
  std::ignore = hNativeDevice;
  std::ignore = hPlatform;
  std::ignore = pProperties;
  std::ignore = phDevice;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetGlobalTimestamps(
    ur_device_handle_t hDevice, uint64_t *pDeviceTimestamp,
    uint64_t *pHostTimestamp) {
  std::ignore = hDevice; // todo
  if (pHostTimestamp) {
    using namespace std::chrono;
    *pHostTimestamp =
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
            .count();
  }
  if (pDeviceTimestamp) {
    // todo: calculate elapsed time properly
    using namespace std::chrono;
    *pDeviceTimestamp =
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
            .count();
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceSelectBinary(
    ur_device_handle_t hDevice, const ur_device_binary_t *pBinaries,
    uint32_t NumBinaries, uint32_t *pSelectedBinary) {
  std::ignore = hDevice;
  std::ignore = pBinaries;
  std::ignore = NumBinaries;
  std::ignore = pSelectedBinary;

#define UR_DEVICE_BINARY_TARGET_NATIVE_CPU "native_cpu"
  // look for a binary with type "native_cpu"
  // Todo: error checking
  // Todo: define UR_DEVICE_BINARY_TARGET_NATIVE_CPU in upstream
  const char *image_target = UR_DEVICE_BINARY_TARGET_NATIVE_CPU;
  for (uint32_t i = 0; i < NumBinaries; ++i) {
    if (strcmp(pBinaries[i].pDeviceTargetSpec, image_target) == 0) {
      *pSelectedBinary = i;
      return UR_RESULT_SUCCESS;
    }
  }

  // No image can be loaded for the given device
  return UR_RESULT_ERROR_INVALID_BINARY;
}
