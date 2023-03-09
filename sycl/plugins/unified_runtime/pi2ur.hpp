//===---------------- pi2ur.hpp - PI API to UR API  --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
#pragma once

#include <unordered_map>

#include "zer_api.h"
#include <sycl/detail/pi.h>
#include <ur/ur.hpp>

// Map of UR error codes to PI error codes
static pi_result ur2piResult(zer_result_t urResult) {
  std::unordered_map<zer_result_t, pi_result> ErrorMapping = {
      {ZER_RESULT_SUCCESS, PI_SUCCESS},
      {ZER_RESULT_ERROR_UNKNOWN, PI_ERROR_UNKNOWN},
      {ZER_RESULT_ERROR_DEVICE_LOST, PI_ERROR_DEVICE_NOT_FOUND},
      {ZER_RESULT_INVALID_OPERATION, PI_ERROR_INVALID_OPERATION},
      {ZER_RESULT_INVALID_PLATFORM, PI_ERROR_INVALID_PLATFORM},
      {ZER_RESULT_ERROR_INVALID_ARGUMENT, PI_ERROR_INVALID_ARG_VALUE},
      {ZER_RESULT_INVALID_VALUE, PI_ERROR_INVALID_VALUE},
      {ZER_RESULT_INVALID_EVENT, PI_ERROR_INVALID_EVENT},
      {ZER_RESULT_INVALID_BINARY, PI_ERROR_INVALID_BINARY},
      {ZER_RESULT_INVALID_KERNEL_NAME, PI_ERROR_INVALID_KERNEL_NAME},
      {ZER_RESULT_ERROR_INVALID_FUNCTION_NAME, PI_ERROR_BUILD_PROGRAM_FAILURE},
      {ZER_RESULT_INVALID_WORK_GROUP_SIZE, PI_ERROR_INVALID_WORK_GROUP_SIZE},
      {ZER_RESULT_ERROR_MODULE_BUILD_FAILURE, PI_ERROR_BUILD_PROGRAM_FAILURE},
      {ZER_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, PI_ERROR_OUT_OF_RESOURCES},
      {ZER_RESULT_ERROR_OUT_OF_HOST_MEMORY, PI_ERROR_OUT_OF_HOST_MEMORY}};

  auto It = ErrorMapping.find(urResult);
  if (It == ErrorMapping.end()) {
    return PI_ERROR_UNKNOWN;
  }
  return It->second;
}

// Early exits on any error
#define HANDLE_ERRORS(urCall)                                                  \
  if (auto Result = urCall)                                                    \
    return ur2piResult(Result);

// A version of return helper that returns pi_result and not zer_result_t
class ReturnHelper : public UrReturnHelper {
public:
  using UrReturnHelper::UrReturnHelper;

  template <class T> pi_result operator()(const T &t) {
    return ur2piResult(UrReturnHelper::operator()(t));
  }
  // Array return value
  template <class T> pi_result operator()(const T *t, size_t s) {
    return ur2piResult(UrReturnHelper::operator()(t, s));
  }
  // Array return value where element type is differrent from T
  template <class RetType, class T> pi_result operator()(const T *t, size_t s) {
    return ur2piResult(UrReturnHelper::operator()<RetType>(t, s));
  }
};

// A version of return helper that supports conversion through a map
class ConvertHelper : public ReturnHelper {
  using ReturnHelper::ReturnHelper;

public:
  // Convert the value using a conversion map
  template <typename TypeUR, typename TypePI>
  pi_result convert(const std::unordered_map<TypeUR, TypePI> &Map) {
    *param_value_size_ret = sizeof(TypePI);

    // There is no value to convert.
    if (!param_value)
      return PI_SUCCESS;

    auto pValueUR = static_cast<TypeUR *>(param_value);
    auto pValuePI = static_cast<TypePI *>(param_value);

    // Cannot convert to a smaller storage type
    PI_ASSERT(sizeof(TypePI) >= sizeof(TypeUR), PI_ERROR_UNKNOWN);

    auto It = Map.find(*pValueUR);
    if (It == Map.end()) {
      die("ConvertHelper: unhandled value");
    }

    *pValuePI = It->second;
    return PI_SUCCESS;
  }

  // Convert the array (0-terminated) using a conversion map
  template <typename TypeUR, typename TypePI>
  pi_result convertArray(const std::unordered_map<TypeUR, TypePI> &Map) {
    // Cannot convert to a smaller element storage type
    PI_ASSERT(sizeof(TypePI) >= sizeof(TypeUR), PI_ERROR_UNKNOWN);
    *param_value_size_ret *= sizeof(TypePI) / sizeof(TypeUR);

    // There is no value to convert. Adjust to a possibly bigger PI storage.
    if (!param_value)
      return PI_SUCCESS;

    PI_ASSERT(*param_value_size_ret % sizeof(TypePI) == 0, PI_ERROR_UNKNOWN);

    // Make a copy of the input UR array as we may possibly overwrite following
    // elements while converting previous ones (if extending).
    auto ValueUR = new char[*param_value_size_ret];
    auto pValueUR = reinterpret_cast<TypeUR *>(ValueUR);
    auto pValuePI = static_cast<TypePI *>(param_value);
    memcpy(pValueUR, param_value, *param_value_size_ret);

    while (pValueUR) {
      if (*pValueUR == 0) {
        *pValuePI = 0;
        break;
      }

      auto It = Map.find(*pValueUR);
      if (It == Map.end()) {
        die("ConvertHelper: unhandled value");
      }
      *pValuePI = It->second;
      ++pValuePI;
      ++pValueUR;
    }

    delete[] ValueUR;
    return PI_SUCCESS;
  }

  // Convert the bitset using a conversion map
  template <typename TypeUR, typename TypePI>
  pi_result convertBitSet(const std::unordered_map<TypeUR, TypePI> &Map) {
    // There is no value to convert.
    if (!param_value)
      return PI_SUCCESS;

    auto pValuePI = static_cast<TypePI *>(param_value);
    auto pValueUR = static_cast<TypeUR *>(param_value);

    // Cannot handle biteset large than size_t
    PI_ASSERT(sizeof(TypeUR) <= sizeof(size_t), PI_ERROR_UNKNOWN);
    size_t In = *pValueUR;
    TypePI Out = 0;

    size_t Val;
    while ((Val = In & -In)) { // Val is the rightmost set bit in In
      In &= In - 1;            // Reset the rightmost set bit

      // Convert the Val alone and merge it into Out
      *pValueUR = TypeUR(Val);
      if (auto Res = convert(Map))
        return Res;
      Out |= *pValuePI;
    }
    *pValuePI = TypePI(Out);
    return PI_SUCCESS;
  }
};

// Translate UR info values to PI info values
inline pi_result ur2piInfoValue(zer_device_info_t ParamName,
                                size_t ParamValueSizePI,
                                size_t *ParamValueSizeUR, void *ParamValue) {

  ConvertHelper Value(ParamValueSizePI, ParamValue, ParamValueSizeUR);

  if (ParamName == ZER_DEVICE_INFO_TYPE) {
    static std::unordered_map<zer_device_type_t, pi_device_type> Map = {
        {ZER_DEVICE_TYPE_CPU, PI_DEVICE_TYPE_CPU},
        {ZER_DEVICE_TYPE_GPU, PI_DEVICE_TYPE_GPU},
        {ZER_DEVICE_TYPE_FPGA, PI_DEVICE_TYPE_ACC},
    };
    return Value.convert(Map);
  } else if (ParamName == ZER_DEVICE_INFO_QUEUE_PROPERTIES) {
    static std::unordered_map<zer_queue_flag_t, pi_queue_properties> Map = {
        {ZER_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE,
         PI_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE},
        {ZER_QUEUE_FLAG_PROFILING_ENABLE, PI_QUEUE_FLAG_PROFILING_ENABLE},
        {ZER_QUEUE_FLAG_ON_DEVICE, PI_QUEUE_FLAG_ON_DEVICE},
        {ZER_QUEUE_FLAG_ON_DEVICE_DEFAULT, PI_QUEUE_FLAG_ON_DEVICE_DEFAULT},
    };
    return Value.convertBitSet(Map);
  } else if (ParamName == ZER_DEVICE_INFO_EXECUTION_CAPABILITIES) {
    static std::unordered_map<zer_device_exec_capability_flag_t,
                              pi_queue_properties>
        Map = {
            {ZER_DEVICE_EXEC_CAPABILITY_FLAG_KERNEL,
             PI_DEVICE_EXEC_CAPABILITIES_KERNEL},
            {ZER_DEVICE_EXEC_CAPABILITY_FLAG_NATIVE_KERNEL,
             PI_DEVICE_EXEC_CAPABILITIES_NATIVE_KERNEL},
        };
    return Value.convertBitSet(Map);
  } else if (ParamName == ZER_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN) {
    static std::unordered_map<zer_device_affinity_domain_flag_t,
                              pi_device_affinity_domain>
        Map = {
            {ZER_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA,
             PI_DEVICE_AFFINITY_DOMAIN_NUMA},
            {ZER_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE,
             PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE},
        };
    return Value.convertBitSet(Map);
  } else if (ParamName == ZER_DEVICE_INFO_PARTITION_TYPE) {
    static std::unordered_map<zer_device_partition_property_flag_t,
                              pi_device_partition_property>
        Map = {
            {ZER_DEVICE_PARTITION_PROPERTY_FLAG_BY_AFFINITY_DOMAIN,
             PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN},
            {ZER_EXT_DEVICE_PARTITION_PROPERTY_FLAG_BY_CSLICE,
             PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE},
            {(zer_device_partition_property_flag_t)
                 ZER_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE,
             (pi_device_partition_property)
                 PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE},
        };
    return Value.convertArray(Map);
  } else if (ParamName == ZER_DEVICE_INFO_PARTITION_PROPERTIES) {
    static std::unordered_map<zer_device_partition_property_flag_t,
                              pi_device_partition_property>
        Map = {
            {ZER_DEVICE_PARTITION_PROPERTY_FLAG_BY_AFFINITY_DOMAIN,
             PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN},
            {ZER_EXT_DEVICE_PARTITION_PROPERTY_FLAG_BY_CSLICE,
             PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE},
        };
    return Value.convertArray(Map);
  }

  if (ParamValueSizePI && ParamValueSizePI != *ParamValueSizeUR) {
    fprintf(stderr, "UR InfoType=%d PI=%d but UR=%d\n", ParamName,
            (int)ParamValueSizePI, (int)*ParamValueSizeUR);
    die("ur2piInfoValue: size mismatch");
  }
  return PI_SUCCESS;
}

namespace pi2ur {
inline pi_result piPlatformsGet(pi_uint32 num_entries, pi_platform *platforms,
                                pi_uint32 *num_platforms) {

  // https://spec.oneapi.io/unified-runtime/latest/core/api.html#zerplatformget

  uint32_t Count = num_entries;
  auto phPlatforms = reinterpret_cast<zer_platform_handle_t *>(platforms);
  HANDLE_ERRORS(zerPlatformGet(&Count, phPlatforms));
  if (num_platforms) {
    *num_platforms = Count;
  }
  return PI_SUCCESS;
}

inline pi_result piPlatformGetInfo(pi_platform platform,
                                   pi_platform_info ParamName,
                                   size_t ParamValueSize, void *ParamValue,
                                   size_t *ParamValueSizeRet) {

  static std::unordered_map<pi_platform_info, zer_platform_info_t> InfoMapping =
      {
          {PI_PLATFORM_INFO_EXTENSIONS, ZER_PLATFORM_INFO_NAME},
          {PI_PLATFORM_INFO_NAME, ZER_PLATFORM_INFO_NAME},
          {PI_PLATFORM_INFO_PROFILE, ZER_PLATFORM_INFO_PROFILE},
          {PI_PLATFORM_INFO_VENDOR, ZER_PLATFORM_INFO_VENDOR_NAME},
          {PI_PLATFORM_INFO_VERSION, ZER_PLATFORM_INFO_VERSION},
      };

  auto InfoType = InfoMapping.find(ParamName);
  if (InfoType == InfoMapping.end()) {
    return PI_ERROR_UNKNOWN;
  }

  size_t SizeInOut = ParamValueSize;
  auto hPlatform = reinterpret_cast<zer_platform_handle_t>(platform);
  HANDLE_ERRORS(
      zerPlatformGetInfo(hPlatform, InfoType->second, &SizeInOut, ParamValue));
  if (ParamValueSizeRet) {
    *ParamValueSizeRet = SizeInOut;
  }
  return PI_SUCCESS;
}

inline pi_result piDevicesGet(pi_platform Platform, pi_device_type DeviceType,
                              pi_uint32 NumEntries, pi_device *Devices,
                              pi_uint32 *NumDevices) {

  static std::unordered_map<pi_device_type, zer_device_type_t> TypeMapping = {
      {PI_DEVICE_TYPE_ALL, ZER_DEVICE_TYPE_ALL},
      {PI_DEVICE_TYPE_GPU, ZER_DEVICE_TYPE_GPU},
      {PI_DEVICE_TYPE_CPU, ZER_DEVICE_TYPE_CPU},
      {PI_DEVICE_TYPE_ACC, ZER_DEVICE_TYPE_FPGA},
  };

  auto Type = TypeMapping.find(DeviceType);
  if (Type == TypeMapping.end()) {
    return PI_ERROR_UNKNOWN;
  }

  uint32_t Count = NumEntries;
  auto hPlatform = reinterpret_cast<zer_platform_handle_t>(Platform);
  auto phDevices = reinterpret_cast<zer_device_handle_t *>(Devices);
  HANDLE_ERRORS(zerDeviceGet(hPlatform, Type->second, &Count, phDevices));
  if (NumDevices) {
    *NumDevices = Count;
  }
  return PI_SUCCESS;
}

inline pi_result piDeviceRetain(pi_device Device) {
  auto hDevice = reinterpret_cast<zer_device_handle_t>(Device);
  HANDLE_ERRORS(zerDeviceGetReference(hDevice));
  return PI_SUCCESS;
}

inline pi_result piDeviceRelease(pi_device Device) {
  auto hDevice = reinterpret_cast<zer_device_handle_t>(Device);
  HANDLE_ERRORS(zerDeviceRelease(hDevice));
  return PI_SUCCESS;
}

inline pi_result piPluginGetLastError(char **message) {
  HANDLE_ERRORS(zerPluginGetLastError(message));
  return PI_SUCCESS;
}

inline pi_result piDeviceGetInfo(pi_device Device, pi_device_info ParamName,
                                 size_t ParamValueSize, void *ParamValue,
                                 size_t *ParamValueSizeRet) {

  static std::unordered_map<pi_device_info, zer_device_info_t> InfoMapping = {
      {PI_DEVICE_INFO_TYPE, ZER_DEVICE_INFO_TYPE},
      {PI_DEVICE_INFO_PARENT_DEVICE, ZER_DEVICE_INFO_PARENT_DEVICE},
      {PI_DEVICE_INFO_PLATFORM, ZER_DEVICE_INFO_PLATFORM},
      {PI_DEVICE_INFO_VENDOR_ID, ZER_DEVICE_INFO_VENDOR_ID},
      {PI_DEVICE_INFO_UUID, ZER_DEVICE_INFO_UUID},
      {PI_DEVICE_INFO_ATOMIC_64, ZER_DEVICE_INFO_ATOMIC_64},
      {PI_DEVICE_INFO_EXTENSIONS, ZER_DEVICE_INFO_EXTENSIONS},
      {PI_DEVICE_INFO_NAME, ZER_DEVICE_INFO_NAME},
      {PI_DEVICE_INFO_COMPILER_AVAILABLE, ZER_DEVICE_INFO_COMPILER_AVAILABLE},
      {PI_DEVICE_INFO_LINKER_AVAILABLE, ZER_DEVICE_INFO_LINKER_AVAILABLE},
      {PI_DEVICE_INFO_MAX_COMPUTE_UNITS, ZER_DEVICE_INFO_MAX_COMPUTE_UNITS},
      {PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
       ZER_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS},
      {PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE, ZER_DEVICE_INFO_MAX_WORK_GROUP_SIZE},
      {PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES, ZER_DEVICE_INFO_MAX_WORK_ITEM_SIZES},
      {PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY, ZER_DEVICE_INFO_MAX_CLOCK_FREQUENCY},
      {PI_DEVICE_INFO_ADDRESS_BITS, ZER_DEVICE_INFO_ADDRESS_BITS},
      {PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE, ZER_DEVICE_INFO_MAX_MEM_ALLOC_SIZE},
      {PI_DEVICE_INFO_GLOBAL_MEM_SIZE, ZER_DEVICE_INFO_GLOBAL_MEM_SIZE},
      {PI_DEVICE_INFO_LOCAL_MEM_SIZE, ZER_DEVICE_INFO_LOCAL_MEM_SIZE},
      {PI_DEVICE_INFO_IMAGE_SUPPORT, ZER_DEVICE_INFO_IMAGE_SUPPORTED},
      {PI_DEVICE_INFO_HOST_UNIFIED_MEMORY, ZER_DEVICE_INFO_HOST_UNIFIED_MEMORY},
      {PI_DEVICE_INFO_AVAILABLE, ZER_DEVICE_INFO_AVAILABLE},
      {PI_DEVICE_INFO_VENDOR, ZER_DEVICE_INFO_VENDOR},
      {PI_DEVICE_INFO_DRIVER_VERSION, ZER_DEVICE_INFO_DRIVER_VERSION},
      {PI_DEVICE_INFO_VERSION, ZER_DEVICE_INFO_VERSION},
      {PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES,
       ZER_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES},
      {PI_DEVICE_INFO_REFERENCE_COUNT, ZER_DEVICE_INFO_REFERENCE_COUNT},
      {PI_DEVICE_INFO_PARTITION_PROPERTIES,
       ZER_DEVICE_INFO_PARTITION_PROPERTIES},
      {PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN,
       ZER_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN},
      {PI_DEVICE_INFO_PARTITION_TYPE, ZER_DEVICE_INFO_PARTITION_TYPE},
      {PI_DEVICE_INFO_OPENCL_C_VERSION, ZER_DEVICE_INFO_OPENCL_C_VERSION},
      {PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC,
       ZER_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC},
      {PI_DEVICE_INFO_PRINTF_BUFFER_SIZE, ZER_DEVICE_INFO_PRINTF_BUFFER_SIZE},
      {PI_DEVICE_INFO_PROFILE, ZER_DEVICE_INFO_PROFILE},
      {PI_DEVICE_INFO_BUILT_IN_KERNELS, ZER_DEVICE_INFO_BUILT_IN_KERNELS},
      {PI_DEVICE_INFO_QUEUE_PROPERTIES, ZER_DEVICE_INFO_QUEUE_PROPERTIES},
      {PI_DEVICE_INFO_EXECUTION_CAPABILITIES,
       ZER_DEVICE_INFO_EXECUTION_CAPABILITIES},
      {PI_DEVICE_INFO_ENDIAN_LITTLE, ZER_DEVICE_INFO_ENDIAN_LITTLE},
      {PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT,
       ZER_DEVICE_INFO_ERROR_CORRECTION_SUPPORT},
      {PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION,
       ZER_DEVICE_INFO_PROFILING_TIMER_RESOLUTION},
      {PI_DEVICE_INFO_LOCAL_MEM_TYPE, ZER_DEVICE_INFO_LOCAL_MEM_TYPE},
      {PI_DEVICE_INFO_MAX_CONSTANT_ARGS, ZER_DEVICE_INFO_MAX_CONSTANT_ARGS},
      {PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE,
       ZER_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE},
      {PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE,
       ZER_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE},
      {PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE,
       ZER_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE},
      {PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE,
       ZER_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE},
      {PI_DEVICE_INFO_MAX_PARAMETER_SIZE, ZER_DEVICE_INFO_MAX_PARAMETER_SIZE},
      {PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN, ZER_DEVICE_INFO_MEM_BASE_ADDR_ALIGN},
      {PI_DEVICE_INFO_MAX_SAMPLERS, ZER_DEVICE_INFO_MAX_SAMPLERS},
      {PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS, ZER_DEVICE_INFO_MAX_READ_IMAGE_ARGS},
      {PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS,
       ZER_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS},
      {PI_DEVICE_INFO_SINGLE_FP_CONFIG, ZER_DEVICE_INFO_SINGLE_FP_CONFIG},
      {PI_DEVICE_INFO_HALF_FP_CONFIG, ZER_DEVICE_INFO_HALF_FP_CONFIG},
      {PI_DEVICE_INFO_DOUBLE_FP_CONFIG, ZER_DEVICE_INFO_DOUBLE_FP_CONFIG},
      {PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH, ZER_DEVICE_INFO_IMAGE2D_MAX_WIDTH},
      {PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT, ZER_DEVICE_INFO_IMAGE2D_MAX_HEIGHT},
      {PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH, ZER_DEVICE_INFO_IMAGE3D_MAX_WIDTH},
      {PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT, ZER_DEVICE_INFO_IMAGE3D_MAX_HEIGHT},
      {PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH, ZER_DEVICE_INFO_IMAGE3D_MAX_DEPTH},
      {PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE,
       ZER_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE},
      {PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR,
       ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR,
       ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT,
       ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT,
       ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT,
       ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT,
       ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG,
       ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG,
       ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT,
       ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT,
       ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE,
       ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE,
       ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF,
       ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF,
       ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF},
      {PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS, ZER_DEVICE_INFO_MAX_NUM_SUB_GROUPS},
      {PI_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
       ZER_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS},
      {PI_DEVICE_INFO_SUB_GROUP_SIZES_INTEL,
       ZER_DEVICE_INFO_SUB_GROUP_SIZES_INTEL},
      {PI_DEVICE_INFO_IL_VERSION, ZER_DEVICE_INFO_IL_VERSION},
      {PI_DEVICE_INFO_USM_HOST_SUPPORT, ZER_DEVICE_INFO_USM_HOST_SUPPORT},
      {PI_DEVICE_INFO_USM_DEVICE_SUPPORT, ZER_DEVICE_INFO_USM_DEVICE_SUPPORT},
      {PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT,
       ZER_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT},
      {PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT,
       ZER_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT},
      {PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT,
       ZER_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT},
      {PI_DEVICE_INFO_USM_HOST_SUPPORT, ZER_DEVICE_INFO_USM_HOST_SUPPORT},
      {PI_DEVICE_INFO_USM_DEVICE_SUPPORT, ZER_DEVICE_INFO_USM_DEVICE_SUPPORT},
      {PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT,
       ZER_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT},
      {PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT,
       ZER_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT},
      {PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT,
       ZER_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT},
      {PI_DEVICE_INFO_PCI_ADDRESS, ZER_DEVICE_INFO_PCI_ADDRESS},
      {PI_DEVICE_INFO_GPU_EU_COUNT, ZER_DEVICE_INFO_GPU_EU_COUNT},
      {PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH, ZER_DEVICE_INFO_GPU_EU_SIMD_WIDTH},
      {PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE,
       ZER_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE},
      {PI_DEVICE_INFO_BUILD_ON_SUBDEVICE,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_BUILD_ON_SUBDEVICE},
      {PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_3D,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_MAX_WORK_GROUPS_3D},
      {PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE},
      {PI_DEVICE_INFO_DEVICE_ID,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_DEVICE_ID},
      {PI_EXT_INTEL_DEVICE_INFO_FREE_MEMORY,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_FREE_MEMORY},
      {PI_EXT_INTEL_DEVICE_INFO_MEMORY_CLOCK_RATE,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_MEMORY_CLOCK_RATE},
      {PI_EXT_INTEL_DEVICE_INFO_MEMORY_BUS_WIDTH,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_MEMORY_BUS_WIDTH},
      {PI_EXT_INTEL_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES},
      {PI_DEVICE_INFO_GPU_SLICES,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_GPU_SLICES},
      {PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE},
      {PI_DEVICE_INFO_GPU_HW_THREADS_PER_EU,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_GPU_HW_THREADS_PER_EU},
      {PI_DEVICE_INFO_MAX_MEM_BANDWIDTH,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_MAX_MEM_BANDWIDTH},
      {PI_EXT_ONEAPI_DEVICE_INFO_BFLOAT16_MATH_FUNCTIONS,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_BFLOAT16_MATH_FUNCTIONS},
      {PI_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES,
       (zer_device_info_t)ZER_EXT_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES},
      {PI_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES,
       (zer_device_info_t)ZER_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES},
  };

  auto InfoType = InfoMapping.find(ParamName);
  if (InfoType == InfoMapping.end()) {
    return PI_ERROR_UNKNOWN;
  }

  size_t SizeInOut = ParamValueSize;
  auto hDevice = reinterpret_cast<zer_device_handle_t>(Device);
  HANDLE_ERRORS(
      zerDeviceGetInfo(hDevice, InfoType->second, &SizeInOut, ParamValue));

  ur2piInfoValue(InfoType->second, ParamValueSize, &SizeInOut, ParamValue);

  if (ParamValueSizeRet) {
    *ParamValueSizeRet = SizeInOut;
  }
  return PI_SUCCESS;
}

inline pi_result piDevicePartition(
    pi_device Device, const pi_device_partition_property *Properties,
    pi_uint32 NumEntries, pi_device *SubDevices, pi_uint32 *NumSubDevices) {

  if (!Properties || !Properties[0])
    return PI_ERROR_INVALID_VALUE;

  static std::unordered_map<pi_device_partition_property,
                            zer_device_partition_property_flag_t>
      PropertyMap = {
          {PI_DEVICE_PARTITION_EQUALLY,
           ZER_DEVICE_PARTITION_PROPERTY_FLAG_EQUALLY},
          {PI_DEVICE_PARTITION_BY_COUNTS,
           ZER_DEVICE_PARTITION_PROPERTY_FLAG_BY_COUNTS},
          {PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
           ZER_DEVICE_PARTITION_PROPERTY_FLAG_BY_AFFINITY_DOMAIN},
          {PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE,
           ZER_EXT_DEVICE_PARTITION_PROPERTY_FLAG_BY_CSLICE},
      };

  auto PropertyIt = PropertyMap.find(Properties[0]);
  if (PropertyIt == PropertyMap.end()) {
    return PI_ERROR_UNKNOWN;
  }

  // Some partitioning types require a value
  auto Value = uint32_t(Properties[1]);
  if (PropertyIt->second ==
      ZER_DEVICE_PARTITION_PROPERTY_FLAG_BY_AFFINITY_DOMAIN) {
    static std::unordered_map<pi_device_affinity_domain,
                              zer_device_affinity_domain_flag_t>
        ValueMap = {
            {PI_DEVICE_AFFINITY_DOMAIN_NUMA,
             ZER_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA},
            {PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE,
             ZER_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE},
        };
    auto ValueIt = ValueMap.find(Properties[1]);
    if (ValueIt == ValueMap.end()) {
      return PI_ERROR_UNKNOWN;
    }
    Value = ValueIt->second;
  }

  // Translate partitioning properties from PI-way
  // (array of uintptr_t values) to UR-way
  // (array of {uint32_t, uint32_t} pairs)
  //
  // TODO: correctly terminate the UR properties, see:
  // https://github.com/oneapi-src/unified-runtime/issues/183
  //
  zer_device_partition_property_value_t UrProperties[] = {
      {zer_device_partition_property_flags_t(PropertyIt->second), Value},
      {0, 0}};

  uint32_t Count = NumEntries;
  auto hDevice = reinterpret_cast<zer_device_handle_t>(Device);
  auto phSubDevices = reinterpret_cast<zer_device_handle_t *>(SubDevices);
  HANDLE_ERRORS(
      zerDevicePartition(hDevice, UrProperties, &Count, phSubDevices));
  if (NumSubDevices) {
    *NumSubDevices = Count;
  }
  return PI_SUCCESS;
}
} // namespace pi2ur
