//===---------------- pi2ur.hpp - PI API to UR API  --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
#pragma once

#include <unordered_map>

#include "ur_api.h"
#include <sycl/detail/pi.h>
#include <ur/ur.hpp>

// Map of UR error codes to PI error codes
static pi_result ur2piResult(ur_result_t urResult) {
  std::unordered_map<ur_result_t, pi_result> ErrorMapping = {
      {UR_RESULT_SUCCESS, PI_SUCCESS},
      {UR_RESULT_ERROR_UNKNOWN, PI_ERROR_UNKNOWN},
      {UR_RESULT_ERROR_DEVICE_LOST, PI_ERROR_DEVICE_NOT_FOUND},
      {UR_RESULT_ERROR_INVALID_OPERATION, PI_ERROR_INVALID_OPERATION},
      {UR_RESULT_ERROR_INVALID_PLATFORM, PI_ERROR_INVALID_PLATFORM},
      {UR_RESULT_ERROR_INVALID_ARGUMENT, PI_ERROR_INVALID_ARG_VALUE},
      {UR_RESULT_ERROR_INVALID_VALUE, PI_ERROR_INVALID_VALUE},
      {UR_RESULT_ERROR_INVALID_EVENT, PI_ERROR_INVALID_EVENT},
      {UR_RESULT_ERROR_INVALID_BINARY, PI_ERROR_INVALID_BINARY},
      {UR_RESULT_ERROR_INVALID_KERNEL_NAME, PI_ERROR_INVALID_KERNEL_NAME},
      {UR_RESULT_ERROR_INVALID_FUNCTION_NAME, PI_ERROR_BUILD_PROGRAM_FAILURE},
      {UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE,
       PI_ERROR_INVALID_WORK_GROUP_SIZE},
      {UR_RESULT_ERROR_MODULE_BUILD_FAILURE, PI_ERROR_BUILD_PROGRAM_FAILURE},
      {UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, PI_ERROR_OUT_OF_RESOURCES},
      {UR_RESULT_ERROR_OUT_OF_HOST_MEMORY, PI_ERROR_OUT_OF_HOST_MEMORY}};

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

// A version of return helper that returns pi_result and not ur_result_t
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
inline pi_result ur2piInfoValue(ur_device_info_t ParamName,
                                size_t ParamValueSizePI,
                                size_t *ParamValueSizeUR, void *ParamValue) {

  ConvertHelper Value(ParamValueSizePI, ParamValue, ParamValueSizeUR);

  if (ParamName == UR_DEVICE_INFO_TYPE) {
    static std::unordered_map<ur_device_type_t, pi_device_type> Map = {
        {UR_DEVICE_TYPE_CPU, PI_DEVICE_TYPE_CPU},
        {UR_DEVICE_TYPE_GPU, PI_DEVICE_TYPE_GPU},
        {UR_DEVICE_TYPE_FPGA, PI_DEVICE_TYPE_ACC},
    };
    return Value.convert(Map);
  } else if (ParamName == UR_DEVICE_INFO_QUEUE_PROPERTIES) {
    static std::unordered_map<ur_queue_flag_t, pi_queue_properties> Map = {
        {UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE,
         PI_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE},
        {UR_QUEUE_FLAG_PROFILING_ENABLE, PI_QUEUE_FLAG_PROFILING_ENABLE},
        {UR_QUEUE_FLAG_ON_DEVICE, PI_QUEUE_FLAG_ON_DEVICE},
        {UR_QUEUE_FLAG_ON_DEVICE_DEFAULT, PI_QUEUE_FLAG_ON_DEVICE_DEFAULT},
    };
    return Value.convertBitSet(Map);
  } else if (ParamName == UR_DEVICE_INFO_EXECUTION_CAPABILITIES) {
    static std::unordered_map<ur_device_exec_capability_flag_t,
                              pi_queue_properties>
        Map = {
            {UR_DEVICE_EXEC_CAPABILITY_FLAG_KERNEL,
             PI_DEVICE_EXEC_CAPABILITIES_KERNEL},
            {UR_DEVICE_EXEC_CAPABILITY_FLAG_NATIVE_KERNEL,
             PI_DEVICE_EXEC_CAPABILITIES_NATIVE_KERNEL},
        };
    return Value.convertBitSet(Map);
  } else if (ParamName == UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN) {
    static std::unordered_map<ur_device_affinity_domain_flag_t,
                              pi_device_affinity_domain>
        Map = {
            {UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA,
             PI_DEVICE_AFFINITY_DOMAIN_NUMA},
            {UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE,
             PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE},
        };
    return Value.convertBitSet(Map);
  } else if (ParamName == UR_DEVICE_INFO_PARTITION_TYPE) {
    static std::unordered_map<ur_device_partition_property_t,
                              pi_device_partition_property>
        Map = {
            {UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
             PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN},
            {UR_EXT_DEVICE_PARTITION_PROPERTY_FLAG_BY_CSLICE,
             PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE},
            {(ur_device_partition_property_t)
                 UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE,
             (pi_device_partition_property)
                 PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE},
        };
    return Value.convertArray(Map);
  } else if (ParamName == UR_DEVICE_INFO_PARTITION_PROPERTIES) {
    static std::unordered_map<ur_device_partition_property_t,
                              pi_device_partition_property>
        Map = {
            {UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
             PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN},
            {UR_EXT_DEVICE_PARTITION_PROPERTY_FLAG_BY_CSLICE,
             PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE},
        };
    return Value.convertArray(Map);
  } else if (ParamName == UR_DEVICE_INFO_LOCAL_MEM_TYPE) {
    static std::unordered_map<ur_device_local_mem_type_t,
                              pi_device_local_mem_type>
        Map = {
            {UR_DEVICE_LOCAL_MEM_TYPE_LOCAL, PI_DEVICE_LOCAL_MEM_TYPE_LOCAL},
            {UR_DEVICE_LOCAL_MEM_TYPE_GLOBAL, PI_DEVICE_LOCAL_MEM_TYPE_GLOBAL},
        };
    return Value.convert(Map);
  } else if (ParamName == UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES) {
    static std::unordered_map<ur_memory_order_capability_flag_t,
                              pi_memory_order_capabilities>
        Map = {
            {UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED, PI_MEMORY_ORDER_RELAXED},
            {UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE, PI_MEMORY_ORDER_ACQUIRE},
            {UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE, PI_MEMORY_ORDER_RELEASE},
            {UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL, PI_MEMORY_ORDER_ACQ_REL},
            {UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST, PI_MEMORY_ORDER_SEQ_CST},
        };
    return Value.convertBitSet(Map);
  } else if (ParamName == UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES) {
    static std::unordered_map<ur_memory_scope_capability_flag_t,
                              pi_memory_scope_capabilities>
        Map = {
            {UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM,
             PI_MEMORY_SCOPE_WORK_ITEM},
            {UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP,
             PI_MEMORY_SCOPE_SUB_GROUP},
            {UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP,
             PI_MEMORY_SCOPE_WORK_GROUP},
            {UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE, PI_MEMORY_SCOPE_DEVICE},
            {UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM, PI_MEMORY_SCOPE_SYSTEM},
        };
    return Value.convertBitSet(Map);
  } else if (ParamName == UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES) {
    static std::unordered_map<ur_memory_order_capability_flag_t,
                              pi_memory_order_capabilities>
        Map = {
            {UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED, PI_MEMORY_ORDER_RELAXED},
            {UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE, PI_MEMORY_ORDER_ACQUIRE},
            {UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE, PI_MEMORY_ORDER_RELEASE},
            {UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL, PI_MEMORY_ORDER_ACQ_REL},
            {UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST, PI_MEMORY_ORDER_SEQ_CST},
        };
    return Value.convertBitSet(Map);
  } else if (ParamName == UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES) {
    static std::unordered_map<ur_memory_scope_capability_flag_t,
                              pi_memory_scope_capabilities>
        Map = {
            {UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM,
             PI_MEMORY_SCOPE_WORK_ITEM},
            {UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP,
             PI_MEMORY_SCOPE_SUB_GROUP},
            {UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP,
             PI_MEMORY_SCOPE_WORK_GROUP},
            {UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE, PI_MEMORY_SCOPE_DEVICE},
            {UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM, PI_MEMORY_SCOPE_SYSTEM},
        };
    return Value.convertBitSet(Map);
  } else {
    // TODO: what else needs a UR-PI translation?
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

  uint32_t Count = num_entries;
  auto phPlatforms = reinterpret_cast<ur_platform_handle_t *>(platforms);
  HANDLE_ERRORS(urPlatformGet(Count, phPlatforms, num_platforms));
  return PI_SUCCESS;
}

inline pi_result piPlatformGetInfo(pi_platform platform,
                                   pi_platform_info ParamName,
                                   size_t ParamValueSize, void *ParamValue,
                                   size_t *ParamValueSizeRet) {

  static std::unordered_map<pi_platform_info, ur_platform_info_t> InfoMapping =
      {
          {PI_PLATFORM_INFO_EXTENSIONS, UR_PLATFORM_INFO_NAME},
          {PI_PLATFORM_INFO_NAME, UR_PLATFORM_INFO_NAME},
          {PI_PLATFORM_INFO_PROFILE, UR_PLATFORM_INFO_PROFILE},
          {PI_PLATFORM_INFO_VENDOR, UR_PLATFORM_INFO_VENDOR_NAME},
          {PI_PLATFORM_INFO_VERSION, UR_PLATFORM_INFO_VERSION},
      };

  auto InfoType = InfoMapping.find(ParamName);
  if (InfoType == InfoMapping.end()) {
    return PI_ERROR_UNKNOWN;
  }

  size_t SizeInOut = ParamValueSize;
  auto hPlatform = reinterpret_cast<ur_platform_handle_t>(platform);
  HANDLE_ERRORS(urPlatformGetInfo(hPlatform, InfoType->second, SizeInOut,
                                  ParamValue, ParamValueSizeRet));
  return PI_SUCCESS;
}

inline pi_result piDevicesGet(pi_platform Platform, pi_device_type DeviceType,
                              pi_uint32 NumEntries, pi_device *Devices,
                              pi_uint32 *NumDevices) {

  static std::unordered_map<pi_device_type, ur_device_type_t> TypeMapping = {
      {PI_DEVICE_TYPE_ALL, UR_DEVICE_TYPE_ALL},
      {PI_DEVICE_TYPE_GPU, UR_DEVICE_TYPE_GPU},
      {PI_DEVICE_TYPE_CPU, UR_DEVICE_TYPE_CPU},
      {PI_DEVICE_TYPE_ACC, UR_DEVICE_TYPE_FPGA},
  };

  auto Type = TypeMapping.find(DeviceType);
  if (Type == TypeMapping.end()) {
    return PI_ERROR_UNKNOWN;
  }

  uint32_t Count = NumEntries;
  auto hPlatform = reinterpret_cast<ur_platform_handle_t>(Platform);
  auto phDevices = reinterpret_cast<ur_device_handle_t *>(Devices);
  HANDLE_ERRORS(
      urDeviceGet(hPlatform, Type->second, Count, phDevices, NumDevices));
  return PI_SUCCESS;
}

inline pi_result piDeviceRetain(pi_device Device) {
  auto hDevice = reinterpret_cast<ur_device_handle_t>(Device);
  HANDLE_ERRORS(urDeviceRetain(hDevice));
  return PI_SUCCESS;
}

inline pi_result piDeviceRelease(pi_device Device) {
  auto hDevice = reinterpret_cast<ur_device_handle_t>(Device);
  HANDLE_ERRORS(urDeviceRelease(hDevice));
  return PI_SUCCESS;
}

inline pi_result piPluginGetLastError(char **) { return PI_SUCCESS; }

inline pi_result piDeviceGetInfo(pi_device Device, pi_device_info ParamName,
                                 size_t ParamValueSize, void *ParamValue,
                                 size_t *ParamValueSizeRet) {

  static std::unordered_map<pi_device_info, ur_device_info_t> InfoMapping = {
      {PI_DEVICE_INFO_TYPE, UR_DEVICE_INFO_TYPE},
      {PI_DEVICE_INFO_PARENT_DEVICE, UR_DEVICE_INFO_PARENT_DEVICE},
      {PI_DEVICE_INFO_PLATFORM, UR_DEVICE_INFO_PLATFORM},
      {PI_DEVICE_INFO_VENDOR_ID, UR_DEVICE_INFO_VENDOR_ID},
      {PI_DEVICE_INFO_UUID, UR_DEVICE_INFO_UUID},
      {PI_DEVICE_INFO_ATOMIC_64, UR_DEVICE_INFO_ATOMIC_64},
      {PI_DEVICE_INFO_EXTENSIONS, UR_DEVICE_INFO_EXTENSIONS},
      {PI_DEVICE_INFO_NAME, UR_DEVICE_INFO_NAME},
      {PI_DEVICE_INFO_COMPILER_AVAILABLE, UR_DEVICE_INFO_COMPILER_AVAILABLE},
      {PI_DEVICE_INFO_LINKER_AVAILABLE, UR_DEVICE_INFO_LINKER_AVAILABLE},
      {PI_DEVICE_INFO_MAX_COMPUTE_UNITS, UR_DEVICE_INFO_MAX_COMPUTE_UNITS},
      {PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
       UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS},
      {PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE, UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE},
      {PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES, UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES},
      {PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY, UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY},
      {PI_DEVICE_INFO_ADDRESS_BITS, UR_DEVICE_INFO_ADDRESS_BITS},
      {PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE, UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE},
      {PI_DEVICE_INFO_GLOBAL_MEM_SIZE, UR_DEVICE_INFO_GLOBAL_MEM_SIZE},
      {PI_DEVICE_INFO_LOCAL_MEM_SIZE, UR_DEVICE_INFO_LOCAL_MEM_SIZE},
      {PI_DEVICE_INFO_IMAGE_SUPPORT, UR_DEVICE_INFO_IMAGE_SUPPORTED},
      {PI_DEVICE_INFO_HOST_UNIFIED_MEMORY, UR_DEVICE_INFO_HOST_UNIFIED_MEMORY},
      {PI_DEVICE_INFO_AVAILABLE, UR_DEVICE_INFO_AVAILABLE},
      {PI_DEVICE_INFO_VENDOR, UR_DEVICE_INFO_VENDOR},
      {PI_DEVICE_INFO_DRIVER_VERSION, UR_DEVICE_INFO_DRIVER_VERSION},
      {PI_DEVICE_INFO_VERSION, UR_DEVICE_INFO_VERSION},
      {PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES,
       UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES},
      {PI_DEVICE_INFO_REFERENCE_COUNT, UR_DEVICE_INFO_REFERENCE_COUNT},
      {PI_DEVICE_INFO_PARTITION_PROPERTIES,
       UR_DEVICE_INFO_PARTITION_PROPERTIES},
      {PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN,
       UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN},
      {PI_DEVICE_INFO_PARTITION_TYPE, UR_DEVICE_INFO_PARTITION_TYPE},
      {PI_DEVICE_INFO_OPENCL_C_VERSION, UR_EXT_DEVICE_INFO_OPENCL_C_VERSION},
      {PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC,
       UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC},
      {PI_DEVICE_INFO_PRINTF_BUFFER_SIZE, UR_DEVICE_INFO_PRINTF_BUFFER_SIZE},
      {PI_DEVICE_INFO_PROFILE, UR_DEVICE_INFO_PROFILE},
      {PI_DEVICE_INFO_BUILT_IN_KERNELS, UR_DEVICE_INFO_BUILT_IN_KERNELS},
      {PI_DEVICE_INFO_QUEUE_PROPERTIES, UR_DEVICE_INFO_QUEUE_PROPERTIES},
      {PI_DEVICE_INFO_EXECUTION_CAPABILITIES,
       UR_DEVICE_INFO_EXECUTION_CAPABILITIES},
      {PI_DEVICE_INFO_ENDIAN_LITTLE, UR_DEVICE_INFO_ENDIAN_LITTLE},
      {PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT,
       UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT},
      {PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION,
       UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION},
      {PI_DEVICE_INFO_LOCAL_MEM_TYPE, UR_DEVICE_INFO_LOCAL_MEM_TYPE},
      {PI_DEVICE_INFO_MAX_CONSTANT_ARGS, UR_DEVICE_INFO_MAX_CONSTANT_ARGS},
      {PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE,
       UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE},
      {PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE,
       UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE},
      {PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE,
       UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE},
      {PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE,
       UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE},
      {PI_DEVICE_INFO_MAX_PARAMETER_SIZE, UR_DEVICE_INFO_MAX_PARAMETER_SIZE},
      {PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN, UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN},
      {PI_DEVICE_INFO_MAX_SAMPLERS, UR_DEVICE_INFO_MAX_SAMPLERS},
      {PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS, UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS},
      {PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS,
       UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS},
      {PI_DEVICE_INFO_SINGLE_FP_CONFIG, UR_DEVICE_INFO_SINGLE_FP_CONFIG},
      {PI_DEVICE_INFO_HALF_FP_CONFIG, UR_DEVICE_INFO_HALF_FP_CONFIG},
      {PI_DEVICE_INFO_DOUBLE_FP_CONFIG, UR_DEVICE_INFO_DOUBLE_FP_CONFIG},
      {PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH, UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH},
      {PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT, UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT},
      {PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH, UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH},
      {PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT, UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT},
      {PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH, UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH},
      {PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE,
       UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE},
      {PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE,
       (ur_device_info_t)UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR,
       UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR,
       UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT,
       UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT,
       UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT,
       UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT,
       UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG,
       UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG,
       UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT,
       UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT,
       UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE,
       UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE,
       UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE},
      {PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF,
       UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF},
      {PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF,
       UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF},
      {PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS, UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS},
      {PI_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
       UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS},
      {PI_DEVICE_INFO_SUB_GROUP_SIZES_INTEL,
       UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL},
      {PI_DEVICE_INFO_IL_VERSION, UR_DEVICE_INFO_IL_VERSION},
      {PI_DEVICE_INFO_USM_HOST_SUPPORT, UR_DEVICE_INFO_USM_HOST_SUPPORT},
      {PI_DEVICE_INFO_USM_DEVICE_SUPPORT, UR_DEVICE_INFO_USM_DEVICE_SUPPORT},
      {PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT,
       UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT},
      {PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT,
       UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT},
      {PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT,
       UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT},
      {PI_DEVICE_INFO_USM_HOST_SUPPORT, UR_DEVICE_INFO_USM_HOST_SUPPORT},
      {PI_DEVICE_INFO_USM_DEVICE_SUPPORT, UR_DEVICE_INFO_USM_DEVICE_SUPPORT},
      {PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT,
       UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT},
      {PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT,
       UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT},
      {PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT,
       UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT},
      {PI_DEVICE_INFO_PCI_ADDRESS, UR_DEVICE_INFO_PCI_ADDRESS},
      {PI_DEVICE_INFO_GPU_EU_COUNT, UR_DEVICE_INFO_GPU_EU_COUNT},
      {PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH, UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH},
      {PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE,
       UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE},
      {PI_DEVICE_INFO_BUILD_ON_SUBDEVICE,
       (ur_device_info_t)UR_EXT_DEVICE_INFO_BUILD_ON_SUBDEVICE},
      {PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_3D,
       (ur_device_info_t)UR_EXT_DEVICE_INFO_MAX_WORK_GROUPS_3D},
      {PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE,
       (ur_device_info_t)UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE},
      {PI_DEVICE_INFO_DEVICE_ID, (ur_device_info_t)UR_DEVICE_INFO_DEVICE_ID},
      {PI_EXT_INTEL_DEVICE_INFO_FREE_MEMORY,
       (ur_device_info_t)UR_EXT_DEVICE_INFO_FREE_MEMORY},
      {PI_EXT_INTEL_DEVICE_INFO_MEMORY_CLOCK_RATE,
       (ur_device_info_t)UR_DEVICE_INFO_MEMORY_CLOCK_RATE},
      {PI_EXT_INTEL_DEVICE_INFO_MEMORY_BUS_WIDTH,
       (ur_device_info_t)UR_EXT_DEVICE_INFO_MEMORY_BUS_WIDTH},
      {PI_EXT_INTEL_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES,
       (ur_device_info_t)UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES},
      {PI_DEVICE_INFO_GPU_SLICES,
       (ur_device_info_t)UR_EXT_DEVICE_INFO_GPU_SLICES},
      {PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE,
       (ur_device_info_t)UR_EXT_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE},
      {PI_DEVICE_INFO_GPU_HW_THREADS_PER_EU,
       (ur_device_info_t)UR_EXT_DEVICE_INFO_GPU_HW_THREADS_PER_EU},
      {PI_DEVICE_INFO_MAX_MEM_BANDWIDTH,
       (ur_device_info_t)UR_EXT_DEVICE_INFO_MAX_MEM_BANDWIDTH},
      {PI_EXT_ONEAPI_DEVICE_INFO_BFLOAT16_MATH_FUNCTIONS,
       (ur_device_info_t)UR_DEVICE_INFO_BFLOAT16},
      {PI_EXT_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES,
       (ur_device_info_t)UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES},
      {PI_EXT_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES,
       (ur_device_info_t)UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES},
      {PI_EXT_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES,
       (ur_device_info_t)UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES},
      {PI_EXT_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES,
       (ur_device_info_t)UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES},
  };

  auto InfoType = InfoMapping.find(ParamName);
  if (InfoType == InfoMapping.end()) {
    return PI_ERROR_UNKNOWN;
  }

  size_t SizeInOut = ParamValueSize;
  auto hDevice = reinterpret_cast<ur_device_handle_t>(Device);
  HANDLE_ERRORS(urDeviceGetInfo(hDevice, InfoType->second, SizeInOut,
                                ParamValue, ParamValueSizeRet));

  ur2piInfoValue(InfoType->second, ParamValueSize, &SizeInOut, ParamValue);

  return PI_SUCCESS;
}

inline pi_result piDevicePartition(
    pi_device Device, const pi_device_partition_property *Properties,
    pi_uint32 NumEntries, pi_device *SubDevices, pi_uint32 *NumSubDevices) {

  if (!Properties || !Properties[0])
    return PI_ERROR_INVALID_VALUE;

  static std::unordered_map<pi_device_partition_property,
                            ur_device_partition_property_t>
      PropertyMap = {
          {PI_DEVICE_PARTITION_EQUALLY, UR_DEVICE_PARTITION_EQUALLY},
          {PI_DEVICE_PARTITION_BY_COUNTS, UR_DEVICE_PARTITION_BY_COUNTS},
          {PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
           UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN},
          {PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE,
           UR_EXT_DEVICE_PARTITION_PROPERTY_FLAG_BY_CSLICE},
      };

  auto PropertyIt = PropertyMap.find(Properties[0]);
  if (PropertyIt == PropertyMap.end()) {
    return PI_ERROR_UNKNOWN;
  }

  // Some partitioning types require a value
  auto Value = uint32_t(Properties[1]);
  if (PropertyIt->second == UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN) {
    static std::unordered_map<pi_device_affinity_domain,
                              ur_device_affinity_domain_flag_t>
        ValueMap = {
            {PI_DEVICE_AFFINITY_DOMAIN_NUMA,
             UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA},
            {PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE,
             UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE},
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
  ur_device_partition_property_t UrProperties[] = {
      ur_device_partition_property_t(PropertyIt->second), Value, 0};

  auto hDevice = reinterpret_cast<ur_device_handle_t>(Device);
  auto phSubDevices = reinterpret_cast<ur_device_handle_t *>(SubDevices);
  HANDLE_ERRORS(urDevicePartition(hDevice, UrProperties, NumEntries,
                                  phSubDevices, NumSubDevices));
  return PI_SUCCESS;
}
} // namespace pi2ur
