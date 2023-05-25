//===---------------- pi2ur.hpp - PI API to UR API  --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
#pragma once

#include "ur_api.h"
#include <sycl/detail/pi.h>
#include <ur/ur.hpp>

// Map of UR error codes to PI error codes
static pi_result ur2piResult(ur_result_t urResult) {
  if (urResult == UR_RESULT_SUCCESS)
    return PI_SUCCESS;

  switch (urResult) {
  case UR_RESULT_ERROR_UNKNOWN:
    return PI_ERROR_UNKNOWN;
  case UR_RESULT_ERROR_DEVICE_LOST:
    return PI_ERROR_DEVICE_NOT_FOUND;
  case UR_RESULT_ERROR_INVALID_OPERATION:
    return PI_ERROR_INVALID_OPERATION;
  case UR_RESULT_ERROR_INVALID_PLATFORM:
    return PI_ERROR_INVALID_PLATFORM;
  case UR_RESULT_ERROR_INVALID_ARGUMENT:
    return PI_ERROR_INVALID_ARG_VALUE;
  case UR_RESULT_ERROR_INVALID_VALUE:
    return PI_ERROR_INVALID_VALUE;
  case UR_RESULT_ERROR_INVALID_EVENT:
    return PI_ERROR_INVALID_EVENT;
  case UR_RESULT_ERROR_INVALID_BINARY:
    return PI_ERROR_INVALID_BINARY;
  case UR_RESULT_ERROR_INVALID_KERNEL_NAME:
    return PI_ERROR_INVALID_KERNEL_NAME;
  case UR_RESULT_ERROR_INVALID_FUNCTION_NAME:
    return PI_ERROR_BUILD_PROGRAM_FAILURE;
  case UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE:
    return PI_ERROR_INVALID_WORK_GROUP_SIZE;
  case UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    return PI_ERROR_OUT_OF_RESOURCES;
  case UR_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  case UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE:
    return PI_ERROR_BUILD_PROGRAM_FAILURE;
  case UR_RESULT_ERROR_UNINITIALIZED:
    return PI_ERROR_UNINITIALIZED;
  default:
    return PI_ERROR_UNKNOWN;
  };
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
  pi_result convert(std::function<TypePI(TypeUR)> Func) {
    *param_value_size_ret = sizeof(TypePI);

    // There is no value to convert.
    if (!param_value)
      return PI_SUCCESS;

    auto pValueUR = static_cast<TypeUR *>(param_value);
    auto pValuePI = static_cast<TypePI *>(param_value);

    // Cannot convert to a smaller storage type
    PI_ASSERT(sizeof(TypePI) >= sizeof(TypeUR), PI_ERROR_UNKNOWN);

    *pValuePI = Func(*pValueUR);
    return PI_SUCCESS;
  }

  // Convert the array (0-terminated) using a conversion map
  template <typename TypeUR, typename TypePI>
  pi_result convertArray(std::function<TypePI(TypeUR)> Func) {
    // Cannot convert to a smaller element storage type
    PI_ASSERT(sizeof(TypePI) >= sizeof(TypeUR), PI_ERROR_UNKNOWN);
    *param_value_size_ret *= sizeof(TypePI) / sizeof(TypeUR);

    // There is no value to convert. Adjust to a possibly bigger PI storage.
    if (!param_value)
      return PI_SUCCESS;

    PI_ASSERT(*param_value_size_ret % sizeof(TypePI) == 0, PI_ERROR_UNKNOWN);

    // Make a copy of the input UR array as we may possibly overwrite
    // following elements while converting previous ones (if extending).
    auto ValueUR = new char[*param_value_size_ret];
    auto pValueUR = reinterpret_cast<TypeUR *>(ValueUR);
    auto pValuePI = static_cast<TypePI *>(param_value);
    memcpy(pValueUR, param_value, *param_value_size_ret);

    while (pValueUR) {
      if (*pValueUR == 0) {
        *pValuePI = 0;
        break;
      }

      *pValuePI = Func(*pValueUR);
      ++pValuePI;
      ++pValueUR;
    }

    delete[] ValueUR;
    return PI_SUCCESS;
  }

  // Convert the bitset using a conversion map
  template <typename TypeUR, typename TypePI>
  pi_result convertBitSet(std::function<TypePI(TypeUR)> Func) {
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
      if (auto Res = convert(Func))
        return Res;
      Out |= *pValuePI;
    }
    *pValuePI = TypePI(Out);
    return PI_SUCCESS;
  }
};

// Translate UR platform info values to PI info values
inline pi_result ur2piPlatformInfoValue(ur_platform_info_t ParamName,
                                        size_t ParamValueSizePI,
                                        size_t *ParamValueSizeUR,
                                        void *ParamValue) {

  ConvertHelper Value(ParamValueSizePI, ParamValue, ParamValueSizeUR);

  switch (ParamName) {
  case UR_PLATFORM_INFO_EXTENSIONS:
  case UR_PLATFORM_INFO_NAME:
  case UR_PLATFORM_INFO_PROFILE:
  case UR_PLATFORM_INFO_VENDOR_NAME:
  case UR_PLATFORM_INFO_VERSION:
    // These ones do not need ur2pi translations
    break;
  case UR_PLATFORM_INFO_BACKEND: {
    auto ConvertFunc = [](ur_platform_backend_t UrValue) {
      switch (UrValue) {
      case UR_PLATFORM_BACKEND_UNKNOWN:
        return PI_EXT_PLATFORM_BACKEND_UNKNOWN;
      case UR_PLATFORM_BACKEND_LEVEL_ZERO:
        return PI_EXT_PLATFORM_BACKEND_LEVEL_ZERO;
      case UR_PLATFORM_BACKEND_OPENCL:
        return PI_EXT_PLATFORM_BACKEND_OPENCL;
      case UR_PLATFORM_BACKEND_CUDA:
        return PI_EXT_PLATFORM_BACKEND_CUDA;
      case UR_PLATFORM_BACKEND_HIP:
        return PI_EXT_PLATFORM_BACKEND_CUDA;
      default:
        die("UR_PLATFORM_INFO_BACKEND: unhandled value");
      }
    };
    return Value.convert<ur_platform_backend_t, pi_platform_backend>(
        ConvertFunc);
  }
  default:
    return PI_ERROR_UNKNOWN;
  }

  if (ParamValueSizePI && ParamValueSizePI != *ParamValueSizeUR) {
    fprintf(stderr, "UR PlatformInfoType=%d PI=%d but UR=%d\n", ParamName,
            (int)ParamValueSizePI, (int)*ParamValueSizeUR);
    die("ur2piPlatformInfoValue: size mismatch");
  }
  return PI_SUCCESS;
}

// Translate UR device info values to PI info values
inline pi_result ur2piDeviceInfoValue(ur_device_info_t ParamName,
                                      size_t ParamValueSizePI,
                                      size_t *ParamValueSizeUR,
                                      void *ParamValue) {

  ConvertHelper Value(ParamValueSizePI, ParamValue, ParamValueSizeUR);

  if (ParamName == UR_DEVICE_INFO_TYPE) {
    auto ConvertFunc = [](ur_device_type_t UrValue) {
      switch (UrValue) {
      case UR_DEVICE_TYPE_CPU:
        return PI_DEVICE_TYPE_CPU;
      case UR_DEVICE_TYPE_GPU:
        return PI_DEVICE_TYPE_GPU;
      case UR_DEVICE_TYPE_FPGA:
        return PI_DEVICE_TYPE_ACC;
      default:
        die("UR_DEVICE_INFO_TYPE: unhandled value");
      }
    };
    return Value.convert<ur_device_type_t, pi_device_type>(ConvertFunc);
  } else if (ParamName == UR_DEVICE_INFO_QUEUE_PROPERTIES) {
    auto ConvertFunc = [](ur_queue_flag_t UrValue) {
      switch (UrValue) {
      case UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE:
        return PI_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;
      case UR_QUEUE_FLAG_PROFILING_ENABLE:
        return PI_QUEUE_FLAG_PROFILING_ENABLE;
      case UR_QUEUE_FLAG_ON_DEVICE:
        return PI_QUEUE_FLAG_ON_DEVICE;
      case UR_QUEUE_FLAG_ON_DEVICE_DEFAULT:
        return PI_QUEUE_FLAG_ON_DEVICE_DEFAULT;
      default:
        die("UR_DEVICE_INFO_QUEUE_PROPERTIES: unhandled value");
      }
    };
    return Value.convertBitSet<ur_queue_flag_t, pi_queue_properties>(
        ConvertFunc);
  } else if (ParamName == UR_DEVICE_INFO_EXECUTION_CAPABILITIES) {
    auto ConvertFunc = [](ur_device_exec_capability_flag_t UrValue) {
      switch (UrValue) {
      case UR_DEVICE_EXEC_CAPABILITY_FLAG_KERNEL:
        return PI_DEVICE_EXEC_CAPABILITIES_KERNEL;
      case UR_DEVICE_EXEC_CAPABILITY_FLAG_NATIVE_KERNEL:
        return PI_DEVICE_EXEC_CAPABILITIES_NATIVE_KERNEL;
      default:
        die("UR_DEVICE_INFO_EXECUTION_CAPABILITIES: unhandled value");
      }
    };
    return Value
        .convertBitSet<ur_device_exec_capability_flag_t, pi_queue_properties>(
            ConvertFunc);
  } else if (ParamName == UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN) {
    auto ConvertFunc = [](ur_device_affinity_domain_flag_t UrValue) {
      switch (UrValue) {
      case UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA:
        return PI_DEVICE_AFFINITY_DOMAIN_NUMA;
      case UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE:
        return PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE;
      default:
        die("UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: unhandled value");
      }
    };
    return Value.convertBitSet<ur_device_affinity_domain_flag_t,
                               pi_device_affinity_domain>(ConvertFunc);
  } else if (ParamName == UR_DEVICE_INFO_PARTITION_TYPE) {
    auto ConvertFunc = [](ur_device_partition_property_t UrValue) {
      switch (UrValue) {
      case UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
        return PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
      case UR_DEVICE_PARTITION_BY_CSLICE:
        return PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE;
      case (ur_device_partition_property_t)
          UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE:
        return (pi_device_partition_property)
            PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE;
      default:
        die("UR_DEVICE_INFO_PARTITION_TYPE: unhandled value");
      }
    };
    return Value.convertArray<ur_device_partition_property_t,
                              pi_device_partition_property>(ConvertFunc);
  } else if (ParamName == UR_DEVICE_INFO_PARTITION_PROPERTIES) {
    auto ConvertFunc = [](ur_device_partition_property_t UrValue) {
      switch (UrValue) {
      case UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
        return PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
      case UR_DEVICE_PARTITION_BY_CSLICE:
        return PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE;
      default:
        die("UR_DEVICE_INFO_PARTITION_PROPERTIES: unhandled value");
      }
    };
    return Value.convertArray<ur_device_partition_property_t,
                              pi_device_partition_property>(ConvertFunc);
  } else if (ParamName == UR_DEVICE_INFO_LOCAL_MEM_TYPE) {
    auto ConvertFunc = [](ur_device_local_mem_type_t UrValue) {
      switch (UrValue) {
      case UR_DEVICE_LOCAL_MEM_TYPE_LOCAL:
        return PI_DEVICE_LOCAL_MEM_TYPE_LOCAL;
      case UR_DEVICE_LOCAL_MEM_TYPE_GLOBAL:
        return PI_DEVICE_LOCAL_MEM_TYPE_GLOBAL;
      default:
        die("UR_DEVICE_INFO_LOCAL_MEM_TYPE: unhandled value");
      }
    };
    return Value.convert<ur_device_local_mem_type_t, pi_device_local_mem_type>(
        ConvertFunc);
  } else if (ParamName == UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES ||
             ParamName == UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES) {
    auto ConvertFunc = [](ur_memory_order_capability_flag_t UrValue) {
      switch (UrValue) {
      case UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED:
        return PI_MEMORY_ORDER_RELAXED;
      case UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE:
        return PI_MEMORY_ORDER_ACQUIRE;
      case UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE:
        return PI_MEMORY_ORDER_RELEASE;
      case UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL:
        return PI_MEMORY_ORDER_ACQ_REL;
      case UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST:
        return PI_MEMORY_ORDER_SEQ_CST;
      default:
        die("UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: unhandled "
            "value");
      }
    };
    return Value.convertBitSet<ur_memory_order_capability_flag_t,
                               pi_memory_order_capabilities>(ConvertFunc);
  } else if (ParamName == UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES ||
             ParamName == UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES) {
    auto ConvertFunc = [](ur_memory_scope_capability_flag_t UrValue) {
      switch (UrValue) {
      case UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM:
        return PI_MEMORY_SCOPE_WORK_ITEM;
      case UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP:
        return PI_MEMORY_SCOPE_SUB_GROUP;
      case UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP:
        return PI_MEMORY_SCOPE_WORK_GROUP;
      case UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE:
        return PI_MEMORY_SCOPE_DEVICE;
      case UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM:
        return PI_MEMORY_SCOPE_SYSTEM;
      default:
        die("UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES: unhandled "
            "value");
      }
    };
    return Value.convertBitSet<ur_memory_scope_capability_flag_t,
                               pi_memory_scope_capabilities>(ConvertFunc);
  } else {
    // TODO: what else needs a UR-PI translation?
  }

  if (ParamValueSizePI && ParamValueSizePI != *ParamValueSizeUR) {
    fprintf(stderr, "UR DeviceInfoType=%d PI=%d but UR=%d\n", ParamName,
            (int)ParamValueSizePI, (int)*ParamValueSizeUR);
    die("ur2piDeviceInfoValue: size mismatch");
  }
  return PI_SUCCESS;
}

namespace pi2ur {
inline pi_result piPlatformsGet(pi_uint32 num_entries, pi_platform *platforms,
                                pi_uint32 *num_platforms) {

  urInit(0);
  uint32_t Count = num_entries;
  auto phPlatforms = reinterpret_cast<ur_platform_handle_t *>(platforms);
  HANDLE_ERRORS(urPlatformGet(Count, phPlatforms, num_platforms));
  return PI_SUCCESS;
}

inline pi_result piPlatformGetInfo(pi_platform platform,
                                   pi_platform_info ParamName,
                                   size_t ParamValueSize, void *ParamValue,
                                   size_t *ParamValueSizeRet) {
  ur_platform_info_t InfoType;
  switch (ParamName) {
  case PI_PLATFORM_INFO_EXTENSIONS:
    InfoType = UR_PLATFORM_INFO_NAME;
    break;
  case PI_PLATFORM_INFO_NAME:
    InfoType = UR_PLATFORM_INFO_NAME;
    break;
  case PI_PLATFORM_INFO_PROFILE:
    InfoType = UR_PLATFORM_INFO_PROFILE;
    break;
  case PI_PLATFORM_INFO_VENDOR:
    InfoType = UR_PLATFORM_INFO_VENDOR_NAME;
    break;
  case PI_PLATFORM_INFO_VERSION:
    InfoType = UR_PLATFORM_INFO_VERSION;
    break;
  case PI_EXT_PLATFORM_INFO_BACKEND:
    InfoType = UR_PLATFORM_INFO_BACKEND;
    break;
  default:
    return PI_ERROR_UNKNOWN;
  }

  size_t SizeInOut = ParamValueSize;
  auto hPlatform = reinterpret_cast<ur_platform_handle_t>(platform);
  HANDLE_ERRORS(urPlatformGetInfo(hPlatform, InfoType, SizeInOut, ParamValue,
                                  ParamValueSizeRet));

  ur2piPlatformInfoValue(InfoType, ParamValueSize, &SizeInOut, ParamValue);
  return PI_SUCCESS;
}

inline pi_result piDevicesGet(pi_platform Platform, pi_device_type DeviceType,
                              pi_uint32 NumEntries, pi_device *Devices,
                              pi_uint32 *NumDevices) {
  ur_device_type_t Type;
  switch (DeviceType) {
  case PI_DEVICE_TYPE_ALL:
    Type = UR_DEVICE_TYPE_ALL;
    break;
  case PI_DEVICE_TYPE_GPU:
    Type = UR_DEVICE_TYPE_GPU;
    break;
  case PI_DEVICE_TYPE_CPU:
    Type = UR_DEVICE_TYPE_CPU;
    break;
  case PI_DEVICE_TYPE_ACC:
    Type = UR_DEVICE_TYPE_FPGA;
    break;
  default:
    return PI_ERROR_UNKNOWN;
  }

  uint32_t Count = NumEntries;
  auto hPlatform = reinterpret_cast<ur_platform_handle_t>(Platform);
  auto phDevices = reinterpret_cast<ur_device_handle_t *>(Devices);
  HANDLE_ERRORS(urDeviceGet(hPlatform, Type, Count, phDevices, NumDevices));
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

  ur_device_info_t InfoType;
  switch (ParamName) {
  case PI_DEVICE_INFO_TYPE:
    InfoType = UR_DEVICE_INFO_TYPE;
    break;
  case PI_DEVICE_INFO_PARENT_DEVICE:
    InfoType = UR_DEVICE_INFO_PARENT_DEVICE;
    break;
  case PI_DEVICE_INFO_PLATFORM:
    InfoType = UR_DEVICE_INFO_PLATFORM;
    break;
  case PI_DEVICE_INFO_VENDOR_ID:
    InfoType = UR_DEVICE_INFO_VENDOR_ID;
    break;
  case PI_DEVICE_INFO_UUID:
    InfoType = UR_DEVICE_INFO_UUID;
    break;
  case PI_DEVICE_INFO_ATOMIC_64:
    InfoType = UR_DEVICE_INFO_ATOMIC_64;
    break;
  case PI_DEVICE_INFO_EXTENSIONS:
    InfoType = UR_DEVICE_INFO_EXTENSIONS;
    break;
  case PI_DEVICE_INFO_NAME:
    InfoType = UR_DEVICE_INFO_NAME;
    break;
  case PI_DEVICE_INFO_COMPILER_AVAILABLE:
    InfoType = UR_DEVICE_INFO_COMPILER_AVAILABLE;
    break;
  case PI_DEVICE_INFO_LINKER_AVAILABLE:
    InfoType = UR_DEVICE_INFO_LINKER_AVAILABLE;
    break;
  case PI_DEVICE_INFO_MAX_COMPUTE_UNITS:
    InfoType = UR_DEVICE_INFO_MAX_COMPUTE_UNITS;
    break;
  case PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS:
    InfoType = UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS;
    break;
  case PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
    InfoType = UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE;
    break;
  case PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES:
    InfoType = UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES;
    break;
  case PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY:
    InfoType = UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY;
    break;
  case PI_DEVICE_INFO_ADDRESS_BITS:
    InfoType = UR_DEVICE_INFO_ADDRESS_BITS;
    break;
  case PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE:
    InfoType = UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE;
    break;
  case PI_DEVICE_INFO_GLOBAL_MEM_SIZE:
    InfoType = UR_DEVICE_INFO_GLOBAL_MEM_SIZE;
    break;
  case PI_DEVICE_INFO_LOCAL_MEM_SIZE:
    InfoType = UR_DEVICE_INFO_LOCAL_MEM_SIZE;
    break;
  case PI_DEVICE_INFO_IMAGE_SUPPORT:
    InfoType = UR_DEVICE_INFO_IMAGE_SUPPORTED;
    break;
  case PI_DEVICE_INFO_HOST_UNIFIED_MEMORY:
    InfoType = UR_DEVICE_INFO_HOST_UNIFIED_MEMORY;
    break;
  case PI_DEVICE_INFO_AVAILABLE:
    InfoType = UR_DEVICE_INFO_AVAILABLE;
    break;
  case PI_DEVICE_INFO_VENDOR:
    InfoType = UR_DEVICE_INFO_VENDOR;
    break;
  case PI_DEVICE_INFO_DRIVER_VERSION:
    InfoType = UR_DEVICE_INFO_DRIVER_VERSION;
    break;
  case PI_DEVICE_INFO_VERSION:
    InfoType = UR_DEVICE_INFO_VERSION;
    break;
  case PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES:
    InfoType = UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES;
    break;
  case PI_DEVICE_INFO_REFERENCE_COUNT:
    InfoType = UR_DEVICE_INFO_REFERENCE_COUNT;
    break;
  case PI_DEVICE_INFO_PARTITION_PROPERTIES:
    InfoType = UR_DEVICE_INFO_PARTITION_PROPERTIES;
    break;
  case PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN:
    InfoType = UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN;
    break;
  case PI_DEVICE_INFO_PARTITION_TYPE:
    InfoType = UR_DEVICE_INFO_PARTITION_TYPE;
    break;
  case PI_DEVICE_INFO_OPENCL_C_VERSION:
    InfoType = UR_EXT_DEVICE_INFO_OPENCL_C_VERSION;
    break;
  case PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC:
    InfoType = UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC;
    break;
  case PI_DEVICE_INFO_PRINTF_BUFFER_SIZE:
    InfoType = UR_DEVICE_INFO_PRINTF_BUFFER_SIZE;
    break;
  case PI_DEVICE_INFO_PROFILE:
    InfoType = UR_DEVICE_INFO_PROFILE;
    break;
  case PI_DEVICE_INFO_BUILT_IN_KERNELS:
    InfoType = UR_DEVICE_INFO_BUILT_IN_KERNELS;
    break;
  case PI_DEVICE_INFO_QUEUE_PROPERTIES:
    InfoType = UR_DEVICE_INFO_QUEUE_PROPERTIES;
    break;
  case PI_DEVICE_INFO_EXECUTION_CAPABILITIES:
    InfoType = UR_DEVICE_INFO_EXECUTION_CAPABILITIES;
    break;
  case PI_DEVICE_INFO_ENDIAN_LITTLE:
    InfoType = UR_DEVICE_INFO_ENDIAN_LITTLE;
    break;
  case PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT:
    InfoType = UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT;
    break;
  case PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION:
    InfoType = UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION;
    break;
  case PI_DEVICE_INFO_LOCAL_MEM_TYPE:
    InfoType = UR_DEVICE_INFO_LOCAL_MEM_TYPE;
    break;
  case PI_DEVICE_INFO_MAX_CONSTANT_ARGS:
    InfoType = UR_DEVICE_INFO_MAX_CONSTANT_ARGS;
    break;
  case PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE:
    InfoType = UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE;
    break;
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE:
    InfoType = UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE;
    break;
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE:
    InfoType = UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE;
    break;
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE:
    InfoType = UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE;
    break;
  case PI_DEVICE_INFO_MAX_PARAMETER_SIZE:
    InfoType = UR_DEVICE_INFO_MAX_PARAMETER_SIZE;
    break;
  case PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN:
    InfoType = UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN;
    break;
  case PI_DEVICE_INFO_MAX_SAMPLERS:
    InfoType = UR_DEVICE_INFO_MAX_SAMPLERS;
    break;
  case PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS:
    InfoType = UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS;
    break;
  case PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS:
    InfoType = UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS;
    break;
  case PI_DEVICE_INFO_SINGLE_FP_CONFIG:
    InfoType = UR_DEVICE_INFO_SINGLE_FP_CONFIG;
    break;
  case PI_DEVICE_INFO_HALF_FP_CONFIG:
    InfoType = UR_DEVICE_INFO_HALF_FP_CONFIG;
    break;
  case PI_DEVICE_INFO_DOUBLE_FP_CONFIG:
    InfoType = UR_DEVICE_INFO_DOUBLE_FP_CONFIG;
    break;
  case PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
    InfoType = UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH;
    break;
  case PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
    InfoType = UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT;
    break;
  case PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH:
    InfoType = UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH;
    break;
  case PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT:
    InfoType = UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT;
    break;
  case PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH:
    InfoType = UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH;
    break;
  case PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE:
    InfoType = UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE;
    break;
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR:
    InfoType = UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR;
    break;
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR:
    InfoType = UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR;
    break;
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT:
    InfoType = UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT;
    break;
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT:
    InfoType = UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT;
    break;
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT:
    InfoType = UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT;
    break;
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT:
    InfoType = UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT;
    break;
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG:
    InfoType = UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG;
    break;
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG:
    InfoType = UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG;
    break;
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT:
    InfoType = UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT;
    break;
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT:
    InfoType = UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT;
    break;
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE:
    InfoType = UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE;
    break;
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE:
    InfoType = UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE;
    break;
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF:
    InfoType = UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF;
    break;
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF:
    InfoType = UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF;
    break;
  case PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS:
    InfoType = UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS;
    break;
  case PI_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS:
    InfoType = UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS;
    break;
  case PI_DEVICE_INFO_SUB_GROUP_SIZES_INTEL:
    InfoType = UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL;
    break;
  case PI_DEVICE_INFO_IL_VERSION:
    InfoType = UR_DEVICE_INFO_IL_VERSION;
    break;
  case PI_DEVICE_INFO_USM_HOST_SUPPORT:
    InfoType = UR_DEVICE_INFO_USM_HOST_SUPPORT;
    break;
  case PI_DEVICE_INFO_USM_DEVICE_SUPPORT:
    InfoType = UR_DEVICE_INFO_USM_DEVICE_SUPPORT;
    break;
  case PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
    InfoType = UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT;
    break;
  case PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT:
    InfoType = UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT;
    break;
  case PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT:
    InfoType = UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT;
    break;
  case PI_DEVICE_INFO_PCI_ADDRESS:
    InfoType = UR_DEVICE_INFO_PCI_ADDRESS;
    break;
  case PI_DEVICE_INFO_GPU_EU_COUNT:
    InfoType = UR_DEVICE_INFO_GPU_EU_COUNT;
    break;
  case PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
    InfoType = UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH;
    break;
  case PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
    InfoType = UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE;
    break;
  case PI_DEVICE_INFO_BUILD_ON_SUBDEVICE:
    InfoType = (ur_device_info_t)UR_EXT_DEVICE_INFO_BUILD_ON_SUBDEVICE;
    break;
  case PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_3D:
    InfoType = (ur_device_info_t)UR_EXT_DEVICE_INFO_MAX_WORK_GROUPS_3D;
    break;
  case PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE:
    InfoType = (ur_device_info_t)UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE;
    break;
  case PI_DEVICE_INFO_DEVICE_ID:
    InfoType = (ur_device_info_t)UR_DEVICE_INFO_DEVICE_ID;
    break;
  case PI_EXT_INTEL_DEVICE_INFO_FREE_MEMORY:
    InfoType = (ur_device_info_t)UR_EXT_DEVICE_INFO_FREE_MEMORY;
    break;
  case PI_EXT_INTEL_DEVICE_INFO_MEMORY_CLOCK_RATE:
    InfoType = (ur_device_info_t)UR_DEVICE_INFO_MEMORY_CLOCK_RATE;
    break;
  case PI_EXT_INTEL_DEVICE_INFO_MEMORY_BUS_WIDTH:
    InfoType = (ur_device_info_t)UR_EXT_DEVICE_INFO_MEMORY_BUS_WIDTH;
    break;
  case PI_EXT_INTEL_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES:
    InfoType = (ur_device_info_t)UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES;
    break;
  case PI_DEVICE_INFO_GPU_SLICES:
    InfoType = (ur_device_info_t)UR_EXT_DEVICE_INFO_GPU_SLICES;
    break;
  case PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
    InfoType = (ur_device_info_t)UR_EXT_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE;
    break;
  case PI_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
    InfoType = (ur_device_info_t)UR_EXT_DEVICE_INFO_GPU_HW_THREADS_PER_EU;
    break;
  case PI_DEVICE_INFO_MAX_MEM_BANDWIDTH:
    InfoType = (ur_device_info_t)UR_EXT_DEVICE_INFO_MAX_MEM_BANDWIDTH;
    break;
  case PI_EXT_ONEAPI_DEVICE_INFO_BFLOAT16_MATH_FUNCTIONS:
    InfoType = (ur_device_info_t)UR_DEVICE_INFO_BFLOAT16;
    break;
  case PI_EXT_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES:
    InfoType =
        (ur_device_info_t)UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES;
    break;
  case PI_EXT_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
    InfoType =
        (ur_device_info_t)UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES;
    break;
  case PI_EXT_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES:
    InfoType = (ur_device_info_t)UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES;
    break;
  case PI_EXT_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES:
    InfoType = (ur_device_info_t)UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES;
    break;
  case PI_EXT_INTEL_DEVICE_INFO_MEM_CHANNEL_SUPPORT:
    InfoType = (ur_device_info_t)UR_EXT_DEVICE_INFO_MEM_CHANNEL_SUPPORT;
    break;
  case PI_DEVICE_INFO_IMAGE_SRGB:
    InfoType = (ur_device_info_t)UR_DEVICE_INFO_IMAGE_SRGB;
    break;
  case PI_DEVICE_INFO_BACKEND_VERSION: {
    // TODO: return some meaningful for backend_version below
    ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
    return ReturnValue("");
  }
  default:
    return PI_ERROR_UNKNOWN;
  };

  size_t SizeInOut = ParamValueSize;
  auto hDevice = reinterpret_cast<ur_device_handle_t>(Device);
  HANDLE_ERRORS(urDeviceGetInfo(hDevice, InfoType, SizeInOut, ParamValue,
                                ParamValueSizeRet));

  ur2piDeviceInfoValue(InfoType, ParamValueSize, &SizeInOut, ParamValue);

  return PI_SUCCESS;
}

inline pi_result piDevicePartition(
    pi_device Device, const pi_device_partition_property *Properties,
    pi_uint32 NumEntries, pi_device *SubDevices, pi_uint32 *NumSubDevices) {

  if (!Properties || !Properties[0])
    return PI_ERROR_INVALID_VALUE;

  ur_device_partition_property_t Property;
  switch (Properties[0]) {
  case PI_DEVICE_PARTITION_EQUALLY:
    Property = UR_DEVICE_PARTITION_EQUALLY;
    break;
  case PI_DEVICE_PARTITION_BY_COUNTS:
    Property = UR_DEVICE_PARTITION_BY_COUNTS;
    break;
  case PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
    Property = UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
    break;
  case PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE:
    Property = UR_DEVICE_PARTITION_BY_CSLICE;
    break;
  default:
    return PI_ERROR_UNKNOWN;
  }

  // Some partitioning types require a value
  auto Value = uint32_t(Properties[1]);
  if (Property == UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN) {
    switch (Properties[1]) {
    case PI_DEVICE_AFFINITY_DOMAIN_NUMA:
      Value = UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA;
      break;
    case PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE:
      Value = UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE;
      break;
    default:
      return PI_ERROR_UNKNOWN;
    }
  }

  // Translate partitioning properties from PI-way
  // (array of uintptr_t values) to UR-way
  // (array of {uint32_t, uint32_t} pairs)
  //
  // TODO: correctly terminate the UR properties, see:
  // https://github.com/oneapi-src/unified-runtime/issues/183
  //
  ur_device_partition_property_t UrProperties[] = {
      ur_device_partition_property_t(Property), Value, 0};

  auto hDevice = reinterpret_cast<ur_device_handle_t>(Device);
  auto phSubDevices = reinterpret_cast<ur_device_handle_t *>(SubDevices);
  HANDLE_ERRORS(urDevicePartition(hDevice, UrProperties, NumEntries,
                                  phSubDevices, NumSubDevices));
  return PI_SUCCESS;
}
} // namespace pi2ur
