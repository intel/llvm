//===---------------- pi2ur.hpp - PI API to UR API  --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
#pragma once

#include "ur_api.h"
#include <cstdarg>
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

// Helper for one-liner validation
#define PI_ASSERT(condition, error)                                            \
  if (!(condition))                                                            \
    return error;

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

inline pi_result ur2piSamplerInfoValue(ur_sampler_info_t ParamName,
                                       size_t ParamValueSizePI,
                                       size_t *ParamValueSizeUR,
                                       void *ParamValue) {

  ConvertHelper Value(ParamValueSizePI, ParamValue, ParamValueSizeUR);
  switch (ParamName) {
  case UR_SAMPLER_INFO_ADDRESSING_MODE: {
    auto ConvertFunc = [](ur_sampler_addressing_mode_t UrValue) {
      switch (UrValue) {
      case UR_SAMPLER_ADDRESSING_MODE_CLAMP:
        return PI_SAMPLER_ADDRESSING_MODE_CLAMP;
      case UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE:
        return PI_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE;
      case UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT:
        return PI_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT;
      case UR_SAMPLER_ADDRESSING_MODE_NONE:
        return PI_SAMPLER_ADDRESSING_MODE_NONE;
      case UR_SAMPLER_ADDRESSING_MODE_REPEAT:
        return PI_SAMPLER_ADDRESSING_MODE_REPEAT;

      default:
        die("UR_SAMPLER_ADDRESSING_MODE_TYPE: unhandled value");
      }
    };
    return Value
        .convert<ur_sampler_addressing_mode_t, pi_sampler_addressing_mode>(
            ConvertFunc);
  }
  case UR_SAMPLER_INFO_FILTER_MODE: {
    auto ConvertFunc = [](ur_sampler_filter_mode_t UrValue) {
      switch (UrValue) {
      case UR_SAMPLER_FILTER_MODE_LINEAR:
        return PI_SAMPLER_FILTER_MODE_LINEAR;
      case UR_SAMPLER_FILTER_MODE_NEAREST:
        return PI_SAMPLER_FILTER_MODE_NEAREST;
      default:
        die("UR_SAMPLER_FILTER_MODE: unhandled value");
      }
    };
    return Value.convert<ur_sampler_filter_mode_t, pi_sampler_filter_mode>(
        ConvertFunc);
  }
  default:
    return PI_SUCCESS;
  }
}

// Translate UR device info values to PI info values
inline pi_result ur2piUSMAllocInfoValue(ur_usm_alloc_info_t ParamName,
                                        size_t ParamValueSizePI,
                                        size_t *ParamValueSizeUR,
                                        void *ParamValue) {
  ConvertHelper Value(ParamValueSizePI, ParamValue, ParamValueSizeUR);

  if (ParamName == UR_USM_ALLOC_INFO_TYPE) {
    auto ConvertFunc = [](ur_usm_type_t UrValue) {
      switch (UrValue) {
      case UR_USM_TYPE_UNKNOWN:
        return PI_MEM_TYPE_UNKNOWN;
      case UR_USM_TYPE_HOST:
        return PI_MEM_TYPE_HOST;
      case UR_USM_TYPE_DEVICE:
        return PI_MEM_TYPE_DEVICE;
      case UR_USM_TYPE_SHARED:
        return PI_MEM_TYPE_SHARED;
      default:
        die("UR_USM_ALLOC_INFO_TYPE: unhandled value");
      }
    };
    return Value.convert<ur_usm_type_t, pi_usm_type>(ConvertFunc);
  }

  return PI_SUCCESS;
}

// Handle mismatched PI and UR type return sizes for info queries
inline pi_result fixupInfoValueTypes(size_t ParamValueSizeRetUR,
                                     size_t *ParamValueSizeRetPI,
                                     size_t ParamValueSize, void *ParamValue) {
  if (ParamValueSizeRetUR == 1 && ParamValueSize == 4) {
    // extend bool to pi_bool (uint32_t)
    if (ParamValue) {
      auto *ValIn = static_cast<bool *>(ParamValue);
      auto *ValOut = static_cast<pi_bool *>(ParamValue);
      *ValOut = static_cast<pi_bool>(*ValIn);
    }
    if (ParamValueSizeRetPI) {
      *ParamValueSizeRetPI = sizeof(pi_bool);
    }
  }

  return PI_SUCCESS;
}

inline ur_result_t
mapPIMetadataToUR(const pi_device_binary_property *pi_metadata,
                  ur_program_metadata_t *ur_metadata) {
  ur_metadata->pName = (*pi_metadata)->Name;
  ur_metadata->size = (*pi_metadata)->ValSize;
  switch ((*pi_metadata)->Type) {
  case PI_PROPERTY_TYPE_UINT32:
    ur_metadata->type = UR_PROGRAM_METADATA_TYPE_UINT32;
    ur_metadata->value.data32 = (*pi_metadata)->ValSize;
    return UR_RESULT_SUCCESS;
  case PI_PROPERTY_TYPE_BYTE_ARRAY:
    ur_metadata->type = UR_PROGRAM_METADATA_TYPE_BYTE_ARRAY;
    ur_metadata->value.pData = (*pi_metadata)->ValAddr;
    return UR_RESULT_SUCCESS;
  case PI_PROPERTY_TYPE_STRING:
    ur_metadata->type = UR_PROGRAM_METADATA_TYPE_STRING;
    ur_metadata->value.pString =
        reinterpret_cast<char *>((*pi_metadata)->ValAddr);
    return UR_RESULT_SUCCESS;
  default:
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
}

namespace pi2ur {

inline pi_result piTearDown(void *PluginParameter) {
  std::ignore = PluginParameter;
  // TODO: Dont check for errors in urTearDown, since
  // when using Level Zero plugin, the second urTearDown
  // will fail as ur_loader.so has already been unloaded,
  urTearDown(nullptr);
  return PI_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// Platform
inline pi_result piPlatformsGet(pi_uint32 NumEntries, pi_platform *Platforms,
                                pi_uint32 *NumPlatforms) {

  urInit(0);
  auto phPlatforms = reinterpret_cast<ur_platform_handle_t *>(Platforms);
  HANDLE_ERRORS(urPlatformGet(NumEntries, phPlatforms, NumPlatforms));
  return PI_SUCCESS;
}

inline pi_result piextPlatformGetNativeHandle(pi_platform Platform,
                                              pi_native_handle *NativeHandle) {

  PI_ASSERT(Platform, PI_ERROR_INVALID_PLATFORM);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  auto UrPlatform = reinterpret_cast<ur_platform_handle_t>(Platform);

  ur_native_handle_t UrNativeHandle{};
  HANDLE_ERRORS(urPlatformGetNativeHandle(UrPlatform, &UrNativeHandle));

  *NativeHandle = reinterpret_cast<pi_native_handle>(UrNativeHandle);

  return PI_SUCCESS;
}

inline pi_result
piextPlatformCreateWithNativeHandle(pi_native_handle NativeHandle,
                                    pi_platform *Platform) {

  PI_ASSERT(Platform, PI_ERROR_INVALID_PLATFORM);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  ur_platform_handle_t UrPlatform{};
  ur_native_handle_t UrNativeHandle =
      reinterpret_cast<ur_native_handle_t>(NativeHandle);
  ur_platform_native_properties_t UrProperties{};
  urPlatformCreateWithNativeHandle(UrNativeHandle, &UrProperties, &UrPlatform);

  *Platform = reinterpret_cast<pi_platform>(UrPlatform);

  return PI_SUCCESS;
}

inline pi_result piPlatformGetInfo(pi_platform Platform,
                                   pi_platform_info ParamName,
                                   size_t ParamValueSize, void *ParamValue,
                                   size_t *ParamValueSizeRet) {

  PI_ASSERT(Platform, PI_ERROR_INVALID_PLATFORM);

  ur_platform_info_t UrParamName = {};
  switch (ParamName) {
  case PI_PLATFORM_INFO_EXTENSIONS: {
    UrParamName = UR_PLATFORM_INFO_EXTENSIONS;
    break;
  }
  case PI_PLATFORM_INFO_NAME: {
    UrParamName = UR_PLATFORM_INFO_NAME;
    break;
  }
  case PI_PLATFORM_INFO_PROFILE: {
    UrParamName = UR_PLATFORM_INFO_PROFILE;
    break;
  }
  case PI_PLATFORM_INFO_VENDOR: {
    UrParamName = UR_PLATFORM_INFO_VENDOR_NAME;
    break;
  }
  case PI_PLATFORM_INFO_VERSION: {
    UrParamName = UR_PLATFORM_INFO_VERSION;
    break;
  }
  case PI_EXT_PLATFORM_INFO_BACKEND: {
    UrParamName = UR_PLATFORM_INFO_BACKEND;
    break;
  }
  default:
    die("urGetContextInfo: unsuppported ParamName.");
  }

  size_t UrParamValueSizeRet;
  auto UrPlatform = reinterpret_cast<ur_platform_handle_t>(Platform);
  HANDLE_ERRORS(urPlatformGetInfo(UrPlatform, UrParamName, ParamValueSize,
                                  ParamValue, &UrParamValueSizeRet));

  if (ParamValueSizeRet) {
    *ParamValueSizeRet = UrParamValueSizeRet;
  }
  ur2piPlatformInfoValue(UrParamName, ParamValueSize, &ParamValueSize,
                         ParamValue);
  fixupInfoValueTypes(UrParamValueSizeRet, ParamValueSizeRet, ParamValueSize,
                      ParamValue);

  return PI_SUCCESS;
}

inline pi_result piextPluginGetOpaqueData(void *opaque_data_param,
                                          void **opaque_data_return) {
  (void)opaque_data_param;
  (void)opaque_data_return;
  return PI_ERROR_UNKNOWN;
}

// Returns plugin specific backend option.
// Current support is only for optimization options.
// Return '-ze-opt-disable' for frontend_option = -O0.
// Return '-ze-opt-level=1' for frontend_option = -O1 or -O2.
// Return '-ze-opt-level=2' for frontend_option = -O3.
inline pi_result piPluginGetBackendOption(pi_platform Platform,
                                          const char *FrontendOption,
                                          const char **PlatformOption) {

  auto UrPlatform = reinterpret_cast<ur_platform_handle_t>(Platform);
  HANDLE_ERRORS(
      urPlatformGetBackendOption(UrPlatform, FrontendOption, PlatformOption));

  return PI_SUCCESS;
}

// Platform
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Device
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

  PI_ASSERT(Platform, PI_ERROR_INVALID_PLATFORM);

  auto UrPlatform = reinterpret_cast<ur_platform_handle_t>(Platform);
  auto UrDevices = reinterpret_cast<ur_device_handle_t *>(Devices);
  HANDLE_ERRORS(
      urDeviceGet(UrPlatform, Type, NumEntries, UrDevices, NumDevices));

  return PI_SUCCESS;
}

inline pi_result piDeviceRetain(pi_device Device) {
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  HANDLE_ERRORS(urDeviceRetain(UrDevice));
  return PI_SUCCESS;
}

inline pi_result piDeviceRelease(pi_device Device) {
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  HANDLE_ERRORS(urDeviceRelease(UrDevice));
  return PI_SUCCESS;
}

inline pi_result piPluginGetLastError(char **message) {
  std::ignore = message;
  return PI_SUCCESS;
}

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
    InfoType = UR_DEVICE_INFO_BUILD_ON_SUBDEVICE;
    break;
  case PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_3D:
    InfoType = UR_DEVICE_INFO_MAX_WORK_GROUPS_3D;
    break;
  case PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE:
    InfoType = UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE;
    break;
  case PI_DEVICE_INFO_DEVICE_ID:
    InfoType = UR_DEVICE_INFO_DEVICE_ID;
    break;
  case PI_EXT_INTEL_DEVICE_INFO_FREE_MEMORY:
    InfoType = UR_DEVICE_INFO_GLOBAL_MEM_FREE;
    break;
  case PI_EXT_INTEL_DEVICE_INFO_MEMORY_CLOCK_RATE:
    InfoType = UR_DEVICE_INFO_MEMORY_CLOCK_RATE;
    break;
  case PI_EXT_INTEL_DEVICE_INFO_MEMORY_BUS_WIDTH:
    InfoType = UR_DEVICE_INFO_MEMORY_BUS_WIDTH;
    break;
  case PI_EXT_INTEL_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES:
    InfoType = UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES;
    break;
  case PI_DEVICE_INFO_GPU_SLICES:
    InfoType = UR_DEVICE_INFO_GPU_EU_SLICES;
    break;
  case PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
    InfoType = UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE;
    break;
  case PI_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
    InfoType = UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU;
    break;
  case PI_DEVICE_INFO_MAX_MEM_BANDWIDTH:
    InfoType = UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH;
    break;
  case PI_EXT_ONEAPI_DEVICE_INFO_BFLOAT16_MATH_FUNCTIONS:
    InfoType = UR_DEVICE_INFO_BFLOAT16;
    break;
  case PI_EXT_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES:
    InfoType = UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES;
    break;
  case PI_EXT_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
    InfoType = UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES;
    break;
  case PI_EXT_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES:
    InfoType = UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES;
    break;
  case PI_EXT_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES:
    InfoType = UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES;
    break;
  case PI_EXT_INTEL_DEVICE_INFO_MEM_CHANNEL_SUPPORT:
    InfoType = UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT;
    break;
  case PI_DEVICE_INFO_IMAGE_SRGB:
    InfoType = UR_DEVICE_INFO_IMAGE_SRGB;
    break;
  case PI_DEVICE_INFO_BACKEND_VERSION: {
    InfoType = UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION;
    break;
  }
  case PI_EXT_CODEPLAY_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP: {
    InfoType = UR_EXT_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP;
    break;
  }
  default:
    return PI_ERROR_UNKNOWN;
  };

  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  size_t UrParamValueSizeRet;
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);

  HANDLE_ERRORS(urDeviceGetInfo(UrDevice, InfoType, ParamValueSize, ParamValue,
                                &UrParamValueSizeRet));

  if (ParamValueSizeRet) {
    *ParamValueSizeRet = UrParamValueSizeRet;
  }
  ur2piDeviceInfoValue(InfoType, ParamValueSize, &ParamValueSize, ParamValue);
  fixupInfoValueTypes(UrParamValueSizeRet, ParamValueSizeRet, ParamValueSize,
                      ParamValue);

  return PI_SUCCESS;
}

inline pi_result piextDeviceGetNativeHandle(pi_device Device,
                                            pi_native_handle *NativeHandle) {
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);

  ur_native_handle_t UrNativeHandle{};
  HANDLE_ERRORS(urDeviceGetNativeHandle(UrDevice, &UrNativeHandle));
  *NativeHandle = reinterpret_cast<pi_native_handle>(UrNativeHandle);
  return PI_SUCCESS;
}

inline pi_result
piextDeviceCreateWithNativeHandle(pi_native_handle NativeHandle,
                                  pi_platform Platform, pi_device *Device) {

  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  ur_native_handle_t UrNativeDevice =
      reinterpret_cast<ur_native_handle_t>(NativeHandle);
  ur_platform_handle_t UrPlatform =
      reinterpret_cast<ur_platform_handle_t>(Platform);
  auto UrDevice = reinterpret_cast<ur_device_handle_t *>(Device);
  ur_device_native_properties_t UrProperties{};
  HANDLE_ERRORS(urDeviceCreateWithNativeHandle(UrNativeDevice, UrPlatform,
                                               &UrProperties, UrDevice));

  return PI_SUCCESS;
}

inline pi_result piDevicePartition(
    pi_device Device, const pi_device_partition_property *Properties,
    pi_uint32 NumEntries, pi_device *SubDevices, pi_uint32 *NumSubDevices) {

  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

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

  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  auto UrSubDevices = reinterpret_cast<ur_device_handle_t *>(SubDevices);
  HANDLE_ERRORS(urDevicePartition(UrDevice, UrProperties, NumEntries,
                                  UrSubDevices, NumSubDevices));
  return PI_SUCCESS;
}

inline pi_result piGetDeviceAndHostTimer(pi_device Device, uint64_t *DeviceTime,
                                         uint64_t *HostTime) {
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  HANDLE_ERRORS(urDeviceGetGlobalTimestamps(UrDevice, DeviceTime, HostTime));
  return PI_SUCCESS;
}

inline pi_result
piextDeviceSelectBinary(pi_device Device, // TODO: does this need to be context?
                        pi_device_binary *Binaries, pi_uint32 NumBinaries,
                        pi_uint32 *SelectedBinaryInd) {

  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  std::vector<ur_device_binary_t> UrBinaries(NumBinaries);

  for (uint32_t BinaryCount = 0; BinaryCount < NumBinaries; BinaryCount++) {
    if (strcmp(Binaries[BinaryCount]->DeviceTargetSpec,
               __SYCL_PI_DEVICE_BINARY_TARGET_UNKNOWN) == 0)
      UrBinaries[BinaryCount].pDeviceTargetSpec =
          UR_DEVICE_BINARY_TARGET_UNKNOWN;
    else if (strcmp(Binaries[BinaryCount]->DeviceTargetSpec,
                    __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV32) == 0)
      UrBinaries[BinaryCount].pDeviceTargetSpec =
          UR_DEVICE_BINARY_TARGET_SPIRV32;
    else if (strcmp(Binaries[BinaryCount]->DeviceTargetSpec,
                    __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64) == 0)
      UrBinaries[BinaryCount].pDeviceTargetSpec =
          UR_DEVICE_BINARY_TARGET_SPIRV64;
    else if (strcmp(Binaries[BinaryCount]->DeviceTargetSpec,
                    __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64) == 0)
      UrBinaries[BinaryCount].pDeviceTargetSpec =
          UR_DEVICE_BINARY_TARGET_SPIRV64_X86_64;
    else if (strcmp(Binaries[BinaryCount]->DeviceTargetSpec,
                    __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_GEN) == 0)
      UrBinaries[BinaryCount].pDeviceTargetSpec =
          UR_DEVICE_BINARY_TARGET_SPIRV64_GEN;
    else if (strcmp(Binaries[BinaryCount]->DeviceTargetSpec,
                    __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_FPGA) == 0)
      UrBinaries[BinaryCount].pDeviceTargetSpec =
          UR_DEVICE_BINARY_TARGET_SPIRV64_FPGA;
    else if (strcmp(Binaries[BinaryCount]->DeviceTargetSpec,
                    __SYCL_PI_DEVICE_BINARY_TARGET_NVPTX64) == 0)
      UrBinaries[BinaryCount].pDeviceTargetSpec =
          UR_DEVICE_BINARY_TARGET_NVPTX64;
    else if (strcmp(Binaries[BinaryCount]->DeviceTargetSpec,
                    __SYCL_PI_DEVICE_BINARY_TARGET_AMDGCN) == 0)
      UrBinaries[BinaryCount].pDeviceTargetSpec =
          UR_DEVICE_BINARY_TARGET_AMDGCN;
    else
      UrBinaries[BinaryCount].pDeviceTargetSpec =
          UR_DEVICE_BINARY_TARGET_UNKNOWN;
  }

  HANDLE_ERRORS(urDeviceSelectBinary(UrDevice, UrBinaries.data(), NumBinaries,
                                     SelectedBinaryInd));
  return PI_SUCCESS;
}

// Device
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Context
inline pi_result piContextCreate(const pi_context_properties *Properties,
                                 pi_uint32 NumDevices, const pi_device *Devices,
                                 void (*PFnNotify)(const char *ErrInfo,
                                                   const void *PrivateInfo,
                                                   size_t CB, void *UserData),
                                 void *UserData, pi_context *RetContext) {
  std::ignore = Properties;
  std::ignore = PFnNotify;
  std::ignore = UserData;
  auto UrDevices = reinterpret_cast<const ur_device_handle_t *>(Devices);

  ur_context_handle_t *UrContext =
      reinterpret_cast<ur_context_handle_t *>(RetContext);
  // TODO: Parse PI Context Properties into UR
  ur_context_properties_t UrProperties{};
  HANDLE_ERRORS(
      urContextCreate(NumDevices, UrDevices, &UrProperties, UrContext));
  return PI_SUCCESS;
}

inline pi_result piextContextSetExtendedDeleter(
    pi_context Context, pi_context_extended_deleter Function, void *UserData) {
  auto hContext = reinterpret_cast<ur_context_handle_t>(Context);

  HANDLE_ERRORS(urContextSetExtendedDeleter(hContext, Function, UserData));

  return PI_SUCCESS;
}

inline pi_result piextContextGetNativeHandle(pi_context Context,
                                             pi_native_handle *NativeHandle) {

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  ur_native_handle_t UrNativeHandle{};
  HANDLE_ERRORS(urContextGetNativeHandle(UrContext, &UrNativeHandle));
  *NativeHandle = reinterpret_cast<pi_native_handle>(UrNativeHandle);
  return PI_SUCCESS;
}

inline pi_result piextContextCreateWithNativeHandle(
    pi_native_handle NativeHandle, pi_uint32 NumDevices,
    const pi_device *Devices, bool OwnNativeHandle, pi_context *RetContext) {
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Devices, PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(RetContext, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(NumDevices, PI_ERROR_INVALID_VALUE);

  ur_native_handle_t NativeContext =
      reinterpret_cast<ur_native_handle_t>(NativeHandle);
  const ur_device_handle_t *UrDevices =
      reinterpret_cast<const ur_device_handle_t *>(Devices);
  ur_context_handle_t *UrContext =
      reinterpret_cast<ur_context_handle_t *>(RetContext);

  ur_context_native_properties_t Properties{};
  Properties.isNativeHandleOwned = OwnNativeHandle;
  HANDLE_ERRORS(urContextCreateWithNativeHandle(
      NativeContext, NumDevices, UrDevices, &Properties, UrContext));

  return PI_SUCCESS;
}

inline pi_result piContextGetInfo(pi_context Context, pi_context_info ParamName,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  ur_context_handle_t hContext = reinterpret_cast<ur_context_handle_t>(Context);
  ur_context_info_t ContextInfoType{};

  switch (ParamName) {
  case PI_CONTEXT_INFO_DEVICES: {
    ContextInfoType = UR_CONTEXT_INFO_DEVICES;
    break;
  }
  case PI_CONTEXT_INFO_NUM_DEVICES: {
    ContextInfoType = UR_CONTEXT_INFO_NUM_DEVICES;
    break;
  }
  case PI_CONTEXT_INFO_REFERENCE_COUNT: {
    ContextInfoType = UR_CONTEXT_INFO_REFERENCE_COUNT;
    break;
  }
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT:
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMSET2D_SUPPORT: {
    ContextInfoType = UR_CONTEXT_INFO_USM_FILL2D_SUPPORT;
    break;
  }
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT: {
    ContextInfoType = UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT;
    break;
  }
  case PI_EXT_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES:
  case PI_EXT_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  case PI_EXT_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES:
  case PI_EXT_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    // These queries should be dealt with in context_impl.cpp by calling the
    // queries of each device separately and building the intersection set.
    die("These queries should have never come here");
  }
  default: {
    die("piContextGetInfo: unsuppported ParamName.");
  }
  }

  size_t UrParamValueSizeRet;
  HANDLE_ERRORS(urContextGetInfo(hContext, ContextInfoType, ParamValueSize,
                                 ParamValue, &UrParamValueSizeRet));
  if (ParamValueSizeRet) {
    *ParamValueSizeRet = UrParamValueSizeRet;
  }
  fixupInfoValueTypes(UrParamValueSizeRet, ParamValueSizeRet, ParamValueSize,
                      ParamValue);
  return PI_SUCCESS;
}

inline pi_result piContextRetain(pi_context Context) {
  ur_context_handle_t hContext = reinterpret_cast<ur_context_handle_t>(Context);

  HANDLE_ERRORS(urContextRetain(hContext));

  return PI_SUCCESS;
}

inline pi_result piContextRelease(pi_context Context) {
  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  HANDLE_ERRORS(urContextRelease(UrContext));
  return PI_SUCCESS;
}
// Context
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Queue
inline pi_result piextQueueCreate(pi_context Context, pi_device Device,
                                  pi_queue_properties *Properties,
                                  pi_queue *Queue) {

  PI_ASSERT(Properties, PI_ERROR_INVALID_VALUE);
  // Expect flags mask to be passed first.
  PI_ASSERT(Properties[0] == PI_QUEUE_FLAGS, PI_ERROR_INVALID_VALUE);

  PI_ASSERT(Properties[2] == 0 ||
                (Properties[2] == PI_QUEUE_COMPUTE_INDEX && Properties[4] == 0),
            PI_ERROR_INVALID_VALUE);

  // Check that unexpected bits are not set.
  PI_ASSERT(!(Properties[1] &
              ~(PI_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                PI_QUEUE_FLAG_PROFILING_ENABLE | PI_QUEUE_FLAG_ON_DEVICE |
                PI_QUEUE_FLAG_ON_DEVICE_DEFAULT |
                PI_EXT_ONEAPI_QUEUE_FLAG_DISCARD_EVENTS |
                PI_EXT_ONEAPI_QUEUE_FLAG_PRIORITY_LOW |
                PI_EXT_ONEAPI_QUEUE_FLAG_PRIORITY_HIGH)),
            PI_ERROR_INVALID_VALUE);

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  ur_queue_properties_t UrProperties{};
  UrProperties.stype = UR_STRUCTURE_TYPE_QUEUE_PROPERTIES;
  if (Properties[1] & PI_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    UrProperties.flags |= UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  if (Properties[1] & PI_QUEUE_FLAG_PROFILING_ENABLE)
    UrProperties.flags |= UR_QUEUE_FLAG_PROFILING_ENABLE;
  if (Properties[1] & PI_QUEUE_FLAG_ON_DEVICE)
    UrProperties.flags |= UR_QUEUE_FLAG_ON_DEVICE;
  if (Properties[1] & PI_QUEUE_FLAG_ON_DEVICE_DEFAULT)
    UrProperties.flags |= UR_QUEUE_FLAG_ON_DEVICE_DEFAULT;
  if (Properties[1] & PI_EXT_ONEAPI_QUEUE_FLAG_DISCARD_EVENTS)
    UrProperties.flags |= UR_QUEUE_FLAG_DISCARD_EVENTS;
  if (Properties[1] & PI_EXT_ONEAPI_QUEUE_FLAG_PRIORITY_LOW)
    UrProperties.flags |= UR_QUEUE_FLAG_PRIORITY_LOW;
  if (Properties[1] & PI_EXT_ONEAPI_QUEUE_FLAG_PRIORITY_HIGH)
    UrProperties.flags |= UR_QUEUE_FLAG_PRIORITY_HIGH;

  ur_queue_index_properties_t IndexProperties{};
  IndexProperties.stype = UR_STRUCTURE_TYPE_QUEUE_INDEX_PROPERTIES;
  if (Properties[2] != 0) {
    IndexProperties.computeIndex = Properties[3];
  }

  UrProperties.pNext = &IndexProperties;

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);

  ur_queue_handle_t *UrQueue = reinterpret_cast<ur_queue_handle_t *>(Queue);
  HANDLE_ERRORS(urQueueCreate(UrContext, UrDevice, &UrProperties, UrQueue));

  return PI_SUCCESS;
}

inline pi_result piQueueCreate(pi_context Context, pi_device Device,
                               pi_queue_properties Flags, pi_queue *Queue) {
  pi_queue_properties Properties[] = {PI_QUEUE_FLAGS, Flags, 0};
  return pi2ur::piextQueueCreate(Context, Device, Properties, Queue);
}

inline pi_result piextQueueCreateWithNativeHandle(
    pi_native_handle NativeHandle, int32_t NativeHandleDesc, pi_context Context,
    pi_device Device, bool OwnNativeHandle, pi_queue_properties *Properties,
    pi_queue *Queue) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  ur_device_handle_t UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  ur_native_handle_t UrNativeHandle =
      reinterpret_cast<ur_native_handle_t>(NativeHandle);
  ur_queue_handle_t *UrQueue = reinterpret_cast<ur_queue_handle_t *>(Queue);
  ur_queue_native_properties_t UrNativeProperties{};
  UrNativeProperties.isNativeHandleOwned = OwnNativeHandle;

  ur_queue_properties_t UrProperties{};
  UrProperties.stype = UR_STRUCTURE_TYPE_QUEUE_PROPERTIES;
  if (Properties[1] & PI_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    UrProperties.flags |= UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  if (Properties[1] & PI_QUEUE_FLAG_PROFILING_ENABLE)
    UrProperties.flags |= UR_QUEUE_FLAG_PROFILING_ENABLE;
  if (Properties[1] & PI_QUEUE_FLAG_ON_DEVICE)
    UrProperties.flags |= UR_QUEUE_FLAG_ON_DEVICE;
  if (Properties[1] & PI_QUEUE_FLAG_ON_DEVICE_DEFAULT)
    UrProperties.flags |= UR_QUEUE_FLAG_ON_DEVICE_DEFAULT;
  if (Properties[1] & PI_EXT_ONEAPI_QUEUE_FLAG_DISCARD_EVENTS)
    UrProperties.flags |= UR_QUEUE_FLAG_DISCARD_EVENTS;
  if (Properties[1] & PI_EXT_ONEAPI_QUEUE_FLAG_PRIORITY_LOW)
    UrProperties.flags |= UR_QUEUE_FLAG_PRIORITY_LOW;
  if (Properties[1] & PI_EXT_ONEAPI_QUEUE_FLAG_PRIORITY_HIGH)
    UrProperties.flags |= UR_QUEUE_FLAG_PRIORITY_HIGH;

  ur_queue_native_desc_t UrNativeDesc{};
  UrNativeDesc.stype = UR_STRUCTURE_TYPE_QUEUE_NATIVE_DESC;
  UrNativeDesc.pNativeData = &NativeHandleDesc;

  UrProperties.pNext = &UrNativeDesc;
  UrNativeProperties.pNext = &UrProperties;

  HANDLE_ERRORS(urQueueCreateWithNativeHandle(
      UrNativeHandle, UrContext, UrDevice, &UrNativeProperties, UrQueue));
  return PI_SUCCESS;
}

inline pi_result piextQueueGetNativeHandle(pi_queue Queue,
                                           pi_native_handle *NativeHandle,
                                           int32_t *NativeHandleDesc) {

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  ur_queue_native_desc_t UrNativeDesc{};
  UrNativeDesc.pNativeData = NativeHandleDesc;

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);

  ur_native_handle_t UrNativeQueue{};
  HANDLE_ERRORS(urQueueGetNativeHandle(UrQueue, &UrNativeDesc, &UrNativeQueue));

  *NativeHandle = reinterpret_cast<pi_native_handle>(UrNativeQueue);

  return PI_SUCCESS;
}

inline pi_result piQueueRelease(pi_queue Queue) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);

  HANDLE_ERRORS(urQueueRelease(UrQueue));

  return PI_SUCCESS;
}

inline pi_result piQueueFinish(pi_queue Queue) {
  // Wait until command lists attached to the command queue are executed.
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);

  HANDLE_ERRORS(urQueueFinish(UrQueue));

  return PI_SUCCESS;
}

inline pi_result piQueueGetInfo(pi_queue Queue, pi_queue_info ParamName,
                                size_t ParamValueSize, void *ParamValue,
                                size_t *ParamValueSizeRet) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);

  ur_queue_info_t UrParamName{};

  switch (ParamName) {
  case PI_QUEUE_INFO_CONTEXT: {
    UrParamName = UR_QUEUE_INFO_CONTEXT;
    break;
  }
  case PI_QUEUE_INFO_DEVICE: {
    UrParamName = UR_QUEUE_INFO_DEVICE;
    break;
  }
  case PI_QUEUE_INFO_DEVICE_DEFAULT: {
    UrParamName = UR_QUEUE_INFO_DEVICE_DEFAULT;
    break;
  }
  case PI_QUEUE_INFO_PROPERTIES: {
    UrParamName = UR_QUEUE_INFO_FLAGS;
    break;
  }
  case PI_QUEUE_INFO_REFERENCE_COUNT: {
    UrParamName = UR_QUEUE_INFO_REFERENCE_COUNT;
    break;
  }
  case PI_QUEUE_INFO_SIZE: {
    UrParamName = UR_QUEUE_INFO_SIZE;
    break;
  }
  case PI_EXT_ONEAPI_QUEUE_INFO_EMPTY: {
    UrParamName = UR_QUEUE_INFO_EMPTY;
    break;
  }
  default: {
    die("Unsupported ParamName in piQueueGetInfo");
    return PI_ERROR_INVALID_VALUE;
  }
  }

  HANDLE_ERRORS(urQueueGetInfo(UrQueue, UrParamName, ParamValueSize, ParamValue,
                               ParamValueSizeRet));

  return PI_SUCCESS;
}

inline pi_result piQueueRetain(pi_queue Queue) {

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);

  HANDLE_ERRORS(urQueueRetain(UrQueue));

  return PI_SUCCESS;
}

inline pi_result piQueueFlush(pi_queue Queue) {

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);

  HANDLE_ERRORS(urQueueFlush(UrQueue));

  return PI_SUCCESS;
}

// Queue
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Program

inline pi_result piProgramCreate(pi_context Context, const void *ILBytes,
                                 size_t Length, pi_program *Program) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(ILBytes && Length, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);

  ur_program_properties_t UrProperties{};
  ur_program_handle_t *UrProgram =
      reinterpret_cast<ur_program_handle_t *>(Program);
  HANDLE_ERRORS(urProgramCreateWithIL(UrContext, ILBytes, Length, &UrProperties,
                                      UrProgram));

  return PI_SUCCESS;
}

inline pi_result piProgramCreateWithBinary(
    pi_context Context, pi_uint32 NumDevices, const pi_device *DeviceList,
    const size_t *Lengths, const unsigned char **Binaries,
    size_t NumMetadataEntries, const pi_device_binary_property *Metadata,
    pi_int32 *BinaryStatus, pi_program *Program) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(DeviceList && NumDevices, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Binaries && Lengths, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);

  // For now we support only one device.
  if (NumDevices != 1) {
    die("piProgramCreateWithBinary: level_zero supports only one device.");
    return PI_ERROR_INVALID_VALUE;
  }
  if (!Binaries[0] || !Lengths[0]) {
    if (BinaryStatus)
      *BinaryStatus = PI_ERROR_INVALID_VALUE;
    return PI_ERROR_INVALID_VALUE;
  }

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(DeviceList[0]);

  ur_program_properties_t Properties = {};
  Properties.stype = UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES;
  Properties.pNext = nullptr;
  Properties.count = NumMetadataEntries;

  std::unique_ptr<ur_program_metadata_t[]> pMetadatas;
  if (NumMetadataEntries) {
    pMetadatas.reset(new ur_program_metadata_t[NumMetadataEntries]);
    for (unsigned i = 0; i < NumMetadataEntries; i++) {
      HANDLE_ERRORS(mapPIMetadataToUR(&Metadata[i], &pMetadatas[i]));
    }

    Properties.pMetadatas = pMetadatas.get();
  }

  ur_program_handle_t *UrProgram =
      reinterpret_cast<ur_program_handle_t *>(Program);
  HANDLE_ERRORS(urProgramCreateWithBinary(UrContext, UrDevice, Lengths[0],
                                          Binaries[0], &Properties, UrProgram));

  if (BinaryStatus)
    *BinaryStatus = PI_SUCCESS;

  return PI_SUCCESS;
}

inline pi_result piclProgramCreateWithSource(pi_context Context,
                                             pi_uint32 Count,
                                             const char **Strings,
                                             const size_t *Lengths,
                                             pi_program *RetProgram) {
  std::ignore = Context;
  std::ignore = Count;
  std::ignore = Strings;
  std::ignore = Lengths;
  std::ignore = RetProgram;
  die("piclProgramCreateWithSource: not supported in UR\n");
  return PI_ERROR_INVALID_OPERATION;
}

inline pi_result piProgramGetInfo(pi_program Program, pi_program_info ParamName,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {

  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);

  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);

  ur_program_info_t PropName{};

  switch (ParamName) {
  case PI_PROGRAM_INFO_REFERENCE_COUNT: {
    PropName = UR_PROGRAM_INFO_REFERENCE_COUNT;
    break;
  }
  case PI_PROGRAM_INFO_CONTEXT: {
    PropName = UR_PROGRAM_INFO_CONTEXT;
    break;
  }
  case PI_PROGRAM_INFO_NUM_DEVICES: {
    PropName = UR_PROGRAM_INFO_NUM_DEVICES;
    break;
  }
  case PI_PROGRAM_INFO_DEVICES: {
    PropName = UR_PROGRAM_INFO_DEVICES;
    break;
  }
  case PI_PROGRAM_INFO_SOURCE: {
    PropName = UR_PROGRAM_INFO_SOURCE;
    break;
  }
  case PI_PROGRAM_INFO_BINARY_SIZES: {
    PropName = UR_PROGRAM_INFO_BINARY_SIZES;
    break;
  }
  case PI_PROGRAM_INFO_BINARIES: {
    PropName = UR_PROGRAM_INFO_BINARIES;
    break;
  }
  case PI_PROGRAM_INFO_NUM_KERNELS: {
    PropName = UR_PROGRAM_INFO_NUM_KERNELS;
    break;
  }
  case PI_PROGRAM_INFO_KERNEL_NAMES: {
    PropName = UR_PROGRAM_INFO_KERNEL_NAMES;
    break;
  }
  default: {
    die("urProgramGetInfo: not implemented");
  }
  }

  HANDLE_ERRORS(urProgramGetInfo(UrProgram, PropName, ParamValueSize,
                                 ParamValue, ParamValueSizeRet));

  return PI_SUCCESS;
}

inline pi_result
piProgramLink(pi_context Context, pi_uint32 NumDevices,
              const pi_device *DeviceList, const char *Options,
              pi_uint32 NumInputPrograms, const pi_program *InputPrograms,
              void (*PFnNotify)(pi_program Program, void *UserData),
              void *UserData, pi_program *RetProgram) {
  // We only support one device with Level Zero currently.
  if (NumDevices != 1) {
    die("piProgramLink: level_zero supports only one device.");
    return PI_ERROR_INVALID_VALUE;
  }

  // Validate input parameters.
  PI_ASSERT(DeviceList, PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(!PFnNotify && !UserData, PI_ERROR_INVALID_VALUE);
  if (NumInputPrograms == 0 || InputPrograms == nullptr)
    return PI_ERROR_INVALID_VALUE;

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  const ur_program_handle_t *UrInputPrograms =
      reinterpret_cast<const ur_program_handle_t *>(InputPrograms);
  ur_program_handle_t *UrProgram =
      reinterpret_cast<ur_program_handle_t *>(RetProgram);

  HANDLE_ERRORS(urProgramLink(UrContext, NumInputPrograms, UrInputPrograms,
                              Options, UrProgram));

  return PI_SUCCESS;
}

inline pi_result piProgramCompile(
    pi_program Program, pi_uint32 NumDevices, const pi_device *DeviceList,
    const char *Options, pi_uint32 NumInputHeaders,
    const pi_program *InputHeaders, const char **HeaderIncludeNames,
    void (*PFnNotify)(pi_program Program, void *UserData), void *UserData) {

  std::ignore = NumInputHeaders;
  std::ignore = InputHeaders;
  std::ignore = HeaderIncludeNames;

  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);

  if ((NumDevices && !DeviceList) || (!NumDevices && DeviceList))
    return PI_ERROR_INVALID_VALUE;

  // These aren't supported.
  PI_ASSERT(!PFnNotify && !UserData, PI_ERROR_INVALID_VALUE);

  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);

  ur_program_info_t PropName = UR_PROGRAM_INFO_CONTEXT;
  ur_context_handle_t UrContext{};
  HANDLE_ERRORS(urProgramGetInfo(UrProgram, PropName, sizeof(&UrContext),
                                 &UrContext, nullptr));

  HANDLE_ERRORS(urProgramCompile(UrContext, UrProgram, Options));

  return PI_SUCCESS;
}

inline pi_result
piProgramBuild(pi_program Program, pi_uint32 NumDevices,
               const pi_device *DeviceList, const char *Options,
               void (*PFnNotify)(pi_program Program, void *UserData),
               void *UserData) {
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);
  if ((NumDevices && !DeviceList) || (!NumDevices && DeviceList)) {
    return PI_ERROR_INVALID_VALUE;
  }

  // We only support build to one device with Level Zero now.
  // TODO: we should eventually build to the possibly multiple root
  // devices in the context.
  if (NumDevices != 1) {
    die("piProgramBuild: level_zero supports only one device.");
    return PI_ERROR_INVALID_VALUE;
  }

  // These aren't supported.
  PI_ASSERT(!PFnNotify && !UserData, PI_ERROR_INVALID_VALUE);

  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);
  ur_program_info_t PropName = UR_PROGRAM_INFO_CONTEXT;
  ur_context_handle_t UrContext{};
  HANDLE_ERRORS(urProgramGetInfo(UrProgram, PropName, sizeof(&UrContext),
                                 &UrContext, nullptr));

  HANDLE_ERRORS(urProgramBuild(UrContext, UrProgram, Options));

  return PI_SUCCESS;
}

inline pi_result piextProgramSetSpecializationConstant(pi_program Program,
                                                       pi_uint32 SpecID,
                                                       size_t Size,
                                                       const void *SpecValue) {
  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);
  uint32_t Count = 1;
  ur_specialization_constant_info_t SpecConstant{};
  SpecConstant.id = SpecID;
  SpecConstant.size = Size;
  SpecConstant.pValue = SpecValue;
  HANDLE_ERRORS(
      urProgramSetSpecializationConstants(UrProgram, Count, &SpecConstant));

  return PI_SUCCESS;
}

inline pi_result piKernelCreate(pi_program Program, const char *KernelName,
                                pi_kernel *RetKernel) {
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);
  PI_ASSERT(RetKernel, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(KernelName, PI_ERROR_INVALID_VALUE);

  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);
  ur_kernel_handle_t *UrKernel =
      reinterpret_cast<ur_kernel_handle_t *>(RetKernel);

  HANDLE_ERRORS(urKernelCreate(UrProgram, KernelName, UrKernel));

  return PI_SUCCESS;
}

inline pi_result
piEnqueueMemImageFill(pi_queue Queue, pi_mem Image, const void *FillColor,
                      const size_t *Origin, const size_t *Region,
                      pi_uint32 NumEventsInWaitList,
                      const pi_event *EventsWaitList, pi_event *Event) {

  std::ignore = Image;
  std::ignore = FillColor;
  std::ignore = Origin;
  std::ignore = Region;
  std::ignore = NumEventsInWaitList;
  std::ignore = EventsWaitList;
  std::ignore = Event;

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  die("piEnqueueMemImageFill: not implemented");
  return PI_SUCCESS;
}

inline pi_result
piEnqueueNativeKernel(pi_queue Queue, void (*UserFunc)(void *), void *Args,
                      size_t CbArgs, pi_uint32 NumMemObjects,
                      const pi_mem *MemList, const void **ArgsMemLoc,
                      pi_uint32 NumEventsInWaitList,
                      const pi_event *EventsWaitList, pi_event *Event) {
  std::ignore = UserFunc;
  std::ignore = Args;
  std::ignore = CbArgs;
  std::ignore = NumMemObjects;
  std::ignore = MemList;
  std::ignore = ArgsMemLoc;
  std::ignore = NumEventsInWaitList;
  std::ignore = EventsWaitList;
  std::ignore = Event;

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  die("piEnqueueNativeKernel: not implemented");
  return PI_SUCCESS;
}

inline pi_result piextGetDeviceFunctionPointer(pi_device Device,
                                               pi_program Program,
                                               const char *FunctionName,
                                               pi_uint64 *FunctionPointerRet) {

  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);

  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);

  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);

  void **FunctionPointer = reinterpret_cast<void **>(FunctionPointerRet);

  HANDLE_ERRORS(urProgramGetFunctionPointer(UrDevice, UrProgram, FunctionName,
                                            FunctionPointer));
  return PI_SUCCESS;
}

// Special version of piKernelSetArg to accept pi_mem.
inline pi_result piextKernelSetArgMemObj(pi_kernel Kernel, pi_uint32 ArgIndex,
                                         const pi_mem *ArgValue) {

  // TODO: the better way would probably be to add a new PI API for
  // extracting native PI object from PI handle, and have SYCL
  // RT pass that directly to the regular piKernelSetArg (and
  // then remove this piextKernelSetArgMemObj).

  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  ur_mem_handle_t UrMemory{};
  if (ArgValue)
    UrMemory = reinterpret_cast<ur_mem_handle_t>(*ArgValue);

  // We don't yet know the device where this kernel will next be run on.
  // Thus we can't know the actual memory allocation that needs to be used.
  // Remember the memory object being used as an argument for this kernel
  // to process it later when the device is known (at the kernel enqueue).
  //
  // TODO: for now we have to conservatively assume the access as read-write.
  //       Improve that by passing SYCL buffer accessor type into
  //       piextKernelSetArgMemObj.
  //

  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);
  HANDLE_ERRORS(urKernelSetArgMemObj(UrKernel, ArgIndex, UrMemory));
  return PI_SUCCESS;
}

inline pi_result piKernelSetArg(pi_kernel Kernel, pi_uint32 ArgIndex,
                                size_t ArgSize, const void *ArgValue) {

  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);

  HANDLE_ERRORS(urKernelSetArgValue(UrKernel, ArgIndex, ArgSize, ArgValue));
  return PI_SUCCESS;
}

inline pi_result piKernelSetArgPointer(pi_kernel kernel, pi_uint32 arg_index,
                                       size_t arg_size, const void *arg_value) {
  (void)arg_size;
  auto hKernel = reinterpret_cast<ur_kernel_handle_t>(kernel);
  HANDLE_ERRORS(urKernelSetArgPointer(hKernel, arg_index, arg_value));

  return PI_SUCCESS;
}

inline pi_result
piextKernelCreateWithNativeHandle(pi_native_handle NativeHandle,
                                  pi_context Context, pi_program Program,
                                  bool OwnNativeHandle, pi_kernel *Kernel) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  ur_native_handle_t UrNativeKernel =
      reinterpret_cast<ur_native_handle_t>(NativeHandle);
  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);
  ur_kernel_handle_t *UrKernel = reinterpret_cast<ur_kernel_handle_t *>(Kernel);
  ur_kernel_native_properties_t Properties{};
  Properties.isNativeHandleOwned = OwnNativeHandle;
  HANDLE_ERRORS(urKernelCreateWithNativeHandle(
      UrNativeKernel, UrContext, UrProgram, &Properties, UrKernel));

  return PI_SUCCESS;
}

inline pi_result piProgramRetain(pi_program Program) {
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);

  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);
  HANDLE_ERRORS(
      urProgramRetain(reinterpret_cast<ur_program_handle_t>(UrProgram)));

  return PI_SUCCESS;
}

inline pi_result piKernelSetExecInfo(pi_kernel Kernel,
                                     pi_kernel_exec_info ParamName,
                                     size_t ParamValueSize,
                                     const void *ParamValue) {

  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);
  PI_ASSERT(ParamValue, PI_ERROR_INVALID_VALUE);

  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);
  ur_kernel_exec_info_t PropName{};
  uint64_t PropValue{};
  switch (ParamName) {
  case PI_USM_INDIRECT_ACCESS: {
    PropName = UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS;
    PropValue = *(static_cast<uint64_t *>(const_cast<void *>(ParamValue)));
    break;
  }
  case PI_USM_PTRS: {
    PropName = UR_KERNEL_EXEC_INFO_USM_PTRS;
    break;
  }
  case PI_EXT_KERNEL_EXEC_INFO_CACHE_CONFIG: {
    PropName = UR_EXT_KERNEL_EXEC_INFO_CACHE_CONFIG;
    auto Param = (*(static_cast<const pi_kernel_cache_config *>(ParamValue)));
    if (Param == PI_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_SLM) {
      PropValue =
          static_cast<uint64_t>(UR_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_SLM);
    } else if (Param == PI_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_DATA) {
      PropValue =
          static_cast<uint64_t>(UR_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_DATA);
      break;
    } else if (Param == PI_EXT_KERNEL_EXEC_INFO_CACHE_DEFAULT) {
      PropValue = static_cast<uint64_t>(UR_EXT_KERNEL_EXEC_INFO_CACHE_DEFAULT);
    } else {
      die("piKernelSetExecInfo: unsupported ParamValue\n");
    }
    break;
  }
  default:
    die("piKernelSetExecInfo: unsupported ParamName\n");
  }
  HANDLE_ERRORS(
      urKernelSetExecInfo(UrKernel, PropName, ParamValueSize, &PropValue));

  return PI_SUCCESS;
}

inline pi_result piextProgramGetNativeHandle(pi_program Program,
                                             pi_native_handle *NativeHandle) {
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);
  ur_native_handle_t NativeProgram{};
  HANDLE_ERRORS(urProgramGetNativeHandle(UrProgram, &NativeProgram));

  *NativeHandle = reinterpret_cast<pi_native_handle>(NativeProgram);

  return PI_SUCCESS;
}

inline pi_result
piextProgramCreateWithNativeHandle(pi_native_handle NativeHandle,
                                   pi_context Context, bool OwnNativeHandle,
                                   pi_program *Program) {
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  ur_native_handle_t NativeProgram =
      reinterpret_cast<ur_native_handle_t>(NativeHandle);
  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  ur_program_handle_t *UrProgram =
      reinterpret_cast<ur_program_handle_t *>(Program);
  ur_program_native_properties_t UrProperties{};
  UrProperties.isNativeHandleOwned = OwnNativeHandle;
  HANDLE_ERRORS(urProgramCreateWithNativeHandle(NativeProgram, UrContext,
                                                &UrProperties, UrProgram));
  return PI_SUCCESS;
}

inline pi_result piKernelGetInfo(pi_kernel Kernel, pi_kernel_info ParamName,
                                 size_t ParamValueSize, void *ParamValue,
                                 size_t *ParamValueSizeRet) {
  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);
  ur_kernel_info_t UrParamName{};
  switch (ParamName) {
  case PI_KERNEL_INFO_FUNCTION_NAME: {
    UrParamName = UR_KERNEL_INFO_FUNCTION_NAME;
    break;
  }
  case PI_KERNEL_INFO_NUM_ARGS: {
    UrParamName = UR_KERNEL_INFO_NUM_ARGS;
    break;
  }
  case PI_KERNEL_INFO_REFERENCE_COUNT: {
    UrParamName = UR_KERNEL_INFO_REFERENCE_COUNT;
    break;
  }
  case PI_KERNEL_INFO_CONTEXT: {
    UrParamName = UR_KERNEL_INFO_CONTEXT;
    break;
  }
  case PI_KERNEL_INFO_PROGRAM: {
    UrParamName = UR_KERNEL_INFO_PROGRAM;
    break;
  }
  case PI_KERNEL_INFO_ATTRIBUTES: {
    UrParamName = UR_KERNEL_INFO_ATTRIBUTES;
    break;
  }
  default:
    return PI_ERROR_INVALID_PROPERTY;
  }

  HANDLE_ERRORS(urKernelGetInfo(UrKernel, UrParamName, ParamValueSize,
                                ParamValue, ParamValueSizeRet));

  return PI_SUCCESS;
}

inline pi_result piKernelGetGroupInfo(pi_kernel Kernel, pi_device Device,
                                      pi_kernel_group_info ParamName,
                                      size_t ParamValueSize, void *ParamValue,
                                      size_t *ParamValueSizeRet) {
  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);

  ur_kernel_group_info_t UrParamName{};
  switch (ParamName) {
  case PI_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE: {
    UrParamName = UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE;
    break;
  }
  case PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
    UrParamName = UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE;
    break;
  }
  case PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE: {
    UrParamName = UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE;
    break;
  }
  case PI_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE: {
    UrParamName = UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE;
    break;
  }
  case PI_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
    UrParamName = UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE;
    break;
  }
  case PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
    UrParamName = UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE;
    break;
  }
  // The number of registers used by the compiled kernel (device specific)
  case PI_KERNEL_GROUP_INFO_NUM_REGS: {
    HANDLE_ERRORS(urKernelGetInfo(UrKernel, UR_KERNEL_INFO_NUM_REGS,
                                  ParamValueSize, ParamValue,
                                  ParamValueSizeRet));
    return PI_SUCCESS;
  }
  default: {
    die("Unknown ParamName in piKernelGetGroupInfo");
    return PI_ERROR_INVALID_VALUE;
  }
  }

  HANDLE_ERRORS(urKernelGetGroupInfo(UrKernel, UrDevice, UrParamName,
                                     ParamValueSize, ParamValue,
                                     ParamValueSizeRet));

  return PI_SUCCESS;
}

inline pi_result piKernelRetain(pi_kernel Kernel) {

  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);

  HANDLE_ERRORS(urKernelRetain(UrKernel));

  return PI_SUCCESS;
}

inline pi_result piKernelRelease(pi_kernel Kernel) {

  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);

  HANDLE_ERRORS(urKernelRelease(UrKernel));

  return PI_SUCCESS;
}

inline pi_result piProgramRelease(pi_program Program) {

  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);

  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);

  HANDLE_ERRORS(urProgramRelease(UrProgram));

  return PI_SUCCESS;
}

inline pi_result piextKernelSetArgPointer(pi_kernel Kernel, pi_uint32 ArgIndex,
                                          size_t ArgSize,
                                          const void *ArgValue) {
  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);

  HANDLE_ERRORS(urKernelSetArgValue(UrKernel, ArgIndex, ArgSize, ArgValue));

  return PI_SUCCESS;
}

inline pi_result piKernelGetSubGroupInfo(
    pi_kernel Kernel, pi_device Device, pi_kernel_sub_group_info ParamName,
    size_t InputValueSize, const void *InputValue, size_t ParamValueSize,
    void *ParamValue, size_t *ParamValueSizeRet) {

  std::ignore = InputValueSize;
  std::ignore = InputValue;

  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);

  ur_kernel_sub_group_info_t PropName{};
  switch (ParamName) {
  case PI_KERNEL_MAX_SUB_GROUP_SIZE: {
    PropName = UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE;
    break;
  }
  case PI_KERNEL_MAX_NUM_SUB_GROUPS: {
    PropName = UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS;
    break;
  }
  case PI_KERNEL_COMPILE_NUM_SUB_GROUPS: {
    PropName = UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS;
    break;
  }
  case PI_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL: {
    PropName = UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL;
    break;
  }
  }
  HANDLE_ERRORS(urKernelGetSubGroupInfo(UrKernel, UrDevice, PropName,
                                        ParamValueSize, ParamValue,
                                        ParamValueSizeRet));

  return PI_SUCCESS;
}

inline pi_result piProgramGetBuildInfo(pi_program Program, pi_device Device,
                                       pi_program_build_info ParamName,
                                       size_t ParamValueSize, void *ParamValue,
                                       size_t *ParamValueSizeRet) {

  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);

  ur_program_build_info_t PropName{};
  switch (ParamName) {
  case PI_PROGRAM_BUILD_INFO_STATUS: {
    PropName = UR_PROGRAM_BUILD_INFO_STATUS;
    break;
  }
  case PI_PROGRAM_BUILD_INFO_OPTIONS: {
    PropName = UR_PROGRAM_BUILD_INFO_OPTIONS;
    break;
  }
  case PI_PROGRAM_BUILD_INFO_LOG: {
    PropName = UR_PROGRAM_BUILD_INFO_LOG;
    break;
  }
  case PI_PROGRAM_BUILD_INFO_BINARY_TYPE: {
    PropName = UR_PROGRAM_BUILD_INFO_BINARY_TYPE;
    break;
  }
  default: {
    die("piProgramGetBuildInfo: not implemented");
  }
  }
  HANDLE_ERRORS(urProgramGetBuildInfo(UrProgram, UrDevice, PropName,
                                      ParamValueSize, ParamValue,
                                      ParamValueSizeRet));

  return PI_SUCCESS;
}

inline pi_result piextKernelGetNativeHandle(pi_kernel Kernel,
                                            pi_native_handle *NativeHandle) {
  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);
  ur_native_handle_t NativeKernel{};
  HANDLE_ERRORS(urKernelGetNativeHandle(UrKernel, &NativeKernel));

  *NativeHandle = reinterpret_cast<pi_native_handle>(NativeKernel);

  return PI_SUCCESS;
}

/// API for writing data from host to a device global variable.
///
/// \param Queue is the queue
/// \param Program is the program containing the device global variable
/// \param Name is the unique identifier for the device global variable
/// \param BlockingWrite is true if the write should block
/// \param Count is the number of bytes to copy
/// \param Offset is the byte offset into the device global variable to start
/// copying
/// \param Src is a pointer to where the data must be copied from
/// \param NumEventsInWaitList is a number of events in the wait list
/// \param EventWaitList is the wait list
/// \param Event is the resulting event
inline pi_result piextEnqueueDeviceGlobalVariableWrite(
    pi_queue Queue, pi_program Program, const char *Name, pi_bool BlockingWrite,
    size_t Count, size_t Offset, const void *Src, pi_uint32 NumEventsInWaitList,
    const pi_event *EventsWaitList, pi_event *OutEvent) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);
  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);
  HANDLE_ERRORS(urEnqueueDeviceGlobalVariableWrite(
      UrQueue, UrProgram, Name, BlockingWrite, Count, Offset, Src,
      NumEventsInWaitList, UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

/// API reading data from a device global variable to host.
///
/// \param Queue is the queue
/// \param Program is the program containing the device global variable
/// \param Name is the unique identifier for the device global variable
/// \param BlockingRead is true if the read should block
/// \param Count is the number of bytes to copy
/// \param Offset is the byte offset into the device global variable to start
/// copying
/// \param Dst is a pointer to where the data must be copied to
/// \param NumEventsInWaitList is a number of events in the wait list
/// \param EventWaitList is the wait list
/// \param Event is the resulting event
inline pi_result piextEnqueueDeviceGlobalVariableRead(
    pi_queue Queue, pi_program Program, const char *Name, pi_bool BlockingRead,
    size_t Count, size_t Offset, void *Dst, pi_uint32 NumEventsInWaitList,
    const pi_event *EventsWaitList, pi_event *OutEvent) {

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueDeviceGlobalVariableRead(
      UrQueue, UrProgram, Name, BlockingRead, Count, Offset, Dst,
      NumEventsInWaitList, UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

// Program
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Memory
inline pi_result piMemBufferCreate(pi_context Context, pi_mem_flags Flags,
                                   size_t Size, void *HostPtr, pi_mem *RetMem,
                                   const pi_mem_properties *properties) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(RetMem, PI_ERROR_INVALID_VALUE);

  if (properties != nullptr) {
    die("piMemBufferCreate: no mem properties goes to Level-Zero RT yet");
  }

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);

  ur_mem_flags_t UrBufferFlags{};
  if (Flags & PI_MEM_FLAGS_ACCESS_RW) {
    UrBufferFlags |= UR_MEM_FLAG_READ_WRITE;
  }
  if (Flags & PI_MEM_ACCESS_READ_ONLY) {
    UrBufferFlags |= UR_MEM_FLAG_READ_ONLY;
  }
  if (Flags & PI_MEM_FLAGS_HOST_PTR_USE) {
    UrBufferFlags |= UR_MEM_FLAG_USE_HOST_POINTER;
  }
  if (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) {
    UrBufferFlags |= UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER;
  }
  if (Flags & PI_MEM_FLAGS_HOST_PTR_ALLOC) {
    UrBufferFlags |= UR_MEM_FLAG_ALLOC_HOST_POINTER;
  }

  ur_buffer_properties_t UrProps{};
  UrProps.stype = UR_STRUCTURE_TYPE_BUFFER_PROPERTIES;
  UrProps.pHost = HostPtr;
  ur_mem_handle_t *UrBuffer = reinterpret_cast<ur_mem_handle_t *>(RetMem);
  HANDLE_ERRORS(
      urMemBufferCreate(UrContext, UrBufferFlags, Size, &UrProps, UrBuffer));

  return PI_SUCCESS;
}

inline pi_result piextUSMHostAlloc(void **ResultPtr, pi_context Context,
                                   pi_usm_mem_properties *Properties,
                                   size_t Size, pi_uint32 Alignment) {

  std::ignore = Properties;
  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  ur_usm_desc_t USMDesc{};
  USMDesc.align = Alignment;
  ur_usm_pool_handle_t Pool{};
  HANDLE_ERRORS(urUSMHostAlloc(UrContext, &USMDesc, Pool, Size, ResultPtr));
  return PI_SUCCESS;
}

inline pi_result piMemGetInfo(pi_mem Mem, pi_mem_info ParamName,
                              size_t ParamValueSize, void *ParamValue,
                              size_t *ParamValueSizeRet) {
  PI_ASSERT(Mem, PI_ERROR_INVALID_VALUE);
  // piMemImageGetInfo must be used for images

  ur_mem_handle_t UrMemory = reinterpret_cast<ur_mem_handle_t>(Mem);
  ur_mem_info_t MemInfoType{};
  switch (ParamName) {
  case PI_MEM_CONTEXT: {
    MemInfoType = UR_MEM_INFO_CONTEXT;
    break;
  }
  case PI_MEM_SIZE: {
    MemInfoType = UR_MEM_INFO_SIZE;
    break;
  }
  default: {
    die("piMemGetInfo: unsuppported ParamName.");
  }
  }
  HANDLE_ERRORS(urMemGetInfo(UrMemory, MemInfoType, ParamValueSize, ParamValue,
                             ParamValueSizeRet));
  return PI_SUCCESS;
}

static void pi2urImageDesc(const pi_image_format *ImageFormat,
                           const pi_image_desc *ImageDesc,
                           ur_image_format_t *UrFormat,
                           ur_image_desc_t *UrDesc) {

  switch (ImageFormat->image_channel_data_type) {
  case PI_IMAGE_CHANNEL_TYPE_SNORM_INT8: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_SNORM_INT8;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_SNORM_INT16: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_SNORM_INT16;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_UNORM_INT8: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_UNORM_INT8;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_UNORM_INT16: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_UNORM_INT16;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_UNORM_INT_101010: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_INT_101010;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_SIGNED_INT8: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_SIGNED_INT16: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_SIGNED_INT32: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_HALF_FLOAT: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT;
    break;
  }
  case PI_IMAGE_CHANNEL_TYPE_FLOAT: {
    UrFormat->channelType = UR_IMAGE_CHANNEL_TYPE_FLOAT;
    break;
  }
  default: {
    die("piMemImageCreate: unsuppported image_channel_data_type.");
  }
  }
  switch (ImageFormat->image_channel_order) {
  case PI_IMAGE_CHANNEL_ORDER_A: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_A;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_R: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_R;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_RG: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_RG;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_RA: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_RA;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_RGB: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_RGB;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_RGBA: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_RGBA;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_BGRA: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_BGRA;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_ARGB: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_ARGB;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_ABGR: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_ABGR;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_INTENSITY: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_INTENSITY;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_LUMINANCE: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_LUMINANCE;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_Rx: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_RX;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_RGx: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_RGX;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_RGBx: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_RGBX;
    break;
  }
  case PI_IMAGE_CHANNEL_ORDER_sRGBA: {
    UrFormat->channelOrder = UR_IMAGE_CHANNEL_ORDER_SRGBA;
    break;
  }
  default: {
    die("piMemImageCreate: unsuppported image_channel_data_type.");
  }
  }

  UrDesc->stype = UR_STRUCTURE_TYPE_IMAGE_DESC;
  UrDesc->arraySize = ImageDesc->image_array_size;
  UrDesc->depth = ImageDesc->image_depth;
  UrDesc->height = ImageDesc->image_height;
  UrDesc->numMipLevel = ImageDesc->num_mip_levels;
  UrDesc->numSamples = ImageDesc->num_samples;
  UrDesc->rowPitch = ImageDesc->image_row_pitch;
  UrDesc->slicePitch = ImageDesc->image_slice_pitch;
  switch (ImageDesc->image_type) {
  case PI_MEM_TYPE_BUFFER: {
    UrDesc->type = UR_MEM_TYPE_BUFFER;
    break;
  }
  case PI_MEM_TYPE_IMAGE2D: {
    UrDesc->type = UR_MEM_TYPE_IMAGE2D;
    break;
  }
  case PI_MEM_TYPE_IMAGE3D: {
    UrDesc->type = UR_MEM_TYPE_IMAGE3D;
    break;
  }
  case PI_MEM_TYPE_IMAGE2D_ARRAY: {
    UrDesc->type = UR_MEM_TYPE_IMAGE2D_ARRAY;
    break;
  }
  case PI_MEM_TYPE_IMAGE1D: {
    UrDesc->type = UR_MEM_TYPE_IMAGE1D;
    break;
  }
  case PI_MEM_TYPE_IMAGE1D_ARRAY: {
    UrDesc->type = UR_MEM_TYPE_IMAGE1D_ARRAY;
    break;
  }
  case PI_MEM_TYPE_IMAGE1D_BUFFER: {
    UrDesc->type = UR_MEM_TYPE_IMAGE1D_BUFFER;
    break;
  }
  default: {
    die("piMemImageCreate: unsuppported image_type.");
  }
  }
  UrDesc->width = ImageDesc->image_width;
  UrDesc->arraySize = ImageDesc->image_array_size;
  UrDesc->arraySize = ImageDesc->image_array_size;
}

inline pi_result piMemImageCreate(pi_context Context, pi_mem_flags Flags,
                                  const pi_image_format *ImageFormat,
                                  const pi_image_desc *ImageDesc, void *HostPtr,
                                  pi_mem *RetImage) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(RetImage, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(ImageFormat, PI_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);

  ur_mem_flags_t UrFlags{};
  if (Flags & PI_MEM_FLAGS_ACCESS_RW) {
    UrFlags |= UR_MEM_FLAG_READ_WRITE;
  }
  if (Flags & PI_MEM_ACCESS_READ_ONLY) {
    UrFlags |= UR_MEM_FLAG_READ_ONLY;
  }
  if (Flags & PI_MEM_FLAGS_HOST_PTR_USE) {
    UrFlags |= UR_MEM_FLAG_USE_HOST_POINTER;
  }
  if (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) {
    UrFlags |= UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER;
  }
  if (Flags & PI_MEM_FLAGS_HOST_PTR_ALLOC) {
    UrFlags |= UR_MEM_FLAG_ALLOC_HOST_POINTER;
  }

  ur_image_format_t UrFormat{};
  ur_image_desc_t UrDesc{};
  pi2urImageDesc(ImageFormat, ImageDesc, &UrFormat, &UrDesc);

  // TODO: UrDesc doesn't have something for ImageDesc->buffer

  ur_mem_handle_t *UrMem = reinterpret_cast<ur_mem_handle_t *>(RetImage);
  HANDLE_ERRORS(
      urMemImageCreate(UrContext, UrFlags, &UrFormat, &UrDesc, HostPtr, UrMem));

  return PI_SUCCESS;
}

inline pi_result piextMemImageCreateWithNativeHandle(
    pi_native_handle NativeHandle, pi_context Context, bool OwnNativeHandle,
    const pi_image_format *ImageFormat, const pi_image_desc *ImageDesc,
    pi_mem *RetImage) {

  PI_ASSERT(RetImage, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  ur_native_handle_t UrNativeMem =
      reinterpret_cast<ur_native_handle_t>(NativeHandle);

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);

  ur_mem_handle_t *UrMem = reinterpret_cast<ur_mem_handle_t *>(RetImage);
  ur_mem_native_properties_t Properties{};
  Properties.isNativeHandleOwned = OwnNativeHandle;

  ur_image_format_t UrFormat{};
  ur_image_desc_t UrDesc{};
  pi2urImageDesc(ImageFormat, ImageDesc, &UrFormat, &UrDesc);

  HANDLE_ERRORS(urMemImageCreateWithNativeHandle(
      UrNativeMem, UrContext, &UrFormat, &UrDesc, &Properties, UrMem));

  return PI_SUCCESS;
}

inline pi_result piMemBufferPartition(pi_mem Buffer, pi_mem_flags Flags,
                                      pi_buffer_create_type BufferCreateType,
                                      void *BufferCreateInfo, pi_mem *RetMem) {

  PI_ASSERT(BufferCreateType == PI_BUFFER_CREATE_TYPE_REGION &&
                BufferCreateInfo && RetMem,
            PI_ERROR_INVALID_VALUE);

  auto Region = (pi_buffer_region)BufferCreateInfo;
  PI_ASSERT(Region->size != 0u, PI_ERROR_INVALID_BUFFER_SIZE);
  PI_ASSERT(Region->origin <= (Region->origin + Region->size),
            PI_ERROR_INVALID_VALUE);

  ur_mem_handle_t UrBuffer = reinterpret_cast<ur_mem_handle_t>(Buffer);

  ur_mem_flags_t UrFlags{};
  if (Flags & PI_MEM_FLAGS_ACCESS_RW) {
    UrFlags |= UR_MEM_FLAG_READ_WRITE;
  }
  if (Flags & PI_MEM_ACCESS_READ_ONLY) {
    UrFlags |= UR_MEM_FLAG_READ_ONLY;
  }
  if (Flags & PI_MEM_FLAGS_HOST_PTR_USE) {
    UrFlags |= UR_MEM_FLAG_USE_HOST_POINTER;
  }
  if (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) {
    UrFlags |= UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER;
  }
  if (Flags & PI_MEM_FLAGS_HOST_PTR_ALLOC) {
    UrFlags |= UR_MEM_FLAG_ALLOC_HOST_POINTER;
  }

  ur_buffer_create_type_t UrBufferCreateType{};
  if (BufferCreateType == PI_BUFFER_CREATE_TYPE_REGION) {
    UrBufferCreateType = UR_BUFFER_CREATE_TYPE_REGION;
  }

  ur_buffer_region_t UrBufferCreateInfo{};
  UrBufferCreateInfo.origin = Region->origin;
  UrBufferCreateInfo.size = Region->size;
  ur_mem_handle_t *UrMem = reinterpret_cast<ur_mem_handle_t *>(RetMem);
  HANDLE_ERRORS(urMemBufferPartition(UrBuffer, UrFlags, UrBufferCreateType,
                                     &UrBufferCreateInfo, UrMem));

  return PI_SUCCESS;
}

inline pi_result piextMemGetNativeHandle(pi_mem Mem,
                                         pi_native_handle *NativeHandle) {
  PI_ASSERT(Mem, PI_ERROR_INVALID_MEM_OBJECT);

  ur_mem_handle_t UrMem = reinterpret_cast<ur_mem_handle_t>(Mem);
  ur_native_handle_t NativeMem{};
  HANDLE_ERRORS(urMemGetNativeHandle(UrMem, &NativeMem));

  *NativeHandle = reinterpret_cast<pi_native_handle>(NativeMem);

  return PI_SUCCESS;
}

inline pi_result
piEnqueueMemImageCopy(pi_queue Queue, pi_mem SrcImage, pi_mem DstImage,
                      pi_image_offset SrcOrigin, pi_image_offset DstOrigin,
                      pi_image_region Region, pi_uint32 NumEventsInWaitList,
                      const pi_event *EventsWaitList, pi_event *OutEvent) {

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);

  ur_mem_handle_t UrImageSrc = reinterpret_cast<ur_mem_handle_t>(SrcImage);
  ur_mem_handle_t UrImageDst = reinterpret_cast<ur_mem_handle_t>(DstImage);

  ur_rect_offset_t UrSrcOrigin{SrcOrigin->x, SrcOrigin->y, SrcOrigin->z};
  ur_rect_offset_t UrDstOrigin{DstOrigin->x, DstOrigin->y, DstOrigin->z};
  ur_rect_region_t UrRegion{};
  UrRegion.depth = Region->depth;
  UrRegion.height = Region->height;
  UrRegion.width = Region->width;

  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemImageCopy(
      UrQueue, UrImageSrc, UrImageDst, UrSrcOrigin, UrDstOrigin, UrRegion,
      NumEventsInWaitList, UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

inline pi_result piextMemCreateWithNativeHandle(pi_native_handle NativeHandle,
                                                pi_context Context,
                                                bool OwnNativeHandle,
                                                pi_mem *Mem) {
  PI_ASSERT(Mem, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  ur_native_handle_t UrNativeMem =
      reinterpret_cast<ur_native_handle_t>(NativeHandle);
  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  ur_mem_handle_t *UrMem = reinterpret_cast<ur_mem_handle_t *>(Mem);
  // TODO: Pass OwnNativeHandle to the output parameter
  // while we get it in interface
  ur_mem_native_properties_t Properties{};
  Properties.isNativeHandleOwned = OwnNativeHandle;
  HANDLE_ERRORS(urMemBufferCreateWithNativeHandle(UrNativeMem, UrContext,
                                                  &Properties, UrMem));

  return PI_SUCCESS;
}

inline pi_result piextUSMDeviceAlloc(void **ResultPtr, pi_context Context,
                                     pi_device Device,
                                     pi_usm_mem_properties *Properties,
                                     size_t Size, pi_uint32 Alignment) {

  std::ignore = Properties;
  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);

  ur_usm_desc_t USMDesc{};
  USMDesc.align = Alignment;
  ur_usm_pool_handle_t Pool{};
  HANDLE_ERRORS(
      urUSMDeviceAlloc(UrContext, UrDevice, &USMDesc, Pool, Size, ResultPtr));

  return PI_SUCCESS;
}

inline pi_result piextUSMSharedAlloc(void **ResultPtr, pi_context Context,
                                     pi_device Device,
                                     pi_usm_mem_properties *Properties,
                                     size_t Size, pi_uint32 Alignment) {

  std::ignore = Properties;
  if (Properties && *Properties != 0) {
    PI_ASSERT(*(Properties) == PI_MEM_ALLOC_FLAGS && *(Properties + 2) == 0,
              PI_ERROR_INVALID_VALUE);
  }

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);

  ur_usm_desc_t USMDesc{};
  ur_usm_device_desc_t UsmDeviceDesc{};
  UsmDeviceDesc.stype = UR_STRUCTURE_TYPE_USM_DEVICE_DESC;
  ur_usm_host_desc_t UsmHostDesc{};
  UsmHostDesc.stype = UR_STRUCTURE_TYPE_USM_HOST_DESC;
  if (Properties) {
    if (Properties[0] == PI_MEM_ALLOC_FLAGS) {
      if (Properties[1] == PI_MEM_ALLOC_WRTITE_COMBINED) {
        UsmDeviceDesc.flags |= UR_USM_DEVICE_MEM_FLAG_WRITE_COMBINED;
      }
      if (Properties[1] == PI_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE) {
        UsmDeviceDesc.flags |= UR_USM_DEVICE_MEM_FLAG_INITIAL_PLACEMENT;
      }
      if (Properties[1] == PI_MEM_ALLOC_INITIAL_PLACEMENT_HOST) {
        UsmHostDesc.flags |= UR_USM_HOST_MEM_FLAG_INITIAL_PLACEMENT;
      }
      if (Properties[1] == PI_MEM_ALLOC_DEVICE_READ_ONLY) {
        UsmDeviceDesc.flags |= UR_USM_DEVICE_MEM_FLAG_DEVICE_READ_ONLY;
      }
    }
  }
  UsmDeviceDesc.pNext = &UsmHostDesc;
  USMDesc.pNext = &UsmDeviceDesc;

  USMDesc.align = Alignment;

  ur_usm_pool_handle_t Pool{};
  HANDLE_ERRORS(
      urUSMSharedAlloc(UrContext, UrDevice, &USMDesc, Pool, Size, ResultPtr));

  return PI_SUCCESS;
}

inline pi_result piextUSMFree(pi_context Context, void *Ptr) {
  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  HANDLE_ERRORS(urUSMFree(UrContext, Ptr));
  return PI_SUCCESS;
}

inline pi_result piMemRetain(pi_mem Mem) {
  PI_ASSERT(Mem, PI_ERROR_INVALID_MEM_OBJECT);

  ur_mem_handle_t UrMem = reinterpret_cast<ur_mem_handle_t>(Mem);

  HANDLE_ERRORS(urMemRetain(UrMem));

  return PI_SUCCESS;
}

inline pi_result piMemRelease(pi_mem Mem) {
  PI_ASSERT(Mem, PI_ERROR_INVALID_MEM_OBJECT);

  ur_mem_handle_t UrMem = reinterpret_cast<ur_mem_handle_t>(Mem);

  HANDLE_ERRORS(urMemRelease(UrMem));

  return PI_SUCCESS;
}

/// Hint to migrate memory to the device
///
/// @param Queue is the queue to submit to
/// @param Ptr points to the memory to migrate
/// @param Size is the number of bytes to migrate
/// @param Flags is a bitfield used to specify memory migration options
/// @param NumEventsInWaitList is the number of events to wait on
/// @param EventsWaitList is an array of events to wait on
/// @param Event is the event that represents this operation
inline pi_result piextUSMEnqueuePrefetch(pi_queue Queue, const void *Ptr,
                                         size_t Size,
                                         pi_usm_migration_flags Flags,
                                         pi_uint32 NumEventsInWaitList,
                                         const pi_event *EventsWaitList,
                                         pi_event *OutEvent) {

  // flags is currently unused so fail if set
  PI_ASSERT(Flags == 0, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);

  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  // TODO: to map from pi_usm_migration_flags to
  // ur_usm_migration_flags_t
  // once we have those defined
  ur_usm_migration_flags_t UrFlags{};
  HANDLE_ERRORS(urEnqueueUSMPrefetch(UrQueue, Ptr, Size, UrFlags,
                                     NumEventsInWaitList, UrEventsWaitList,
                                     UrEvent));

  return PI_SUCCESS;
}

/// USM memadvise API to govern behavior of automatic migration mechanisms
///
/// @param Queue is the queue to submit to
/// @param Ptr is the data to be advised
/// @param Length is the size in bytes of the meory to advise
/// @param Advice is device specific advice
/// @param Event is the event that represents this operation
///
inline pi_result piextUSMEnqueueMemAdvise(pi_queue Queue, const void *Ptr,
                                          size_t Length, pi_mem_advice Advice,
                                          pi_event *OutEvent) {

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  ur_usm_advice_flags_t UrAdvice{};
  if (Advice & PI_MEM_ADVICE_CUDA_SET_READ_MOSTLY) {
    UrAdvice |= UR_USM_ADVICE_FLAG_SET_READ_MOSTLY;
  }
  if (Advice & PI_MEM_ADVICE_CUDA_UNSET_READ_MOSTLY) {
    UrAdvice |= UR_USM_ADVICE_FLAG_CLEAR_READ_MOSTLY;
  }
  if (Advice & PI_MEM_ADVICE_CUDA_SET_PREFERRED_LOCATION) {
    UrAdvice |= UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION;
  }
  if (Advice & PI_MEM_ADVICE_CUDA_UNSET_PREFERRED_LOCATION) {
    UrAdvice |= UR_USM_ADVICE_FLAG_CLEAR_PREFERRED_LOCATION;
  }
  if (Advice & PI_MEM_ADVICE_RESET) {
    UrAdvice |= UR_USM_ADVICE_FLAG_DEFAULT;
  }

  HANDLE_ERRORS(urEnqueueUSMAdvise(UrQueue, Ptr, Length, UrAdvice, UrEvent));

  return PI_SUCCESS;
}

/// USM 2D Fill API
///
/// \param queue is the queue to submit to
/// \param ptr is the ptr to fill
/// \param pitch is the total width of the destination memory including padding
/// \param pattern is a pointer with the bytes of the pattern to set
/// \param pattern_size is the size in bytes of the pattern
/// \param width is width in bytes of each row to fill
/// \param height is height the columns to fill
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
inline pi_result piextUSMEnqueueFill2D(pi_queue Queue, void *Ptr, size_t Pitch,
                                       size_t PatternSize, const void *Pattern,
                                       size_t Width, size_t Height,
                                       pi_uint32 NumEventsWaitList,
                                       const pi_event *EventsWaitList,
                                       pi_event *Event) {

  auto hQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  auto phEventWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);
  auto phEvent = reinterpret_cast<ur_event_handle_t *>(Event);

  HANDLE_ERRORS(urEnqueueUSMFill2D(hQueue, Ptr, Pitch, PatternSize, Pattern,
                                   Width, Height, NumEventsWaitList,
                                   phEventWaitList, phEvent));

  return PI_SUCCESS;
}

inline pi_result piextUSMEnqueueMemset2D(pi_queue Queue, void *Ptr,
                                         size_t Pitch, int Value, size_t Width,
                                         size_t Height,
                                         pi_uint32 NumEventsWaitList,
                                         const pi_event *EventsWaitList,
                                         pi_event *Event) {
  std::ignore = Queue;
  std::ignore = Ptr;
  std::ignore = Pitch;
  std::ignore = Value;
  std::ignore = Width;
  std::ignore = Height;
  std::ignore = NumEventsWaitList;
  std::ignore = EventsWaitList;
  std::ignore = Event;
  die("piextUSMEnqueueMemset2D: not implemented");
  return PI_SUCCESS;
}

inline pi_result piextUSMGetMemAllocInfo(pi_context Context, const void *Ptr,
                                         pi_mem_alloc_info ParamName,
                                         size_t ParamValueSize,
                                         void *ParamValue,
                                         size_t *ParamValueSizeRet) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);

  ur_usm_alloc_info_t UrParamName{};
  switch (ParamName) {
  case PI_MEM_ALLOC_TYPE: {
    UrParamName = UR_USM_ALLOC_INFO_TYPE;
    break;
  }
  case PI_MEM_ALLOC_BASE_PTR: {
    UrParamName = UR_USM_ALLOC_INFO_BASE_PTR;
    break;
  }
  case PI_MEM_ALLOC_SIZE: {
    UrParamName = UR_USM_ALLOC_INFO_SIZE;
    break;
  }
  case PI_MEM_ALLOC_DEVICE: {
    UrParamName = UR_USM_ALLOC_INFO_DEVICE;
    break;
  }
  default: {
    die("piextUSMGetMemAllocInfo: unsuppported ParamName.");
  }
  }

  size_t SizeInOut = ParamValueSize;
  HANDLE_ERRORS(urUSMGetMemAllocInfo(UrContext, Ptr, UrParamName,
                                     ParamValueSize, ParamValue,
                                     ParamValueSizeRet))
  ur2piUSMAllocInfoValue(UrParamName, ParamValueSize, &SizeInOut, ParamValue);
  return PI_SUCCESS;
}

inline pi_result piMemImageGetInfo(pi_mem Image, pi_image_info ParamName,
                                   size_t ParamValueSize, void *ParamValue,
                                   size_t *ParamValueSizeRet) {

  auto hMem = reinterpret_cast<ur_mem_handle_t>(Image);

  ur_image_info_t UrParamName{};
  switch (ParamName) {
  case PI_IMAGE_INFO_FORMAT: {
    UrParamName = UR_IMAGE_INFO_FORMAT;
    break;
  }
  case PI_IMAGE_INFO_ELEMENT_SIZE: {
    UrParamName = UR_IMAGE_INFO_ELEMENT_SIZE;
    break;
  }
  case PI_IMAGE_INFO_ROW_PITCH: {
    UrParamName = UR_IMAGE_INFO_ROW_PITCH;
    break;
  }
  case PI_IMAGE_INFO_SLICE_PITCH: {
    UrParamName = UR_IMAGE_INFO_SLICE_PITCH;
    break;
  }
  case PI_IMAGE_INFO_WIDTH: {
    UrParamName = UR_IMAGE_INFO_WIDTH;
    break;
  }
  case PI_IMAGE_INFO_HEIGHT: {
    UrParamName = UR_IMAGE_INFO_HEIGHT;
    break;
  }
  case PI_IMAGE_INFO_DEPTH: {
    UrParamName = UR_IMAGE_INFO_DEPTH;
    break;
  }
  default:
    return PI_ERROR_UNKNOWN;
  }

  HANDLE_ERRORS(urMemImageGetInfo(hMem, UrParamName, ParamValueSize, ParamValue,
                                  ParamValueSizeRet));
  return PI_SUCCESS;
}

/// USM 2D Memcpy API
///
/// \param queue is the queue to submit to
/// \param blocking is whether this operation should block the host
/// \param dst_ptr is the location the data will be copied
/// \param dst_pitch is the total width of the destination memory including
/// padding
/// \param src_ptr is the data to be copied
/// \param dst_pitch is the total width of the source memory including padding
/// \param width is width in bytes of each row to be copied
/// \param height is height the columns to be copied
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
inline pi_result piextUSMEnqueueMemcpy2D(pi_queue Queue, pi_bool Blocking,
                                         void *DstPtr, size_t DstPitch,
                                         const void *SrcPtr, size_t SrcPitch,
                                         size_t Width, size_t Height,
                                         pi_uint32 NumEventsInWaitList,
                                         const pi_event *EventsWaitList,
                                         pi_event *OutEvent) {

  if (!DstPtr || !SrcPtr)
    return PI_ERROR_INVALID_VALUE;

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueUSMMemcpy2D(
      UrQueue, Blocking, DstPtr, DstPitch, SrcPtr, SrcPitch, Width, Height,
      NumEventsInWaitList, UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

// Memory
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Enqueue

inline pi_result
piEnqueueKernelLaunch(pi_queue Queue, pi_kernel Kernel, pi_uint32 WorkDim,
                      const size_t *GlobalWorkOffset,
                      const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
                      pi_uint32 NumEventsInWaitList,
                      const pi_event *EventsWaitList, pi_event *OutEvent) {

  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);
  PI_ASSERT((WorkDim > 0) && (WorkDim < 4), PI_ERROR_INVALID_WORK_DIMENSION);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueKernelLaunch(
      UrQueue, UrKernel, WorkDim, GlobalWorkOffset, GlobalWorkSize,
      LocalWorkSize, NumEventsInWaitList, UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

inline pi_result
piEnqueueMemImageWrite(pi_queue Queue, pi_mem Image, pi_bool BlockingWrite,
                       pi_image_offset Origin, pi_image_region Region,
                       size_t InputRowPitch, size_t InputSlicePitch,
                       const void *Ptr, pi_uint32 NumEventsInWaitList,
                       const pi_event *EventsWaitList, pi_event *OutEvent) {

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_mem_handle_t UrImage = reinterpret_cast<ur_mem_handle_t>(Image);
  ur_rect_offset_t UrOrigin{Origin->x, Origin->y, Origin->z};
  ur_rect_region_t UrRegion{};
  UrRegion.depth = Region->depth;
  UrRegion.height = Region->height;
  UrRegion.width = Region->width;
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemImageWrite(
      UrQueue, UrImage, BlockingWrite, UrOrigin, UrRegion, InputRowPitch,
      InputSlicePitch, const_cast<void *>(Ptr), NumEventsInWaitList,
      UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

inline pi_result
piEnqueueMemImageRead(pi_queue Queue, pi_mem Image, pi_bool BlockingRead,
                      pi_image_offset Origin, pi_image_region Region,
                      size_t RowPitch, size_t SlicePitch, void *Ptr,
                      pi_uint32 NumEventsInWaitList,
                      const pi_event *EventsWaitList, pi_event *OutEvent) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_mem_handle_t UrImage = reinterpret_cast<ur_mem_handle_t>(Image);
  ur_rect_offset_t UrOrigin{Origin->x, Origin->y, Origin->z};
  ur_rect_region_t UrRegion{};
  UrRegion.depth = Region->depth;
  UrRegion.height = Region->height;
  UrRegion.width = Region->width;
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemImageRead(
      UrQueue, UrImage, BlockingRead, UrOrigin, UrRegion, RowPitch, SlicePitch,
      Ptr, NumEventsInWaitList, UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

inline pi_result piEnqueueMemBufferMap(
    pi_queue Queue, pi_mem Mem, pi_bool BlockingMap, pi_map_flags MapFlags,
    size_t Offset, size_t Size, pi_uint32 NumEventsInWaitList,
    const pi_event *EventsWaitList, pi_event *OutEvent, void **RetMap) {
  // TODO: we don't implement read-only or write-only, always read-write.
  // assert((map_flags & PI_MAP_READ) != 0);
  // assert((map_flags & PI_MAP_WRITE) != 0);
  PI_ASSERT(Mem, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_mem_handle_t UrMem = reinterpret_cast<ur_mem_handle_t>(Mem);

  ur_map_flags_t UrMapFlags{};
  if (MapFlags & PI_MAP_READ)
    UrMapFlags |= UR_MAP_FLAG_READ;
  if (MapFlags & PI_MAP_WRITE)
    UrMapFlags |= UR_MAP_FLAG_WRITE;
  if (MapFlags & PI_MAP_WRITE_INVALIDATE_REGION)
    UrMapFlags |= UR_MAP_FLAG_WRITE_INVALIDATE_REGION;

  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferMap(UrQueue, UrMem, BlockingMap, UrMapFlags,
                                      Offset, Size, NumEventsInWaitList,
                                      UrEventsWaitList, UrEvent, RetMap));

  return PI_SUCCESS;
}

inline pi_result piEnqueueMemUnmap(pi_queue Queue, pi_mem Mem, void *MappedPtr,
                                   pi_uint32 NumEventsInWaitList,
                                   const pi_event *EventsWaitList,
                                   pi_event *OutEvent) {

  PI_ASSERT(Mem, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_mem_handle_t UrMem = reinterpret_cast<ur_mem_handle_t>(Mem);
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemUnmap(UrQueue, UrMem, MappedPtr,
                                  NumEventsInWaitList, UrEventsWaitList,
                                  UrEvent));

  return PI_SUCCESS;
}

inline pi_result piEnqueueMemBufferFill(pi_queue Queue, pi_mem Buffer,
                                        const void *Pattern, size_t PatternSize,
                                        size_t Offset, size_t Size,
                                        pi_uint32 NumEventsInWaitList,
                                        const pi_event *EventsWaitList,
                                        pi_event *OutEvent) {
  PI_ASSERT(Buffer, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_mem_handle_t UrBuffer = reinterpret_cast<ur_mem_handle_t>(Buffer);
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferFill(UrQueue, UrBuffer, Pattern, PatternSize,
                                       Offset, Size, NumEventsInWaitList,
                                       UrEventsWaitList, UrEvent));
  return PI_SUCCESS;
}

inline pi_result piextUSMEnqueueMemset(pi_queue Queue, void *Ptr,
                                       pi_int32 Value, size_t Count,
                                       pi_uint32 NumEventsInWaitList,
                                       const pi_event *EventsWaitList,
                                       pi_event *OutEvent) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);
  if (!Ptr) {
    return PI_ERROR_INVALID_VALUE;
  }

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  size_t PatternSize = 1;
  HANDLE_ERRORS(urEnqueueUSMFill(UrQueue, Ptr, PatternSize, &Value, Count,
                                 NumEventsInWaitList, UrEventsWaitList,
                                 UrEvent));

  return PI_SUCCESS;
}

inline pi_result piEnqueueMemBufferCopyRect(
    pi_queue Queue, pi_mem SrcMem, pi_mem DstMem, pi_buff_rect_offset SrcOrigin,
    pi_buff_rect_offset DstOrigin, pi_buff_rect_region Region,
    size_t SrcRowPitch, size_t SrcSlicePitch, size_t DstRowPitch,
    size_t DstSlicePitch, pi_uint32 NumEventsInWaitList,
    const pi_event *EventsWaitList, pi_event *OutEvent) {

  PI_ASSERT(SrcMem && DstMem, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_mem_handle_t UrBufferSrc = reinterpret_cast<ur_mem_handle_t>(SrcMem);
  ur_mem_handle_t UrBufferDst = reinterpret_cast<ur_mem_handle_t>(DstMem);
  ur_rect_offset_t UrSrcOrigin{SrcOrigin->x_bytes, SrcOrigin->y_scalar,
                               SrcOrigin->z_scalar};
  ur_rect_offset_t UrDstOrigin{DstOrigin->x_bytes, DstOrigin->y_scalar,
                               DstOrigin->z_scalar};
  ur_rect_region_t UrRegion{};
  UrRegion.depth = Region->depth_scalar;
  UrRegion.height = Region->height_scalar;
  UrRegion.width = Region->width_bytes;
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferCopyRect(
      UrQueue, UrBufferSrc, UrBufferDst, UrSrcOrigin, UrDstOrigin, UrRegion,
      SrcRowPitch, SrcSlicePitch, DstRowPitch, DstSlicePitch,
      NumEventsInWaitList, UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

inline pi_result piEnqueueMemBufferCopy(pi_queue Queue, pi_mem SrcMem,
                                        pi_mem DstMem, size_t SrcOffset,
                                        size_t DstOffset, size_t Size,
                                        pi_uint32 NumEventsInWaitList,
                                        const pi_event *EventsWaitList,
                                        pi_event *OutEvent) {

  PI_ASSERT(SrcMem && DstMem, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_mem_handle_t UrBufferSrc = reinterpret_cast<ur_mem_handle_t>(SrcMem);
  ur_mem_handle_t UrBufferDst = reinterpret_cast<ur_mem_handle_t>(DstMem);
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferCopy(
      UrQueue, UrBufferSrc, UrBufferDst, SrcOffset, DstOffset, Size,
      NumEventsInWaitList, UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

inline pi_result piextUSMEnqueueMemcpy(pi_queue Queue, pi_bool Blocking,
                                       void *DstPtr, const void *SrcPtr,
                                       size_t Size,
                                       pi_uint32 NumEventsInWaitList,
                                       const pi_event *EventsWaitList,
                                       pi_event *OutEvent) {

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueUSMMemcpy(UrQueue, Blocking, DstPtr, SrcPtr, Size,
                                   NumEventsInWaitList, UrEventsWaitList,
                                   UrEvent));

  return PI_SUCCESS;
}

inline pi_result piEnqueueMemBufferWriteRect(
    pi_queue Queue, pi_mem Buffer, pi_bool BlockingWrite,
    pi_buff_rect_offset BufferOffset, pi_buff_rect_offset HostOffset,
    pi_buff_rect_region Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, const void *Ptr,
    pi_uint32 NumEventsInWaitList, const pi_event *EventsWaitList,
    pi_event *OutEvent) {

  PI_ASSERT(Buffer, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_mem_handle_t UrBuffer = reinterpret_cast<ur_mem_handle_t>(Buffer);
  ur_rect_offset_t UrBufferOffset{BufferOffset->x_bytes, BufferOffset->y_scalar,
                                  BufferOffset->z_scalar};
  ur_rect_offset_t UrHostOffset{HostOffset->x_bytes, HostOffset->y_scalar,
                                HostOffset->z_scalar};
  ur_rect_region_t UrRegion{};
  UrRegion.depth = Region->depth_scalar;
  UrRegion.height = Region->height_scalar;
  UrRegion.width = Region->width_bytes;
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferWriteRect(
      UrQueue, UrBuffer, BlockingWrite, UrBufferOffset, UrHostOffset, UrRegion,
      BufferRowPitch, BufferSlicePitch, HostRowPitch, HostSlicePitch,
      const_cast<void *>(Ptr), NumEventsInWaitList, UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

inline pi_result piEnqueueMemBufferWrite(pi_queue Queue, pi_mem Buffer,
                                         pi_bool BlockingWrite, size_t Offset,
                                         size_t Size, const void *Ptr,
                                         pi_uint32 NumEventsInWaitList,
                                         const pi_event *EventsWaitList,
                                         pi_event *OutEvent) {

  PI_ASSERT(Buffer, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_mem_handle_t UrBuffer = reinterpret_cast<ur_mem_handle_t>(Buffer);
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferWrite(
      UrQueue, UrBuffer, BlockingWrite, Offset, Size, const_cast<void *>(Ptr),
      NumEventsInWaitList, UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

inline pi_result piEnqueueMemBufferReadRect(
    pi_queue Queue, pi_mem Buffer, pi_bool BlockingRead,
    pi_buff_rect_offset BufferOffset, pi_buff_rect_offset HostOffset,
    pi_buff_rect_region Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, void *Ptr,
    pi_uint32 NumEventsInWaitList, const pi_event *EventsWaitList,
    pi_event *OutEvent) {

  PI_ASSERT(Buffer, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_mem_handle_t UrBuffer = reinterpret_cast<ur_mem_handle_t>(Buffer);
  ur_rect_offset_t UrBufferOffset{BufferOffset->x_bytes, BufferOffset->y_scalar,
                                  BufferOffset->z_scalar};
  ur_rect_offset_t UrHostOffset{HostOffset->x_bytes, HostOffset->y_scalar,
                                HostOffset->z_scalar};
  ur_rect_region_t UrRegion{};
  UrRegion.depth = Region->depth_scalar;
  UrRegion.height = Region->height_scalar;
  UrRegion.width = Region->width_bytes;

  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferReadRect(
      UrQueue, UrBuffer, BlockingRead, UrBufferOffset, UrHostOffset, UrRegion,
      BufferRowPitch, BufferSlicePitch, HostRowPitch, HostSlicePitch, Ptr,
      NumEventsInWaitList, UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

inline pi_result piEnqueueMemBufferRead(pi_queue Queue, pi_mem Src,
                                        pi_bool BlockingRead, size_t Offset,
                                        size_t Size, void *Dst,
                                        pi_uint32 NumEventsInWaitList,
                                        const pi_event *EventsWaitList,
                                        pi_event *OutEvent) {
  PI_ASSERT(Src, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  ur_mem_handle_t UrBuffer = reinterpret_cast<ur_mem_handle_t>(Src);
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferRead(UrQueue, UrBuffer, BlockingRead, Offset,
                                       Size, Dst, NumEventsInWaitList,
                                       UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

inline pi_result piEnqueueEventsWaitWithBarrier(pi_queue Queue,
                                                pi_uint32 NumEventsInWaitList,
                                                const pi_event *EventsWaitList,
                                                pi_event *OutEvent) {

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueEventsWaitWithBarrier(UrQueue, NumEventsInWaitList,
                                               UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

inline pi_result piEnqueueEventsWait(pi_queue Queue,
                                     pi_uint32 NumEventsInWaitList,
                                     const pi_event *EventsWaitList,
                                     pi_event *OutEvent) {

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);
  if (EventsWaitList) {
    PI_ASSERT(NumEventsInWaitList > 0, PI_ERROR_INVALID_VALUE);
  }

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueEventsWait(UrQueue, NumEventsInWaitList,
                                    UrEventsWaitList, UrEvent));

  return PI_SUCCESS;
}

inline pi_result
piextEnqueueReadHostPipe(pi_queue queue, pi_program program,
                         const char *pipe_symbol, pi_bool blocking, void *ptr,
                         size_t size, pi_uint32 num_events_in_waitlist,
                         const pi_event *events_waitlist, pi_event *event) {
  auto hQueue = reinterpret_cast<ur_queue_handle_t>(queue);
  auto hProgram = reinterpret_cast<ur_program_handle_t>(program);
  auto phEventWaitList =
      reinterpret_cast<const ur_event_handle_t *>(events_waitlist);
  auto phEvent = reinterpret_cast<ur_event_handle_t *>(event);

  HANDLE_ERRORS(urEnqueueReadHostPipe(hQueue, hProgram, pipe_symbol, blocking,
                                      ptr, size, num_events_in_waitlist,
                                      phEventWaitList, phEvent));

  return PI_SUCCESS;
}

inline pi_result
piextEnqueueWriteHostPipe(pi_queue queue, pi_program program,
                          const char *pipe_symbol, pi_bool blocking, void *ptr,
                          size_t size, pi_uint32 num_events_in_waitlist,
                          const pi_event *events_waitlist, pi_event *event) {
  auto hQueue = reinterpret_cast<ur_queue_handle_t>(queue);
  auto hProgram = reinterpret_cast<ur_program_handle_t>(program);
  auto phEventWaitList =
      reinterpret_cast<const ur_event_handle_t *>(events_waitlist);
  auto phEvent = reinterpret_cast<ur_event_handle_t *>(event);

  HANDLE_ERRORS(urEnqueueWriteHostPipe(hQueue, hProgram, pipe_symbol, blocking,
                                       ptr, size, num_events_in_waitlist,
                                       phEventWaitList, phEvent));

  return PI_SUCCESS;
}
// Enqueue
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Events
inline pi_result piEventsWait(pi_uint32 NumEvents,
                              const pi_event *EventsWaitList) {
  if (NumEvents && !EventsWaitList) {
    return PI_ERROR_INVALID_EVENT;
  }

  const ur_event_handle_t *UrEventsWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventsWaitList);

  HANDLE_ERRORS(urEventWait(NumEvents, UrEventsWaitList));

  return PI_SUCCESS;
}

inline pi_result piEventGetInfo(pi_event Event, pi_event_info ParamName,
                                size_t ParamValueSize, void *ParamValue,
                                size_t *ParamValueSizeRet) {

  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);

  ur_event_handle_t UrEvent = reinterpret_cast<ur_event_handle_t>(Event);

  ur_event_info_t PropName{};
  if (ParamName == PI_EVENT_INFO_COMMAND_QUEUE) {
    PropName = UR_EVENT_INFO_COMMAND_QUEUE;
  } else if (ParamName == PI_EVENT_INFO_CONTEXT) {
    PropName = UR_EVENT_INFO_CONTEXT;
  } else if (ParamName == PI_EVENT_INFO_COMMAND_TYPE) {
    PropName = UR_EVENT_INFO_COMMAND_TYPE;
  } else if (ParamName == PI_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
    PropName = UR_EVENT_INFO_COMMAND_EXECUTION_STATUS;
  } else if (ParamName == PI_EVENT_INFO_REFERENCE_COUNT) {
    PropName = UR_EVENT_INFO_REFERENCE_COUNT;
  } else {
    return PI_ERROR_INVALID_VALUE;
  }

  HANDLE_ERRORS(urEventGetInfo(UrEvent, PropName, ParamValueSize, ParamValue,
                               ParamValueSizeRet));

  return PI_SUCCESS;
}

inline pi_result piextEventGetNativeHandle(pi_event Event,
                                           pi_native_handle *NativeHandle) {

  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  ur_event_handle_t UrEvent = reinterpret_cast<ur_event_handle_t>(Event);

  ur_native_handle_t *UrNativeEvent =
      reinterpret_cast<ur_native_handle_t *>(NativeHandle);
  HANDLE_ERRORS(urEventGetNativeHandle(UrEvent, UrNativeEvent));

  return PI_SUCCESS;
}

inline pi_result piEventGetProfilingInfo(pi_event Event,
                                         pi_profiling_info ParamName,
                                         size_t ParamValueSize,
                                         void *ParamValue,
                                         size_t *ParamValueSizeRet) {

  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);

  ur_event_handle_t UrEvent = reinterpret_cast<ur_event_handle_t>(Event);

  ur_profiling_info_t PropName{};
  switch (ParamName) {
  case PI_PROFILING_INFO_COMMAND_QUEUED: {
    PropName = UR_PROFILING_INFO_COMMAND_QUEUED;
    break;
  }
  case PI_PROFILING_INFO_COMMAND_SUBMIT: {
    PropName = UR_PROFILING_INFO_COMMAND_SUBMIT;
    break;
  }
  case PI_PROFILING_INFO_COMMAND_START: {
    PropName = UR_PROFILING_INFO_COMMAND_START;
    break;
  }
  case PI_PROFILING_INFO_COMMAND_END: {
    PropName = UR_PROFILING_INFO_COMMAND_END;
    break;
  }
  default:
    return PI_ERROR_INVALID_PROPERTY;
  }

  HANDLE_ERRORS(urEventGetProfilingInfo(UrEvent, PropName, ParamValueSize,
                                        ParamValue, ParamValueSizeRet));

  return PI_SUCCESS;
}

inline pi_result piEventCreate(pi_context Context, pi_event *RetEvent) {

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(RetEvent);
  // pass null for the hNativeHandle to use urEventCreateWithNativeHandle
  // as urEventCreate
  ur_event_native_properties_t Properties{};
  HANDLE_ERRORS(
      urEventCreateWithNativeHandle(nullptr, UrContext, &Properties, UrEvent));

  return PI_SUCCESS;
}

inline pi_result piextEventCreateWithNativeHandle(pi_native_handle NativeHandle,
                                                  pi_context Context,
                                                  bool OwnNativeHandle,
                                                  pi_event *Event) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  ur_native_handle_t UrNativeKernel =
      reinterpret_cast<ur_native_handle_t>(NativeHandle);

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);

  ur_event_handle_t *UrEvent = reinterpret_cast<ur_event_handle_t *>(Event);
  ur_event_native_properties_t Properties{};
  Properties.isNativeHandleOwned = OwnNativeHandle;
  HANDLE_ERRORS(urEventCreateWithNativeHandle(UrNativeKernel, UrContext,
                                              &Properties, UrEvent));

  return PI_SUCCESS;
}

inline pi_result piEventSetCallback(
    pi_event Event, pi_int32 CommandExecCallbackType,
    void (*PFnNotify)(pi_event Event, pi_int32 EventCommandStatus,
                      void *UserData),
    void *UserData) {
  std::ignore = Event;
  std::ignore = CommandExecCallbackType;
  std::ignore = PFnNotify;
  std::ignore = UserData;
  die("piEventSetCallback: deprecated, to be removed");
  return PI_SUCCESS;
}

inline pi_result piEventSetStatus(pi_event Event, pi_int32 ExecutionStatus) {
  std::ignore = Event;
  std::ignore = ExecutionStatus;
  die("piEventSetStatus: deprecated, to be removed");
  return PI_SUCCESS;
}

inline pi_result piEventRetain(pi_event Event) {
  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);

  ur_event_handle_t UrEvent = reinterpret_cast<ur_event_handle_t>(Event);
  HANDLE_ERRORS(urEventRetain(UrEvent));

  return PI_SUCCESS;
}

inline pi_result piEventRelease(pi_event Event) {
  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);

  ur_event_handle_t UrEvent = reinterpret_cast<ur_event_handle_t>(Event);
  HANDLE_ERRORS(urEventRelease(UrEvent));

  return PI_SUCCESS;
}

// Events
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Sampler
inline pi_result piSamplerCreate(pi_context Context,
                                 const pi_sampler_properties *SamplerProperties,
                                 pi_sampler *RetSampler) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(RetSampler, PI_ERROR_INVALID_VALUE);

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  ur_sampler_desc_t UrProps{};
  UrProps.stype = UR_STRUCTURE_TYPE_SAMPLER_DESC;
  const pi_sampler_properties *CurProperty = SamplerProperties;
  while (*CurProperty != 0) {
    switch (*CurProperty) {
    case PI_SAMPLER_PROPERTIES_NORMALIZED_COORDS: {
      UrProps.normalizedCoords = ur_cast<pi_bool>(*(++CurProperty));
    } break;

    case PI_SAMPLER_PROPERTIES_ADDRESSING_MODE: {
      pi_sampler_addressing_mode CurValueAddressingMode =
          ur_cast<pi_sampler_addressing_mode>(
              ur_cast<pi_uint32>(*(++CurProperty)));

      if (CurValueAddressingMode == PI_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT)
        UrProps.addressingMode = UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT;
      else if (CurValueAddressingMode == PI_SAMPLER_ADDRESSING_MODE_REPEAT)
        UrProps.addressingMode = UR_SAMPLER_ADDRESSING_MODE_REPEAT;
      else if (CurValueAddressingMode ==
               PI_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE)
        UrProps.addressingMode = UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE;
      else if (CurValueAddressingMode == PI_SAMPLER_ADDRESSING_MODE_CLAMP)
        UrProps.addressingMode = UR_SAMPLER_ADDRESSING_MODE_CLAMP;
      else if (CurValueAddressingMode == PI_SAMPLER_ADDRESSING_MODE_NONE)
        UrProps.addressingMode = UR_SAMPLER_ADDRESSING_MODE_NONE;
    } break;

    case PI_SAMPLER_PROPERTIES_FILTER_MODE: {
      pi_sampler_filter_mode CurValueFilterMode =
          ur_cast<pi_sampler_filter_mode>(ur_cast<pi_uint32>(*(++CurProperty)));

      if (CurValueFilterMode == PI_SAMPLER_FILTER_MODE_NEAREST)
        UrProps.filterMode = UR_SAMPLER_FILTER_MODE_NEAREST;
      else if (CurValueFilterMode == PI_SAMPLER_FILTER_MODE_LINEAR)
        UrProps.filterMode = UR_SAMPLER_FILTER_MODE_LINEAR;
    } break;

    default:
      break;
    }
    CurProperty++;
  }

  ur_sampler_handle_t *UrSampler =
      reinterpret_cast<ur_sampler_handle_t *>(RetSampler);

  HANDLE_ERRORS(urSamplerCreate(UrContext, &UrProps, UrSampler));

  return PI_SUCCESS;
}

inline pi_result piSamplerGetInfo(pi_sampler Sampler, pi_sampler_info ParamName,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {
  ur_sampler_info_t InfoType{};
  switch (ParamName) {
  case PI_SAMPLER_INFO_REFERENCE_COUNT:
    InfoType = UR_SAMPLER_INFO_REFERENCE_COUNT;
    break;
  case PI_SAMPLER_INFO_CONTEXT:
    InfoType = UR_SAMPLER_INFO_CONTEXT;
    break;
  case PI_SAMPLER_INFO_NORMALIZED_COORDS:
    InfoType = UR_SAMPLER_INFO_NORMALIZED_COORDS;
    break;
  case PI_SAMPLER_INFO_ADDRESSING_MODE:
    InfoType = UR_SAMPLER_INFO_ADDRESSING_MODE;
    break;
  case PI_SAMPLER_INFO_FILTER_MODE:
    InfoType = UR_SAMPLER_INFO_FILTER_MODE;
    break;
  default:
    return PI_ERROR_UNKNOWN;
  }

  size_t UrParamValueSizeRet;
  auto hSampler = reinterpret_cast<ur_sampler_handle_t>(Sampler);
  HANDLE_ERRORS(urSamplerGetInfo(hSampler, InfoType, ParamValueSize, ParamValue,
                                 &UrParamValueSizeRet));
  if (ParamValueSizeRet) {
    *ParamValueSizeRet = UrParamValueSizeRet;
  }
  ur2piSamplerInfoValue(InfoType, ParamValueSize, &ParamValueSize, ParamValue);
  fixupInfoValueTypes(UrParamValueSizeRet, ParamValueSizeRet, ParamValueSize,
                      ParamValue);
  return PI_SUCCESS;
}

// Special version of piKernelSetArg to accept pi_sampler.
inline pi_result piextKernelSetArgSampler(pi_kernel Kernel, pi_uint32 ArgIndex,
                                          const pi_sampler *ArgValue) {
  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);
  ur_sampler_handle_t UrSampler =
      reinterpret_cast<ur_sampler_handle_t>(*ArgValue);

  HANDLE_ERRORS(urKernelSetArgSampler(UrKernel, ArgIndex, UrSampler));

  return PI_SUCCESS;
}

inline pi_result piSamplerRetain(pi_sampler Sampler) {
  PI_ASSERT(Sampler, PI_ERROR_INVALID_SAMPLER);

  ur_sampler_handle_t UrSampler =
      reinterpret_cast<ur_sampler_handle_t>(Sampler);

  HANDLE_ERRORS(urSamplerRetain(UrSampler));

  return PI_SUCCESS;
}

inline pi_result piSamplerRelease(pi_sampler Sampler) {
  PI_ASSERT(Sampler, PI_ERROR_INVALID_SAMPLER);

  ur_sampler_handle_t UrSampler =
      reinterpret_cast<ur_sampler_handle_t>(Sampler);

  HANDLE_ERRORS(urSamplerRelease(UrSampler));

  return PI_SUCCESS;
}

// Sampler
///////////////////////////////////////////////////////////////////////////////

} // namespace pi2ur
