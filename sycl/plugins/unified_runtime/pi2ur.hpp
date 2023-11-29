//===---------------- pi2ur.hpp - PI API to UR API  ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "ur_api.h"
#include <cstdarg>
#include <sycl/detail/cuda_definitions.hpp>
#include <sycl/detail/pi.h>
#include <ur/ur.hpp>

// Map of UR error codes to PI error codes
static pi_result ur2piResult(ur_result_t urResult) {
  if (urResult == UR_RESULT_SUCCESS)
    return PI_SUCCESS;

  switch (urResult) {
  case UR_RESULT_ERROR_INVALID_OPERATION:
    return PI_ERROR_INVALID_OPERATION;
  case UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES:
    return PI_ERROR_INVALID_QUEUE_PROPERTIES;
  case UR_RESULT_ERROR_INVALID_QUEUE:
    return PI_ERROR_INVALID_QUEUE;
  case UR_RESULT_ERROR_INVALID_VALUE:
    return PI_ERROR_INVALID_VALUE;
  case UR_RESULT_ERROR_INVALID_CONTEXT:
    return PI_ERROR_INVALID_CONTEXT;
  case UR_RESULT_ERROR_INVALID_PLATFORM:
    return PI_ERROR_INVALID_PLATFORM;
  case UR_RESULT_ERROR_INVALID_BINARY:
    return PI_ERROR_INVALID_BINARY;
  case UR_RESULT_ERROR_INVALID_PROGRAM:
    return PI_ERROR_INVALID_PROGRAM;
  case UR_RESULT_ERROR_INVALID_SAMPLER:
    return PI_ERROR_INVALID_SAMPLER;
  case UR_RESULT_ERROR_INVALID_MEM_OBJECT:
    return PI_ERROR_INVALID_MEM_OBJECT;
  case UR_RESULT_ERROR_INVALID_EVENT:
    return PI_ERROR_INVALID_EVENT;
  case UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST:
    return PI_ERROR_INVALID_EVENT_WAIT_LIST;
  case UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET:
    return PI_ERROR_MISALIGNED_SUB_BUFFER_OFFSET;
  case UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE:
    return PI_ERROR_INVALID_WORK_GROUP_SIZE;
  case UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE:
    return PI_ERROR_COMPILER_NOT_AVAILABLE;
  case UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE:
    return PI_ERROR_PROFILING_INFO_NOT_AVAILABLE;
  case UR_RESULT_ERROR_DEVICE_NOT_FOUND:
    return PI_ERROR_DEVICE_NOT_FOUND;
  case UR_RESULT_ERROR_INVALID_DEVICE:
    return PI_ERROR_INVALID_DEVICE;
  case UR_RESULT_ERROR_DEVICE_REQUIRES_RESET:
  case UR_RESULT_ERROR_DEVICE_LOST:
    return PI_ERROR_DEVICE_NOT_AVAILABLE;
  case UR_RESULT_ERROR_DEVICE_PARTITION_FAILED:
    return PI_ERROR_DEVICE_PARTITION_FAILED;
  case UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT:
    return PI_ERROR_INVALID_DEVICE_PARTITION_COUNT;
  case UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE:
    return PI_ERROR_INVALID_WORK_ITEM_SIZE;
  case UR_RESULT_ERROR_INVALID_WORK_DIMENSION:
    return PI_ERROR_INVALID_WORK_DIMENSION;
  case UR_RESULT_ERROR_INVALID_KERNEL_ARGS:
    return PI_ERROR_INVALID_KERNEL_ARGS;
  case UR_RESULT_ERROR_INVALID_KERNEL:
    return PI_ERROR_INVALID_KERNEL;
  case UR_RESULT_ERROR_INVALID_KERNEL_NAME:
    return PI_ERROR_INVALID_KERNEL_NAME;
  case UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
    return PI_ERROR_INVALID_ARG_INDEX;
  case UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
    return PI_ERROR_INVALID_ARG_SIZE;
  case UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
    return PI_ERROR_INVALID_VALUE;
  case UR_RESULT_ERROR_INVALID_IMAGE_SIZE:
    return PI_ERROR_INVALID_IMAGE_SIZE;
  case UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return PI_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR;
  case UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED:
    return PI_ERROR_IMAGE_FORMAT_NOT_SUPPORTED;
  case UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE:
    return PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE;
  case UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE:
    return PI_ERROR_INVALID_PROGRAM_EXECUTABLE;
  case UR_RESULT_ERROR_UNINITIALIZED:
    return PI_ERROR_UNINITIALIZED;
  case UR_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  case UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
  case UR_RESULT_ERROR_OUT_OF_RESOURCES:
    return PI_ERROR_OUT_OF_RESOURCES;
  case UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE:
    return PI_ERROR_BUILD_PROGRAM_FAILURE;
  case UR_RESULT_ERROR_PROGRAM_LINK_FAILURE:
    return PI_ERROR_LINK_PROGRAM_FAILURE;
  case UR_RESULT_ERROR_UNSUPPORTED_VERSION:
  case UR_RESULT_ERROR_UNSUPPORTED_FEATURE:
    return PI_ERROR_INVALID_OPERATION;
  case UR_RESULT_ERROR_INVALID_ARGUMENT:
  case UR_RESULT_ERROR_INVALID_NULL_HANDLE:
  case UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
  case UR_RESULT_ERROR_INVALID_NULL_POINTER:
    return PI_ERROR_INVALID_VALUE;
  case UR_RESULT_ERROR_INVALID_SIZE:
  case UR_RESULT_ERROR_UNSUPPORTED_SIZE:
    return PI_ERROR_INVALID_BUFFER_SIZE;
  case UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
    return PI_ERROR_INVALID_VALUE;
  case UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
  case UR_RESULT_ERROR_INVALID_ENUMERATION:
  case UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
    return PI_ERROR_INVALID_VALUE;
  case UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
    return PI_ERROR_IMAGE_FORMAT_NOT_SUPPORTED;
  case UR_RESULT_ERROR_INVALID_NATIVE_BINARY:
    return PI_ERROR_INVALID_BINARY;
  case UR_RESULT_ERROR_INVALID_GLOBAL_NAME:
    return PI_ERROR_INVALID_VALUE;
  case UR_RESULT_ERROR_INVALID_FUNCTION_NAME:
    return PI_ERROR_FUNCTION_ADDRESS_IS_NOT_AVAILABLE;
  case UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
    return PI_ERROR_INVALID_WORK_DIMENSION;
  case UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
    return PI_ERROR_INVALID_VALUE;
  case UR_RESULT_ERROR_PROGRAM_UNLINKED:
    return PI_ERROR_INVALID_PROGRAM_EXECUTABLE;
  case UR_RESULT_ERROR_OVERLAPPING_REGIONS:
    return PI_ERROR_MEM_COPY_OVERLAP;
  case UR_RESULT_ERROR_INVALID_HOST_PTR:
    return PI_ERROR_INVALID_HOST_PTR;
  case UR_RESULT_ERROR_INVALID_USM_SIZE:
    return PI_ERROR_INVALID_BUFFER_SIZE;
  case UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE:
    return PI_ERROR_OUT_OF_RESOURCES;
  case UR_RESULT_ERROR_ADAPTER_SPECIFIC:
    return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
  case UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP:
    return PI_ERROR_INVALID_COMMAND_BUFFER_KHR;
  case UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP:
    return PI_ERROR_INVALID_SYNC_POINT_WAIT_LIST_KHR;
  case UR_RESULT_ERROR_UNKNOWN:
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
  // Array return value where element type is different from T
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

  // Convert the array using a conversion map
  template <typename TypeUR, typename TypePI>
  pi_result convertArray(std::function<TypePI(TypeUR)> Func) {
    // Cannot convert to a smaller element storage type
    PI_ASSERT(sizeof(TypePI) >= sizeof(TypeUR), PI_ERROR_UNKNOWN);

    const uint32_t NumberElements =
        *param_value_size_ret / sizeof(ur_device_partition_t);

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

    for (uint32_t I = 0; I < NumberElements; ++I) {
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

// Handle mismatched PI and UR type return sizes for info queries
inline void fixupInfoValueTypes(size_t ParamValueSizeRetUR,
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
}

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
        return PI_EXT_PLATFORM_BACKEND_HIP;
      case UR_PLATFORM_BACKEND_NATIVE_CPU:
        return PI_EXT_PLATFORM_BACKEND_NATIVE_CPU;
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

/**
 * Translate UR device info values to PI info values
 * @param ParamName The name of the parameter
 * @param ParamValueSize[in] The size of ParamValue passed to the PI plugin.
 * @param ParamValue[in, out] Input: The ParamValue returned by the UR adapter.
 * Output: The UR output converted to PI.
 * @param ParamValueSizeRet[in, out] Input: The value of ParamValueSizeRet that
 * UR returned. Output: The value of ParamValueSizeRet after conversion.
 */
inline pi_result ur2piDeviceInfoValue(ur_device_info_t ParamName,
                                      size_t ParamValueSize, void *ParamValue,
                                      size_t *ParamValueSizeRet) {

  /* Helper function to perform conversions in-place */
  ConvertHelper Value(ParamValueSize, ParamValue, ParamValueSizeRet);

  pi_result Error = PI_SUCCESS;
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
      case UR_QUEUE_FLAG_SYNC_WITH_DEFAULT_STREAM:
        return static_cast<uint64_t>(__SYCL_PI_CUDA_SYNC_WITH_DEFAULT);
      case UR_QUEUE_FLAG_USE_DEFAULT_STREAM:
        return static_cast<uint64_t>(__SYCL_PI_CUDA_USE_DEFAULT_STREAM);
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

    auto ConvertFunc = [](ur_device_partition_t UrValue) {
      switch (static_cast<uint32_t>(UrValue)) {
      case UR_DEVICE_PARTITION_EQUALLY:
        return PI_DEVICE_PARTITION_EQUALLY;
      case UR_DEVICE_PARTITION_BY_COUNTS:
        return PI_DEVICE_PARTITION_BY_COUNTS;
      case UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
        return PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
      case UR_DEVICE_PARTITION_BY_CSLICE:
        return PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE;
      default:
        die("UR_DEVICE_INFO_PARTITION_TYPE: unhandled value");
      }
    };

    /*
     * This property returns the argument specified in piCreateSubDevices.
     * Each partition name is immediately followed by a value. The list is
     * terminated with 0. In the case where the properties argument to
     * piCreateSubDevices is [PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
     * PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE], the affinity domain used
     * to perform the partition will be returned. */

    PI_ASSERT(sizeof(pi_device_partition_property) ==
                  sizeof(ur_device_partition_property_t),
              PI_ERROR_UNKNOWN);

    const uint32_t UrNumberElements =
        *ParamValueSizeRet / sizeof(ur_device_partition_property_t);

    if (ParamValue) {
      auto ParamValueCopy =
          std::make_unique<ur_device_partition_property_t[]>(UrNumberElements);
      std::memcpy(ParamValueCopy.get(), ParamValue,
                  UrNumberElements * sizeof(ur_device_partition_property_t));
      pi_device_partition_property *pValuePI =
          reinterpret_cast<pi_device_partition_property *>(ParamValue);
      ur_device_partition_property_t *pValueUR =
          reinterpret_cast<ur_device_partition_property_t *>(
              ParamValueCopy.get());
      const ur_device_partition_t Type = pValueUR->type;
      *pValuePI = ConvertFunc(Type);
      ++pValuePI;

      for (uint32_t i = 0; i < UrNumberElements; ++i) {
        switch (pValueUR->type) {
        case UR_DEVICE_PARTITION_EQUALLY: {
          *pValuePI = pValueUR->value.equally;
          break;
        }
        case UR_DEVICE_PARTITION_BY_COUNTS: {
          *pValuePI = pValueUR->value.count;
          break;
        }
        case UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN: {
          *pValuePI = pValueUR->value.affinity_domain;
          break;
        }
        case UR_DEVICE_PARTITION_BY_CSLICE: {
          *pValuePI = 0;
          break;
        }
        default:
          die("UR_DEVICE_INFO_PARTITION_TYPE query returned unsupported type");
        }
        ++pValuePI;
        ++pValueUR;
      }
      *pValuePI = 0;
    }

    if (ParamValueSizeRet && *ParamValueSizeRet != 0) {
      /* Add 2 extra elements to the return value (one for the type at the
       * beginning and another to terminate the array with a 0 */
      *ParamValueSizeRet =
          (UrNumberElements + 2) * sizeof(pi_device_partition_property);
    }
  }

  else if (ParamName == UR_DEVICE_INFO_SUPPORTED_PARTITIONS) {
    auto ConvertFunc = [](ur_device_partition_t UrValue) {
      switch (static_cast<uint32_t>(UrValue)) {
      case UR_DEVICE_PARTITION_EQUALLY:
        return PI_DEVICE_PARTITION_EQUALLY;
      case UR_DEVICE_PARTITION_BY_COUNTS:
        return PI_DEVICE_PARTITION_BY_COUNTS;
      case UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
        return PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
      case UR_DEVICE_PARTITION_BY_CSLICE:
        return PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE;
      default:
        die("UR_DEVICE_INFO_SUPPORTED_PARTITIONS: unhandled value");
      }
    };

    Value.convertArray<ur_device_partition_t, pi_device_partition_property>(
        ConvertFunc);

    if (ParamValue) {
      const uint32_t NumberElements =
          *ParamValueSizeRet / sizeof(pi_device_partition_property);
      reinterpret_cast<pi_device_partition_property *>(
          ParamValue)[NumberElements] = 0;
    }

    if (ParamValueSizeRet && *ParamValueSizeRet != 0) {
      *ParamValueSizeRet += sizeof(pi_device_partition_property);
    }

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
  } else if (*ParamValueSizeRet == 1 && ParamValueSize == 4) {
    /* PI type: pi_bool
     * UR type: ur_bool_t
     * Need to convert from pi_bool (4 bytes) to ur_bool_t (1 byte)
     */
    fixupInfoValueTypes(*ParamValueSizeRet, ParamValueSizeRet, ParamValueSize,
                        ParamValue);
  } else if (ParamName == UR_DEVICE_INFO_QUEUE_PROPERTIES ||
             ParamName == UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES ||
             ParamName == UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES ||
             ParamName == UR_DEVICE_INFO_EXECUTION_CAPABILITIES ||
             ParamName == UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN ||
             ParamName == UR_DEVICE_INFO_USM_HOST_SUPPORT ||
             ParamName == UR_DEVICE_INFO_USM_DEVICE_SUPPORT ||
             ParamName == UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT ||
             ParamName == UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT ||
             ParamName == UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT) {
    /* PI type: pi_bitfield
     * UR type: ur_flags_t (uint32_t)
     * No need to convert since types are compatible
     */
    *ParamValueSizeRet = sizeof(pi_bitfield);
  } else if (ParamName == UR_DEVICE_INFO_SINGLE_FP_CONFIG ||
             ParamName == UR_DEVICE_INFO_HALF_FP_CONFIG ||
             ParamName == UR_DEVICE_INFO_DOUBLE_FP_CONFIG) {
    /* CL type: pi_device_fp_config
     * UR type: ur_device_fp_capability_flags_t
     * No need to convert since types are compatible
     */
    *ParamValueSizeRet = sizeof(pi_device_fp_config);
  } else {

    // TODO: what else needs a UR-PI translation?
  }

  if (ParamValueSize && ParamValueSizeRet &&
      ParamValueSize != *ParamValueSizeRet) {
    fprintf(stderr, "UR DeviceInfoType=%d PI=%d but UR=%d\n", ParamName,
            (int)ParamValueSize, (int)*ParamValueSizeRet);
    die("ur2piDeviceInfoValue: size mismatch");
  }
  return Error;
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

// Translate UR program build info values to PI info values
inline pi_result ur2piProgramBuildInfoValue(ur_program_build_info_t ParamName,
                                            size_t ParamValueSizePI,
                                            size_t *ParamValueSizeUR,
                                            void *ParamValue) {
  ConvertHelper Value(ParamValueSizePI, ParamValue, ParamValueSizeUR);

  if (ParamName == UR_PROGRAM_BUILD_INFO_BINARY_TYPE) {
    auto ConvertFunc = [](ur_program_binary_type_t UrValue) {
      switch (UrValue) {
      case UR_PROGRAM_BINARY_TYPE_NONE:
        return PI_PROGRAM_BINARY_TYPE_NONE;
      case UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT:
        return PI_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;
      case UR_PROGRAM_BINARY_TYPE_LIBRARY:
        return PI_PROGRAM_BINARY_TYPE_LIBRARY;
      case UR_PROGRAM_BINARY_TYPE_EXECUTABLE:
        return PI_PROGRAM_BINARY_TYPE_EXECUTABLE;
      default:
        die("ur_program_binary_type_t: unhandled value");
      }
    };
    return Value.convert<ur_program_binary_type_t, pi_program_binary_type>(
        ConvertFunc);
  }

  if (ParamName == UR_PROGRAM_BUILD_INFO_STATUS) {
    auto ConvertFunc = [](ur_program_build_status_t UrValue) {
      switch (UrValue) {
      case UR_PROGRAM_BUILD_STATUS_NONE:
        return PI_PROGRAM_BUILD_STATUS_NONE;
      case UR_PROGRAM_BUILD_STATUS_ERROR:
        return PI_PROGRAM_BUILD_STATUS_ERROR;
      case UR_PROGRAM_BUILD_STATUS_SUCCESS:
        return PI_PROGRAM_BUILD_STATUS_SUCCESS;
      case UR_PROGRAM_BUILD_STATUS_IN_PROGRESS:
        return PI_PROGRAM_BUILD_STATUS_IN_PROGRESS;
      default:
        die("ur_program_build_status_t: unhandled value");
      }
    };
    return Value.convert<ur_program_build_status_t, pi_program_build_status>(
        ConvertFunc);
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
  bool *pluginTeardown = static_cast<bool *>(PluginParameter);
  *pluginTeardown = true;
  // Fetch the single known adapter (the one which is statically linked) so we
  // can release it. Fetching it for a second time (after piPlatformsGet)
  // increases the reference count, so we need to release it twice.
  // pi_unified_runtime has its own implementation of piTearDown.
  static std::once_flag AdapterReleaseFlag;
  ur_adapter_handle_t Adapter;
  ur_result_t Ret = UR_RESULT_SUCCESS;
  std::call_once(AdapterReleaseFlag, [&]() {
    Ret = urAdapterGet(1, &Adapter, nullptr);
    if (Ret == UR_RESULT_SUCCESS) {
      Ret = urAdapterRelease(Adapter);
      Ret = urAdapterRelease(Adapter);
    }
  });
  HANDLE_ERRORS(Ret);

  return PI_SUCCESS;
}

inline pi_result PiGetAdapter(ur_adapter_handle_t &adapter) {
  // We're not going through the UR loader so we're guaranteed to have exactly
  // one adapter (whichever is statically linked). The PI plugin for UR has its
  // own implementation of piPlatformsGet.
  static ur_adapter_handle_t Adapter;
  static std::once_flag AdapterGetFlag;
  ur_result_t Ret = UR_RESULT_SUCCESS;
  std::call_once(AdapterGetFlag,
                 [&Ret]() { Ret = urAdapterGet(1, &Adapter, nullptr); });
  HANDLE_ERRORS(Ret);

  adapter = Adapter;

  return PI_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// Platform
inline pi_result piPlatformsGet(pi_uint32 NumEntries, pi_platform *Platforms,
                                pi_uint32 *NumPlatforms) {
  ur_adapter_handle_t adapter = nullptr;
  if (auto res = PiGetAdapter(adapter); res != PI_SUCCESS) {
    return res;
  }

  auto phPlatforms = reinterpret_cast<ur_platform_handle_t *>(Platforms);
  HANDLE_ERRORS(
      urPlatformGet(&adapter, 1, NumEntries, phPlatforms, NumPlatforms));
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

  ur_adapter_handle_t adapter = nullptr;
  if (auto res = PiGetAdapter(adapter); res != PI_SUCCESS) {
    return res;
  }
  (void)adapter;

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

inline pi_result piPluginGetLastError(char **Message) {
  // We're not going through the UR loader so we're guaranteed to have exactly
  // one adapter (whichever is statically linked). The PI plugin for UR has its
  // own implementation of piPluginGetLastError. Materialize the adapter
  // reference for the urAdapterGetLastError call, then release it.
  ur_adapter_handle_t Adapter;
  urAdapterGet(1, &Adapter, nullptr);
  // FIXME: ErrorCode should store a native error, but these are not being used
  // in CUDA adapter at the moment
  int32_t ErrorCode;
  ur_result_t Res = urAdapterGetLastError(
      Adapter, const_cast<const char **>(Message), &ErrorCode);
  urAdapterRelease(Adapter);

  return ur2piResult(Res);
}

inline pi_result piDeviceGetInfo(pi_device Device, pi_device_info ParamName,
                                 size_t ParamValueSize, void *ParamValue,
                                 size_t *ParamValueSizeRet) {

  ur_device_info_t InfoType;
  switch (ParamName) {
#define PI_TO_UR_MAP_DEVICE_INFO(FROM, TO)                                     \
  case FROM: {                                                                 \
    InfoType = TO;                                                             \
    break;                                                                     \
  }
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_TYPE, UR_DEVICE_INFO_TYPE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PARENT_DEVICE,
                             UR_DEVICE_INFO_PARENT_DEVICE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PLATFORM, UR_DEVICE_INFO_PLATFORM)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_VENDOR_ID, UR_DEVICE_INFO_VENDOR_ID)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_UUID, UR_DEVICE_INFO_UUID)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_ATOMIC_64, UR_DEVICE_INFO_ATOMIC_64)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_EXTENSIONS,
                             UR_DEVICE_INFO_EXTENSIONS)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_NAME, UR_DEVICE_INFO_NAME)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_COMPILER_AVAILABLE,
                             UR_DEVICE_INFO_COMPILER_AVAILABLE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_LINKER_AVAILABLE,
                             UR_DEVICE_INFO_LINKER_AVAILABLE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_COMPUTE_UNITS,
                             UR_DEVICE_INFO_MAX_COMPUTE_UNITS)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
                             UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
                             UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES,
                             UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY,
                             UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_ADDRESS_BITS,
                             UR_DEVICE_INFO_ADDRESS_BITS)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE,
                             UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_GLOBAL_MEM_SIZE,
                             UR_DEVICE_INFO_GLOBAL_MEM_SIZE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_LOCAL_MEM_SIZE,
                             UR_DEVICE_INFO_LOCAL_MEM_SIZE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_IMAGE_SUPPORT,
                             UR_DEVICE_INFO_IMAGE_SUPPORTED)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_HOST_UNIFIED_MEMORY,
                             UR_DEVICE_INFO_HOST_UNIFIED_MEMORY)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_AVAILABLE, UR_DEVICE_INFO_AVAILABLE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_VENDOR, UR_DEVICE_INFO_VENDOR)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_DRIVER_VERSION,
                             UR_DEVICE_INFO_DRIVER_VERSION)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_VERSION, UR_DEVICE_INFO_VERSION)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES,
                             UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_REFERENCE_COUNT,
                             UR_DEVICE_INFO_REFERENCE_COUNT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PARTITION_PROPERTIES,
                             UR_DEVICE_INFO_SUPPORTED_PARTITIONS)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN,
                             UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PARTITION_TYPE,
                             UR_DEVICE_INFO_PARTITION_TYPE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_OPENCL_C_VERSION,
                             UR_EXT_DEVICE_INFO_OPENCL_C_VERSION)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC,
                             UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PRINTF_BUFFER_SIZE,
                             UR_DEVICE_INFO_PRINTF_BUFFER_SIZE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PROFILE, UR_DEVICE_INFO_PROFILE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_BUILT_IN_KERNELS,
                             UR_DEVICE_INFO_BUILT_IN_KERNELS)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_QUEUE_PROPERTIES,
                             UR_DEVICE_INFO_QUEUE_PROPERTIES)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_EXECUTION_CAPABILITIES,
                             UR_DEVICE_INFO_EXECUTION_CAPABILITIES)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_ENDIAN_LITTLE,
                             UR_DEVICE_INFO_ENDIAN_LITTLE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT,
                             UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION,
                             UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_LOCAL_MEM_TYPE,
                             UR_DEVICE_INFO_LOCAL_MEM_TYPE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_CONSTANT_ARGS,
                             UR_DEVICE_INFO_MAX_CONSTANT_ARGS)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE,
                             UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE,
                             UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE,
                             UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE,
                             UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_PARAMETER_SIZE,
                             UR_DEVICE_INFO_MAX_PARAMETER_SIZE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN,
                             UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_SAMPLERS,
                             UR_DEVICE_INFO_MAX_SAMPLERS)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS,
                             UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS,
                             UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_SINGLE_FP_CONFIG,
                             UR_DEVICE_INFO_SINGLE_FP_CONFIG)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_HALF_FP_CONFIG,
                             UR_DEVICE_INFO_HALF_FP_CONFIG)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_DOUBLE_FP_CONFIG,
                             UR_DEVICE_INFO_DOUBLE_FP_CONFIG)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH,
                             UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT,
                             UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH,
                             UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT,
                             UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH,
                             UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE,
                             UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR,
                             UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR,
                             UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT,
                             UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT,
                             UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT,
                             UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT,
                             UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG,
                             UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG,
                             UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT,
                             UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT,
                             UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE,
                             UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE,
                             UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF,
                             UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF,
                             UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS,
                             UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS)
    PI_TO_UR_MAP_DEVICE_INFO(
        PI_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
        UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_SUB_GROUP_SIZES_INTEL,
                             UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_IL_VERSION,
                             UR_DEVICE_INFO_IL_VERSION)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_USM_HOST_SUPPORT,
                             UR_DEVICE_INFO_USM_HOST_SUPPORT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_USM_DEVICE_SUPPORT,
                             UR_DEVICE_INFO_USM_DEVICE_SUPPORT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT,
                             UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT,
                             UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT,
                             UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_PCI_ADDRESS,
                             UR_DEVICE_INFO_PCI_ADDRESS)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_GPU_EU_COUNT,
                             UR_DEVICE_INFO_GPU_EU_COUNT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH,
                             UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE,
                             UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_ONEAPI_DEVICE_INFO_IP_VERSION,
                             UR_DEVICE_INFO_IP_VERSION)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_BUILD_ON_SUBDEVICE,
                             UR_DEVICE_INFO_BUILD_ON_SUBDEVICE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_3D,
                             UR_DEVICE_INFO_MAX_WORK_GROUPS_3D)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE,
                             UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_DEVICE_ID, UR_DEVICE_INFO_DEVICE_ID)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_INTEL_DEVICE_INFO_FREE_MEMORY,
                             UR_DEVICE_INFO_GLOBAL_MEM_FREE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_INTEL_DEVICE_INFO_MEMORY_CLOCK_RATE,
                             UR_DEVICE_INFO_MEMORY_CLOCK_RATE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_INTEL_DEVICE_INFO_MEMORY_BUS_WIDTH,
                             UR_DEVICE_INFO_MEMORY_BUS_WIDTH)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_INTEL_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES,
                             UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_GPU_SLICES,
                             UR_DEVICE_INFO_GPU_EU_SLICES)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE,
                             UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_GPU_HW_THREADS_PER_EU,
                             UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_MAX_MEM_BANDWIDTH,
                             UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_ONEAPI_DEVICE_INFO_BFLOAT16_MATH_FUNCTIONS,
                             UR_DEVICE_INFO_BFLOAT16)
    PI_TO_UR_MAP_DEVICE_INFO(
        PI_EXT_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES,
        UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES)
    PI_TO_UR_MAP_DEVICE_INFO(
        PI_EXT_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES,
        UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES,
                             UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES,
                             UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_INTEL_DEVICE_INFO_MEM_CHANNEL_SUPPORT,
                             UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_IMAGE_SRGB,
                             UR_DEVICE_INFO_IMAGE_SRGB)
    PI_TO_UR_MAP_DEVICE_INFO(PI_DEVICE_INFO_BACKEND_VERSION,
                             UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION)
    PI_TO_UR_MAP_DEVICE_INFO(
        PI_EXT_CODEPLAY_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP,
        UR_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_ONEAPI_DEVICE_INFO_BINDLESS_IMAGES_SUPPORT,
                             UR_DEVICE_INFO_BINDLESS_IMAGES_SUPPORT_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(
        PI_EXT_ONEAPI_DEVICE_INFO_BINDLESS_IMAGES_SHARED_USM_SUPPORT,
        UR_DEVICE_INFO_BINDLESS_IMAGES_SHARED_USM_SUPPORT_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(
        PI_EXT_ONEAPI_DEVICE_INFO_BINDLESS_IMAGES_1D_USM_SUPPORT,
        UR_DEVICE_INFO_BINDLESS_IMAGES_1D_USM_SUPPORT_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(
        PI_EXT_ONEAPI_DEVICE_INFO_BINDLESS_IMAGES_2D_USM_SUPPORT,
        UR_DEVICE_INFO_BINDLESS_IMAGES_2D_USM_SUPPORT_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_ONEAPI_DEVICE_INFO_IMAGE_PITCH_ALIGN,
                             UR_DEVICE_INFO_IMAGE_PITCH_ALIGN_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_ONEAPI_DEVICE_INFO_MAX_IMAGE_LINEAR_WIDTH,
                             UR_DEVICE_INFO_MAX_IMAGE_LINEAR_WIDTH_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_ONEAPI_DEVICE_INFO_MAX_IMAGE_LINEAR_HEIGHT,
                             UR_DEVICE_INFO_MAX_IMAGE_LINEAR_HEIGHT_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_ONEAPI_DEVICE_INFO_MAX_IMAGE_LINEAR_PITCH,
                             UR_DEVICE_INFO_MAX_IMAGE_LINEAR_PITCH_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_ONEAPI_DEVICE_INFO_MIPMAP_SUPPORT,
                             UR_DEVICE_INFO_MIPMAP_SUPPORT_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(
        PI_EXT_ONEAPI_DEVICE_INFO_MIPMAP_ANISOTROPY_SUPPORT,
        UR_DEVICE_INFO_MIPMAP_ANISOTROPY_SUPPORT_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_ONEAPI_DEVICE_INFO_MIPMAP_MAX_ANISOTROPY,
                             UR_DEVICE_INFO_MIPMAP_MAX_ANISOTROPY_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(
        PI_EXT_ONEAPI_DEVICE_INFO_MIPMAP_LEVEL_REFERENCE_SUPPORT,
        UR_DEVICE_INFO_MIPMAP_LEVEL_REFERENCE_SUPPORT_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(
        PI_EXT_ONEAPI_DEVICE_INFO_INTEROP_MEMORY_IMPORT_SUPPORT,
        UR_DEVICE_INFO_INTEROP_MEMORY_IMPORT_SUPPORT_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(
        PI_EXT_ONEAPI_DEVICE_INFO_INTEROP_MEMORY_EXPORT_SUPPORT,
        UR_DEVICE_INFO_INTEROP_MEMORY_EXPORT_SUPPORT_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(
        PI_EXT_ONEAPI_DEVICE_INFO_INTEROP_SEMAPHORE_IMPORT_SUPPORT,
        UR_DEVICE_INFO_INTEROP_SEMAPHORE_IMPORT_SUPPORT_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(
        PI_EXT_ONEAPI_DEVICE_INFO_INTEROP_SEMAPHORE_EXPORT_SUPPORT,
        UR_DEVICE_INFO_INTEROP_SEMAPHORE_EXPORT_SUPPORT_EXP)
    PI_TO_UR_MAP_DEVICE_INFO(PI_EXT_INTEL_DEVICE_INFO_ESIMD_SUPPORT,
                             UR_DEVICE_INFO_ESIMD_SUPPORT)
#undef PI_TO_UR_MAP_DEVICE_INFO
  default:
    return PI_ERROR_UNKNOWN;
  };

  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  size_t ParamValueSizeRetUR;
  auto DeviceUR = reinterpret_cast<ur_device_handle_t>(Device);

  HANDLE_ERRORS(urDeviceGetInfo(DeviceUR, InfoType, ParamValueSize, ParamValue,
                                &ParamValueSizeRetUR));

  ur2piDeviceInfoValue(InfoType, ParamValueSize, ParamValue,
                       &ParamValueSizeRetUR);

  if (ParamValueSizeRet) {
    *ParamValueSizeRet = ParamValueSizeRetUR;
  }

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

  if (!Properties || !Properties[0]) {
    return PI_ERROR_INVALID_VALUE;
  }

  ur_device_partition_t UrType;
  switch (Properties[0]) {
  case PI_DEVICE_PARTITION_EQUALLY:
    UrType = UR_DEVICE_PARTITION_EQUALLY;
    break;
  case PI_DEVICE_PARTITION_BY_COUNTS:
    UrType = UR_DEVICE_PARTITION_BY_COUNTS;
    break;
  case PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
    UrType = UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
    break;
  case PI_EXT_INTEL_DEVICE_PARTITION_BY_CSLICE:
    UrType = UR_DEVICE_PARTITION_BY_CSLICE;
    break;
  default:
    return PI_ERROR_UNKNOWN;
  }

  std::vector<ur_device_partition_property_t> UrProperties{};

  // UR_DEVICE_PARTITION_BY_CSLICE doesn't have a value, so
  // handle it outside the while loop below.
  if (UrType == UR_DEVICE_PARTITION_BY_CSLICE) {
    ur_device_partition_property_t UrProperty{};
    UrProperty.type = UrType;
    UrProperties.push_back(UrProperty);
  }
  while (*(++Properties)) {
    ur_device_partition_property_t UrProperty;
    UrProperty.type = UrType;
    switch (UrType) {
    case UR_DEVICE_PARTITION_EQUALLY: {
      UrProperty.value.equally = *Properties;
      break;
    }
    case UR_DEVICE_PARTITION_BY_COUNTS: {
      UrProperty.value.count = *Properties;
      break;
    }
    case UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN: {
      /* No need to convert affinity domain enums from pi to ur because they
       * are equivalent */
      UrProperty.value.affinity_domain = *Properties;
      break;
    }
    default: {
      die("Invalid properties for call to piDevicePartition");
    }
    }
    UrProperties.push_back(UrProperty);
  }

  const ur_device_partition_properties_t UrPropertiesStruct{
      UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES,
      nullptr,
      UrProperties.data(),
      UrProperties.size(),
  };

  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  auto UrSubDevices = reinterpret_cast<ur_device_handle_t *>(SubDevices);
  HANDLE_ERRORS(urDevicePartition(UrDevice, &UrPropertiesStruct, NumEntries,
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
    else if (strcmp(Binaries[BinaryCount]->DeviceTargetSpec,
                    __SYCL_PI_DEVICE_BINARY_TARGET_NATIVE_CPU) == 0)
      UrBinaries[BinaryCount].pDeviceTargetSpec =
          "native_cpu"; // todo: define UR_DEVICE_BINARY_TARGET_NATIVE_CPU;
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
  PI_ASSERT(RetContext, PI_ERROR_INVALID_VALUE);

  ur_native_handle_t NativeContext =
      reinterpret_cast<ur_native_handle_t>(NativeHandle);
  const ur_device_handle_t *UrDevices =
      reinterpret_cast<const ur_device_handle_t *>(Devices);
  ur_context_handle_t *UrContext =
      reinterpret_cast<ur_context_handle_t *>(RetContext);

  ur_context_native_properties_t Properties{
      UR_STRUCTURE_TYPE_CONTEXT_NATIVE_PROPERTIES, nullptr, OwnNativeHandle};

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
                PI_EXT_ONEAPI_QUEUE_FLAG_PRIORITY_HIGH |
                PI_EXT_QUEUE_FLAG_SUBMISSION_NO_IMMEDIATE |
                PI_EXT_QUEUE_FLAG_SUBMISSION_IMMEDIATE)),
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
  if (Properties[1] & __SYCL_PI_CUDA_SYNC_WITH_DEFAULT)
    UrProperties.flags |= UR_QUEUE_FLAG_SYNC_WITH_DEFAULT_STREAM;
  if (Properties[1] & __SYCL_PI_CUDA_USE_DEFAULT_STREAM)
    UrProperties.flags |= UR_QUEUE_FLAG_USE_DEFAULT_STREAM;
  if (Properties[1] & PI_EXT_QUEUE_FLAG_SUBMISSION_NO_IMMEDIATE)
    UrProperties.flags |= UR_QUEUE_FLAG_SUBMISSION_BATCHED;
  if (Properties[1] & PI_EXT_QUEUE_FLAG_SUBMISSION_IMMEDIATE)
    UrProperties.flags |= UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE;

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

  auto UrDevices = reinterpret_cast<ur_device_handle_t *>(
      const_cast<pi_device *>(DeviceList));

  auto urResult =
      urProgramLinkExp(UrContext, NumDevices, UrDevices, NumInputPrograms,
                       UrInputPrograms, Options, UrProgram);
  if (urResult == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    urResult = urProgramLink(UrContext, NumInputPrograms, UrInputPrograms,
                             Options, UrProgram);
  }
  return ur2piResult(urResult);
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

  auto UrDevices = reinterpret_cast<ur_device_handle_t *>(
      const_cast<pi_device *>(DeviceList));

  auto urResult =
      urProgramCompileExp(UrProgram, NumDevices, UrDevices, Options);
  if (urResult == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    urResult = urProgramCompile(UrContext, UrProgram, Options);
  }
  return ur2piResult(urResult);
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

  // These aren't supported.
  PI_ASSERT(!PFnNotify && !UserData, PI_ERROR_INVALID_VALUE);

  ur_program_handle_t UrProgram =
      reinterpret_cast<ur_program_handle_t>(Program);
  ur_program_info_t PropName = UR_PROGRAM_INFO_CONTEXT;
  ur_context_handle_t UrContext{};
  HANDLE_ERRORS(urProgramGetInfo(UrProgram, PropName, sizeof(&UrContext),
                                 &UrContext, nullptr));

  auto UrDevices = reinterpret_cast<ur_device_handle_t *>(
      const_cast<pi_device *>(DeviceList));

  auto urResult = urProgramBuildExp(UrProgram, NumDevices, UrDevices, Options);
  if (urResult == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    urResult = urProgramBuild(UrContext, UrProgram, Options);
  }
  return ur2piResult(urResult);
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
inline pi_result
piextKernelSetArgMemObj(pi_kernel Kernel, pi_uint32 ArgIndex,
                        const pi_mem_obj_property *ArgProperties,
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
  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);
  // the only applicable type, just ignore anything else
  if (ArgProperties && ArgProperties->type == PI_KERNEL_ARG_MEM_OBJ_ACCESS) {
    // following structure layout checks to be replaced with
    // std::is_layout_compatible after move to C++20
    static_assert(sizeof(pi_mem_obj_property) ==
                  sizeof(ur_kernel_arg_mem_obj_properties_t));
    static_assert(sizeof(pi_mem_obj_property::type) ==
                  sizeof(ur_kernel_arg_mem_obj_properties_t::stype));
    static_assert(sizeof(pi_mem_obj_property::pNext) ==
                  sizeof(ur_kernel_arg_mem_obj_properties_t::pNext));
    static_assert(sizeof(pi_mem_obj_property::mem_access) ==
                  sizeof(ur_kernel_arg_mem_obj_properties_t::memoryAccess));

    static_assert(uint32_t(PI_ACCESS_READ_WRITE) ==
                  uint32_t(UR_MEM_FLAG_READ_WRITE));
    static_assert(uint32_t(PI_ACCESS_READ_ONLY) ==
                  uint32_t(UR_MEM_FLAG_READ_ONLY));
    static_assert(uint32_t(PI_ACCESS_WRITE_ONLY) ==
                  uint32_t(UR_MEM_FLAG_WRITE_ONLY));
    static_assert(uint32_t(PI_KERNEL_ARG_MEM_OBJ_ACCESS) ==
                  uint32_t(UR_STRUCTURE_TYPE_KERNEL_ARG_MEM_OBJ_PROPERTIES));

    const ur_kernel_arg_mem_obj_properties_t *UrMemProperties =
        reinterpret_cast<const ur_kernel_arg_mem_obj_properties_t *>(
            ArgProperties);
    HANDLE_ERRORS(
        urKernelSetArgMemObj(UrKernel, ArgIndex, UrMemProperties, UrMemory));
  } else {
    HANDLE_ERRORS(urKernelSetArgMemObj(UrKernel, ArgIndex, nullptr, UrMemory));
  }

  return PI_SUCCESS;
}

inline pi_result piKernelSetArg(pi_kernel Kernel, pi_uint32 ArgIndex,
                                size_t ArgSize, const void *ArgValue) {

  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);

  if (ArgValue) {
    HANDLE_ERRORS(
        urKernelSetArgValue(UrKernel, ArgIndex, ArgSize, nullptr, ArgValue));
  } else {
    HANDLE_ERRORS(urKernelSetArgLocal(UrKernel, ArgIndex, ArgSize, nullptr));
  }
  return PI_SUCCESS;
}

inline pi_result piKernelSetArgPointer(pi_kernel Kernel, pi_uint32 ArgIndex,
                                       size_t ArgSize, const void *ArgValue) {
  std::ignore = ArgSize;
  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);
  HANDLE_ERRORS(urKernelSetArgPointer(UrKernel, ArgIndex, nullptr, ArgValue));

  return PI_SUCCESS;
}

inline pi_result
piextKernelCreateWithNativeHandle(pi_native_handle NativeHandle,
                                  pi_context Context, pi_program Program,
                                  bool OwnNativeHandle, pi_kernel *Kernel) {
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
    PropName = UR_KERNEL_EXEC_INFO_CACHE_CONFIG;
    auto Param = (*(static_cast<const pi_kernel_cache_config *>(ParamValue)));
    if (Param == PI_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_SLM) {
      PropValue = static_cast<uint64_t>(UR_KERNEL_CACHE_CONFIG_LARGE_SLM);
    } else if (Param == PI_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_DATA) {
      PropValue = static_cast<uint64_t>(UR_KERNEL_CACHE_CONFIG_LARGE_DATA);
      break;
    } else if (Param == PI_EXT_KERNEL_EXEC_INFO_CACHE_DEFAULT) {
      PropValue = static_cast<uint64_t>(UR_KERNEL_CACHE_CONFIG_DEFAULT);
    } else {
      die("piKernelSetExecInfo: unsupported ParamValue\n");
    }
    break;
  }
  default:
    die("piKernelSetExecInfo: unsupported ParamName\n");
  }
  HANDLE_ERRORS(urKernelSetExecInfo(UrKernel, PropName, ParamValueSize, nullptr,
                                    &PropValue));

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
    size_t NumArgs = 0;
    HANDLE_ERRORS(urKernelGetInfo(UrKernel, UR_KERNEL_INFO_NUM_ARGS,
                                  sizeof(NumArgs), &NumArgs, nullptr));
    if (ParamValueSizeRet) {
      *ParamValueSizeRet = sizeof(uint32_t);
    }
    if (ParamValue) {
      if (ParamValueSize != sizeof(uint32_t))
        return PI_ERROR_INVALID_BUFFER_SIZE;
      *static_cast<uint32_t *>(ParamValue) = static_cast<uint32_t>(NumArgs);
    }
    return PI_SUCCESS;
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
                                          size_t, const void *ArgValue) {
  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);
  HANDLE_ERRORS(urKernelSetArgPointer(UrKernel, ArgIndex, nullptr, ArgValue));

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

  size_t SizeInOut = ParamValueSize;
  HANDLE_ERRORS(urProgramGetBuildInfo(UrProgram, UrDevice, PropName,
                                      ParamValueSize, ParamValue,
                                      ParamValueSizeRet));
  ur2piProgramBuildInfoValue(PropName, ParamValueSize, &SizeInOut, ParamValue);
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
  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);
  HANDLE_ERRORS(urEnqueueDeviceGlobalVariableWrite(
      UrQueue, UrProgram, Name, BlockingWrite, Count, Offset, Src,
      NumEventsInWaitList, UrEventsWaitList, UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueDeviceGlobalVariableRead(
      UrQueue, UrProgram, Name, BlockingRead, Count, Offset, Dst,
      NumEventsInWaitList, UrEventsWaitList, UREvent));

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
  ur_usm_desc_t USMDesc{};
  USMDesc.align = Alignment;

  ur_usm_alloc_location_desc_t UsmLocationDesc{};
  UsmLocationDesc.stype = UR_STRUCTURE_TYPE_USM_ALLOC_LOCATION_DESC;

  if (Properties) {
    uint32_t Next = 0;
    while (Properties[Next]) {
      if (Properties[Next] == PI_MEM_USM_ALLOC_BUFFER_LOCATION) {
        UsmLocationDesc.location = static_cast<uint32_t>(Properties[Next + 1]);
        USMDesc.pNext = &UsmLocationDesc;
      } else {
        return PI_ERROR_INVALID_VALUE;
      }
      Next += 2;
    }
  }

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);

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
#define PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(FROM, TO)                              \
  case FROM: {                                                                 \
    UrFormat->channelType = TO;                                                \
    break;                                                                     \
  }
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_SNORM_INT8,
                                    UR_IMAGE_CHANNEL_TYPE_SNORM_INT8)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_SNORM_INT16,
                                    UR_IMAGE_CHANNEL_TYPE_SNORM_INT16)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_UNORM_INT8,
                                    UR_IMAGE_CHANNEL_TYPE_UNORM_INT8)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_UNORM_INT16,
                                    UR_IMAGE_CHANNEL_TYPE_UNORM_INT16)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565,
                                    UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555,
                                    UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_UNORM_INT_101010,
                                    UR_IMAGE_CHANNEL_TYPE_INT_101010)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_SIGNED_INT8,
                                    UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_SIGNED_INT16,
                                    UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_SIGNED_INT32,
                                    UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8,
                                    UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16,
                                    UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32,
                                    UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_HALF_FLOAT,
                                    UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT)
    PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE(PI_IMAGE_CHANNEL_TYPE_FLOAT,
                                    UR_IMAGE_CHANNEL_TYPE_FLOAT)
#undef PI_TO_UR_MAP_IMAGE_CHANNEL_TYPE
  default: {
    die("piMemImageCreate: unsuppported image_channel_data_type.");
  }
  }
  switch (ImageFormat->image_channel_order) {
#define PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(FROM, TO)                             \
  case FROM: {                                                                 \
    UrFormat->channelOrder = TO;                                               \
    break;                                                                     \
  }
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_A,
                                     UR_IMAGE_CHANNEL_ORDER_A)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_R,
                                     UR_IMAGE_CHANNEL_ORDER_R)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_RG,
                                     UR_IMAGE_CHANNEL_ORDER_RG)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_RA,
                                     UR_IMAGE_CHANNEL_ORDER_RA)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_RGB,
                                     UR_IMAGE_CHANNEL_ORDER_RGB)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_RGBA,
                                     UR_IMAGE_CHANNEL_ORDER_RGBA)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_BGRA,
                                     UR_IMAGE_CHANNEL_ORDER_BGRA)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_ARGB,
                                     UR_IMAGE_CHANNEL_ORDER_ARGB)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_ABGR,
                                     UR_IMAGE_CHANNEL_ORDER_ABGR)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_INTENSITY,
                                     UR_IMAGE_CHANNEL_ORDER_INTENSITY)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_LUMINANCE,
                                     UR_IMAGE_CHANNEL_ORDER_LUMINANCE)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_Rx,
                                     UR_IMAGE_CHANNEL_ORDER_RX)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_RGx,
                                     UR_IMAGE_CHANNEL_ORDER_RGX)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_RGBx,
                                     UR_IMAGE_CHANNEL_ORDER_RGBX)
    PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER(PI_IMAGE_CHANNEL_ORDER_sRGBA,
                                     UR_IMAGE_CHANNEL_ORDER_SRGBA)
#undef PI_TO_UR_MAP_IMAGE_CHANNEL_ORDER
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
#define PI_TO_UR_MAP_IMAGE_TYPE(FROM, TO)                                      \
  case FROM: {                                                                 \
    UrDesc->type = TO;                                                         \
    break;                                                                     \
  }
    PI_TO_UR_MAP_IMAGE_TYPE(PI_MEM_TYPE_BUFFER, UR_MEM_TYPE_BUFFER)
    PI_TO_UR_MAP_IMAGE_TYPE(PI_MEM_TYPE_IMAGE2D, UR_MEM_TYPE_IMAGE2D)
    PI_TO_UR_MAP_IMAGE_TYPE(PI_MEM_TYPE_IMAGE3D, UR_MEM_TYPE_IMAGE3D)
    PI_TO_UR_MAP_IMAGE_TYPE(PI_MEM_TYPE_IMAGE2D_ARRAY,
                            UR_MEM_TYPE_IMAGE2D_ARRAY)
    PI_TO_UR_MAP_IMAGE_TYPE(PI_MEM_TYPE_IMAGE1D, UR_MEM_TYPE_IMAGE1D)
    PI_TO_UR_MAP_IMAGE_TYPE(PI_MEM_TYPE_IMAGE1D_ARRAY,
                            UR_MEM_TYPE_IMAGE1D_ARRAY)
    PI_TO_UR_MAP_IMAGE_TYPE(PI_MEM_TYPE_IMAGE1D_BUFFER,
                            UR_MEM_TYPE_IMAGE1D_BUFFER)
#undef PI_TO_UR_MAP_IMAGE_TYPE
  default: {
    die("piMemImageCreate: unsuppported image_type.");
  }
  }
  UrDesc->width = ImageDesc->image_width;
  UrDesc->arraySize = ImageDesc->image_array_size;
  UrDesc->arraySize = ImageDesc->image_array_size;
}

static void ur2piImageFormat(const ur_image_format_t *UrFormat,
                             pi_image_format *PiFormat) {
  switch (UrFormat->channelOrder) {
#define UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(FROM, TO)                             \
  case FROM: {                                                                 \
    PiFormat->image_channel_order = TO;                                        \
    break;                                                                     \
  }
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_A,
                                     PI_IMAGE_CHANNEL_ORDER_A)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_R,
                                     PI_IMAGE_CHANNEL_ORDER_R)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_RG,
                                     PI_IMAGE_CHANNEL_ORDER_RG)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_RA,
                                     PI_IMAGE_CHANNEL_ORDER_RA)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_RGB,
                                     PI_IMAGE_CHANNEL_ORDER_RGB)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_RGBA,
                                     PI_IMAGE_CHANNEL_ORDER_RGBA)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_BGRA,
                                     PI_IMAGE_CHANNEL_ORDER_BGRA)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_ARGB,
                                     PI_IMAGE_CHANNEL_ORDER_ARGB)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_ABGR,
                                     PI_IMAGE_CHANNEL_ORDER_ABGR)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_INTENSITY,
                                     PI_IMAGE_CHANNEL_ORDER_INTENSITY)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_LUMINANCE,
                                     PI_IMAGE_CHANNEL_ORDER_LUMINANCE)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_RX,
                                     PI_IMAGE_CHANNEL_ORDER_Rx)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_RGX,
                                     PI_IMAGE_CHANNEL_ORDER_RGx)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_RGBX,
                                     PI_IMAGE_CHANNEL_ORDER_RGBx)
    UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER(UR_IMAGE_CHANNEL_ORDER_SRGBA,
                                     PI_IMAGE_CHANNEL_ORDER_sRGBA)
#undef UR_TO_PI_MAP_IMAGE_CHANNEL_ORDER
  default: {
    die("ur2piImageFormat: unsuppported channelOrder.");
  }
  }

  switch (UrFormat->channelType) {
#define UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(FROM, TO)                              \
  case FROM: {                                                                 \
    PiFormat->image_channel_data_type = TO;                                    \
    break;                                                                     \
  }
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_SNORM_INT8,
                                    PI_IMAGE_CHANNEL_TYPE_SNORM_INT8)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_SNORM_INT16,
                                    PI_IMAGE_CHANNEL_TYPE_SNORM_INT16)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_UNORM_INT8,
                                    PI_IMAGE_CHANNEL_TYPE_UNORM_INT8)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_UNORM_INT16,
                                    PI_IMAGE_CHANNEL_TYPE_UNORM_INT16)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565,
                                    PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555,
                                    PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_INT_101010,
                                    PI_IMAGE_CHANNEL_TYPE_UNORM_INT_101010)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8,
                                    PI_IMAGE_CHANNEL_TYPE_SIGNED_INT8)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16,
                                    PI_IMAGE_CHANNEL_TYPE_SIGNED_INT16)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32,
                                    PI_IMAGE_CHANNEL_TYPE_SIGNED_INT32)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8,
                                    PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16,
                                    PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32,
                                    PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT,
                                    PI_IMAGE_CHANNEL_TYPE_HALF_FLOAT)
    UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE(UR_IMAGE_CHANNEL_TYPE_FLOAT,
                                    PI_IMAGE_CHANNEL_TYPE_FLOAT)
#undef UR_TO_PI_MAP_IMAGE_CHANNEL_TYPE
  default: {
    die("ur2piImageFormat: unsuppported channelType.");
  }
  }
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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemImageCopy(
      UrQueue, UrImageSrc, UrImageDst, UrSrcOrigin, UrDstOrigin, UrRegion,
      NumEventsInWaitList, UrEventsWaitList, UREvent));

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
  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);

  ur_usm_desc_t USMDesc{};
  USMDesc.align = Alignment;

  ur_usm_alloc_location_desc_t UsmLocDesc{};
  UsmLocDesc.stype = UR_STRUCTURE_TYPE_USM_ALLOC_LOCATION_DESC;

  if (Properties) {
    uint32_t Next = 0;
    while (Properties[Next]) {
      if (Properties[Next] == PI_MEM_USM_ALLOC_BUFFER_LOCATION) {
        UsmLocDesc.location = static_cast<uint32_t>(Properties[Next + 1]);
        USMDesc.pNext = &UsmLocDesc;
      } else {
        return PI_ERROR_INVALID_VALUE;
      }
      Next += 2;
    }
  }

  ur_usm_pool_handle_t Pool{};
  HANDLE_ERRORS(
      urUSMDeviceAlloc(UrContext, UrDevice, &USMDesc, Pool, Size, ResultPtr));

  return PI_SUCCESS;
}

inline pi_result piextUSMPitchedAlloc(void **ResultPtr, size_t *ResultPitch,
                                      pi_context Context, pi_device Device,
                                      pi_usm_mem_properties *Properties,
                                      size_t WidthInBytes, size_t Height,
                                      unsigned int ElementSizeBytes) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  std::ignore = Properties;
  ur_usm_desc_t USMDesc{};
  ur_usm_pool_handle_t Pool{};

  HANDLE_ERRORS(urUSMPitchedAllocExp(UrContext, UrDevice, &USMDesc, Pool,
                                     WidthInBytes, Height, ElementSizeBytes,
                                     ResultPtr, ResultPitch));

  return PI_SUCCESS;
}

inline pi_result piextUSMSharedAlloc(void **ResultPtr, pi_context Context,
                                     pi_device Device,
                                     pi_usm_mem_properties *Properties,
                                     size_t Size, pi_uint32 Alignment) {
  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);

  ur_usm_desc_t USMDesc{};
  USMDesc.align = Alignment;
  ur_usm_device_desc_t UsmDeviceDesc{};
  UsmDeviceDesc.stype = UR_STRUCTURE_TYPE_USM_DEVICE_DESC;
  ur_usm_host_desc_t UsmHostDesc{};
  UsmHostDesc.stype = UR_STRUCTURE_TYPE_USM_HOST_DESC;
  ur_usm_alloc_location_desc_t UsmLocationDesc{};
  UsmLocationDesc.stype = UR_STRUCTURE_TYPE_USM_ALLOC_LOCATION_DESC;

  // One properties bitfield can correspond to a host_desc and a device_desc
  // struct, since having `0` values in these is harmless we can set up this
  // pNext chain in advance.
  USMDesc.pNext = &UsmDeviceDesc;
  UsmDeviceDesc.pNext = &UsmHostDesc;

  if (Properties) {
    uint32_t Next = 0;
    while (Properties[Next]) {
      switch (Properties[Next]) {
      case PI_MEM_ALLOC_FLAGS: {
        if (Properties[Next + 1] & PI_MEM_ALLOC_WRTITE_COMBINED) {
          UsmDeviceDesc.flags |= UR_USM_DEVICE_MEM_FLAG_WRITE_COMBINED;
        }
        if (Properties[Next + 1] & PI_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE) {
          UsmDeviceDesc.flags |= UR_USM_DEVICE_MEM_FLAG_INITIAL_PLACEMENT;
        }
        if (Properties[Next + 1] & PI_MEM_ALLOC_INITIAL_PLACEMENT_HOST) {
          UsmHostDesc.flags |= UR_USM_HOST_MEM_FLAG_INITIAL_PLACEMENT;
        }
        if (Properties[Next + 1] & PI_MEM_ALLOC_DEVICE_READ_ONLY) {
          UsmDeviceDesc.flags |= UR_USM_DEVICE_MEM_FLAG_DEVICE_READ_ONLY;
        }
        break;
      }
      case PI_MEM_USM_ALLOC_BUFFER_LOCATION: {
        UsmLocationDesc.location = static_cast<uint32_t>(Properties[Next + 1]);
        // We wait until we've seen a BUFFER_LOCATION property to tack this
        // onto the end of the chain, a `0` here might be valid as far as we
        // know so we must exclude it unless we've been given a value.
        UsmHostDesc.pNext = &UsmLocationDesc;
        break;
      }
      default:
        return PI_ERROR_INVALID_VALUE;
      }
      Next += 2;
    }
  }

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  // TODO: to map from pi_usm_migration_flags to
  // ur_usm_migration_flags_t
  // once we have those defined
  ur_usm_migration_flags_t UrFlags{};
  HANDLE_ERRORS(urEnqueueUSMPrefetch(UrQueue, Ptr, Size, UrFlags,
                                     NumEventsInWaitList, UrEventsWaitList,
                                     UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

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
  if (Advice & PI_MEM_ADVICE_CUDA_SET_ACCESSED_BY) {
    UrAdvice |= UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE;
  }
  if (Advice & PI_MEM_ADVICE_CUDA_UNSET_ACCESSED_BY) {
    UrAdvice |= UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_DEVICE;
  }
  if (Advice & PI_MEM_ADVICE_CUDA_SET_ACCESSED_BY_HOST) {
    UrAdvice |= UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_HOST;
  }
  if (Advice & PI_MEM_ADVICE_CUDA_UNSET_ACCESSED_BY_HOST) {
    UrAdvice |= UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_HOST;
  }
  if (Advice & PI_MEM_ADVICE_RESET) {
    UrAdvice |= UR_USM_ADVICE_FLAG_DEFAULT;
  }

  HANDLE_ERRORS(urEnqueueUSMAdvise(UrQueue, Ptr, Length, UrAdvice, UREvent));

  return PI_SUCCESS;
}

/// USM 2D Fill API
///
/// \param queue is the queue to submit to
/// \param ptr is the ptr to fill
/// \param pitch is the total width of the destination memory including
/// padding \param pattern is a pointer with the bytes of the pattern to set
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

inline pi_result piextUSMImport(const void *HostPtr, size_t Size,
                                pi_context Context) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);

  HANDLE_ERRORS(urUSMImportExp(UrContext, const_cast<void *>(HostPtr), Size));
  return PI_SUCCESS;
}

inline pi_result piextUSMRelease(const void *HostPtr, pi_context Context) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);

  HANDLE_ERRORS(urUSMReleaseExp(UrContext, const_cast<void *>(HostPtr)));
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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueUSMMemcpy2D(
      UrQueue, Blocking, DstPtr, DstPitch, SrcPtr, SrcPitch, Width, Height,
      NumEventsInWaitList, UrEventsWaitList, UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueKernelLaunch(
      UrQueue, UrKernel, WorkDim, GlobalWorkOffset, GlobalWorkSize,
      LocalWorkSize, NumEventsInWaitList, UrEventsWaitList, UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemImageWrite(
      UrQueue, UrImage, BlockingWrite, UrOrigin, UrRegion, InputRowPitch,
      InputSlicePitch, const_cast<void *>(Ptr), NumEventsInWaitList,
      UrEventsWaitList, UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemImageRead(
      UrQueue, UrImage, BlockingRead, UrOrigin, UrRegion, RowPitch, SlicePitch,
      Ptr, NumEventsInWaitList, UrEventsWaitList, UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferMap(UrQueue, UrMem, BlockingMap, UrMapFlags,
                                      Offset, Size, NumEventsInWaitList,
                                      UrEventsWaitList, UREvent, RetMap));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemUnmap(UrQueue, UrMem, MappedPtr,
                                  NumEventsInWaitList, UrEventsWaitList,
                                  UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferFill(UrQueue, UrBuffer, Pattern, PatternSize,
                                       Offset, Size, NumEventsInWaitList,
                                       UrEventsWaitList, UREvent));
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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  size_t PatternSize = 1;
  HANDLE_ERRORS(urEnqueueUSMFill(UrQueue, Ptr, PatternSize, &Value, Count,
                                 NumEventsInWaitList, UrEventsWaitList,
                                 UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferCopyRect(
      UrQueue, UrBufferSrc, UrBufferDst, UrSrcOrigin, UrDstOrigin, UrRegion,
      SrcRowPitch, SrcSlicePitch, DstRowPitch, DstSlicePitch,
      NumEventsInWaitList, UrEventsWaitList, UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferCopy(
      UrQueue, UrBufferSrc, UrBufferDst, SrcOffset, DstOffset, Size,
      NumEventsInWaitList, UrEventsWaitList, UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueUSMMemcpy(UrQueue, Blocking, DstPtr, SrcPtr, Size,
                                   NumEventsInWaitList, UrEventsWaitList,
                                   UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferWriteRect(
      UrQueue, UrBuffer, BlockingWrite, UrBufferOffset, UrHostOffset, UrRegion,
      BufferRowPitch, BufferSlicePitch, HostRowPitch, HostSlicePitch,
      const_cast<void *>(Ptr), NumEventsInWaitList, UrEventsWaitList, UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferWrite(
      UrQueue, UrBuffer, BlockingWrite, Offset, Size, const_cast<void *>(Ptr),
      NumEventsInWaitList, UrEventsWaitList, UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferReadRect(
      UrQueue, UrBuffer, BlockingRead, UrBufferOffset, UrHostOffset, UrRegion,
      BufferRowPitch, BufferSlicePitch, HostRowPitch, HostSlicePitch, Ptr,
      NumEventsInWaitList, UrEventsWaitList, UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueMemBufferRead(UrQueue, UrBuffer, BlockingRead, Offset,
                                       Size, Dst, NumEventsInWaitList,
                                       UrEventsWaitList, UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueEventsWaitWithBarrier(UrQueue, NumEventsInWaitList,
                                               UrEventsWaitList, UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(OutEvent);

  HANDLE_ERRORS(urEnqueueEventsWait(UrQueue, NumEventsInWaitList,
                                    UrEventsWaitList, UREvent));

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

  ur_event_handle_t UREvent = reinterpret_cast<ur_event_handle_t>(Event);

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

  HANDLE_ERRORS(urEventGetInfo(UREvent, PropName, ParamValueSize, ParamValue,
                               ParamValueSizeRet));

  return PI_SUCCESS;
}

inline pi_result piextEventGetNativeHandle(pi_event Event,
                                           pi_native_handle *NativeHandle) {

  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  ur_event_handle_t UREvent = reinterpret_cast<ur_event_handle_t>(Event);

  ur_native_handle_t *UrNativeEvent =
      reinterpret_cast<ur_native_handle_t *>(NativeHandle);
  HANDLE_ERRORS(urEventGetNativeHandle(UREvent, UrNativeEvent));

  return PI_SUCCESS;
}

inline pi_result piEventGetProfilingInfo(pi_event Event,
                                         pi_profiling_info ParamName,
                                         size_t ParamValueSize,
                                         void *ParamValue,
                                         size_t *ParamValueSizeRet) {

  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);

  ur_event_handle_t UREvent = reinterpret_cast<ur_event_handle_t>(Event);

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

  HANDLE_ERRORS(urEventGetProfilingInfo(UREvent, PropName, ParamValueSize,
                                        ParamValue, ParamValueSizeRet));

  return PI_SUCCESS;
}

inline pi_result piEventCreate(pi_context Context, pi_event *RetEvent) {

  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(RetEvent);
  // pass null for the hNativeHandle to use urEventCreateWithNativeHandle
  // as urEventCreate
  ur_event_native_properties_t Properties{};
  HANDLE_ERRORS(
      urEventCreateWithNativeHandle(nullptr, UrContext, &Properties, UREvent));

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

  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(Event);
  ur_event_native_properties_t Properties{};
  Properties.isNativeHandleOwned = OwnNativeHandle;
  HANDLE_ERRORS(urEventCreateWithNativeHandle(UrNativeKernel, UrContext,
                                              &Properties, UREvent));

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

  ur_event_handle_t UREvent = reinterpret_cast<ur_event_handle_t>(Event);
  HANDLE_ERRORS(urEventRetain(UREvent));

  return PI_SUCCESS;
}

inline pi_result piEventRelease(pi_event Event) {
  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);

  ur_event_handle_t UREvent = reinterpret_cast<ur_event_handle_t>(Event);
  HANDLE_ERRORS(urEventRelease(UREvent));

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

  HANDLE_ERRORS(urKernelSetArgSampler(UrKernel, ArgIndex, nullptr, UrSampler));

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

///////////////////////////////////////////////////////////////////////////////
// Command-buffer extension

inline pi_result
piextCommandBufferCreate(pi_context Context, pi_device Device,
                         const pi_ext_command_buffer_desc *Desc,
                         pi_ext_command_buffer *RetCommandBuffer) {
  ur_context_handle_t UrContext =
      reinterpret_cast<ur_context_handle_t>(Context);
  ur_device_handle_t UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  const ur_exp_command_buffer_desc_t *UrDesc =
      reinterpret_cast<const ur_exp_command_buffer_desc_t *>(Desc);
  ur_exp_command_buffer_handle_t *UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t *>(RetCommandBuffer);

  HANDLE_ERRORS(
      urCommandBufferCreateExp(UrContext, UrDevice, UrDesc, UrCommandBuffer));

  return PI_SUCCESS;
}

inline pi_result piextCommandBufferRetain(pi_ext_command_buffer CommandBuffer) {
  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);

  HANDLE_ERRORS(urCommandBufferRetainExp(UrCommandBuffer));

  return PI_SUCCESS;
}

inline pi_result
piextCommandBufferRelease(pi_ext_command_buffer CommandBuffer) {
  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);

  HANDLE_ERRORS(urCommandBufferReleaseExp(UrCommandBuffer));

  return PI_SUCCESS;
}

inline pi_result
piextCommandBufferFinalize(pi_ext_command_buffer CommandBuffer) {
  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);

  HANDLE_ERRORS(urCommandBufferFinalizeExp(UrCommandBuffer));

  return PI_SUCCESS;
}

inline pi_result piextCommandBufferNDRangeKernel(
    pi_ext_command_buffer CommandBuffer, pi_kernel Kernel, pi_uint32 WorkDim,
    const size_t *GlobalWorkOffset, const size_t *GlobalWorkSize,
    const size_t *LocalWorkSize, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);

  ur_kernel_handle_t UrKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);

  HANDLE_ERRORS(urCommandBufferAppendKernelLaunchExp(
      UrCommandBuffer, UrKernel, WorkDim, GlobalWorkOffset, GlobalWorkSize,
      LocalWorkSize, NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint));

  return PI_SUCCESS;
}

inline pi_result piextCommandBufferMemcpyUSM(
    pi_ext_command_buffer CommandBuffer, void *DstPtr, const void *SrcPtr,
    size_t Size, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);

  HANDLE_ERRORS(urCommandBufferAppendUSMMemcpyExp(
      UrCommandBuffer, DstPtr, SrcPtr, Size, NumSyncPointsInWaitList,
      SyncPointWaitList, SyncPoint));

  return PI_SUCCESS;
}

inline pi_result piextCommandBufferMemBufferCopy(
    pi_ext_command_buffer CommandBuffer, pi_mem SrcMem, pi_mem DstMem,
    size_t SrcOffset, size_t DstOffset, size_t Size,
    pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);

  ur_mem_handle_t UrSrcMem = reinterpret_cast<ur_mem_handle_t>(SrcMem);
  ur_mem_handle_t UrDstMem = reinterpret_cast<ur_mem_handle_t>(DstMem);

  HANDLE_ERRORS(urCommandBufferAppendMemBufferCopyExp(
      UrCommandBuffer, UrSrcMem, UrDstMem, SrcOffset, DstOffset, Size,
      NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint));

  return PI_SUCCESS;
}

inline pi_result piextCommandBufferMemBufferCopyRect(
    pi_ext_command_buffer CommandBuffer, pi_mem SrcMem, pi_mem DstMem,
    pi_buff_rect_offset SrcOrigin, pi_buff_rect_offset DstOrigin,
    pi_buff_rect_region Region, size_t SrcRowPitch, size_t SrcSlicePitch,
    size_t DstRowPitch, size_t DstSlicePitch, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);

  ur_mem_handle_t UrSrcMem = reinterpret_cast<ur_mem_handle_t>(SrcMem);
  ur_mem_handle_t UrDstMem = reinterpret_cast<ur_mem_handle_t>(DstMem);

  ur_rect_offset_t UrSrcOrigin{SrcOrigin->x_bytes, SrcOrigin->y_scalar,
                               SrcOrigin->z_scalar};
  ur_rect_offset_t UrDstOrigin{DstOrigin->x_bytes, DstOrigin->y_scalar,
                               DstOrigin->z_scalar};
  ur_rect_region_t UrRegion{};
  UrRegion.depth = Region->depth_scalar;
  UrRegion.height = Region->height_scalar;
  UrRegion.width = Region->width_bytes;

  HANDLE_ERRORS(urCommandBufferAppendMemBufferCopyRectExp(
      UrCommandBuffer, UrSrcMem, UrDstMem, UrSrcOrigin, UrDstOrigin, UrRegion,
      SrcRowPitch, SrcSlicePitch, DstRowPitch, DstSlicePitch,
      NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint));

  return PI_SUCCESS;
}

inline pi_result piextCommandBufferMemBufferReadRect(
    pi_ext_command_buffer CommandBuffer, pi_mem Buffer,
    pi_buff_rect_offset BufferOffset, pi_buff_rect_offset HostOffset,
    pi_buff_rect_region Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, void *Ptr,
    pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {

  PI_ASSERT(Buffer, PI_ERROR_INVALID_MEM_OBJECT);

  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);
  ur_mem_handle_t UrBuffer = reinterpret_cast<ur_mem_handle_t>(Buffer);
  ur_rect_offset_t UrBufferOffset{BufferOffset->x_bytes, BufferOffset->y_scalar,
                                  BufferOffset->z_scalar};
  ur_rect_offset_t UrHostOffset{HostOffset->x_bytes, HostOffset->y_scalar,
                                HostOffset->z_scalar};
  ur_rect_region_t UrRegion{};
  UrRegion.depth = Region->depth_scalar;
  UrRegion.height = Region->height_scalar;
  UrRegion.width = Region->width_bytes;

  HANDLE_ERRORS(urCommandBufferAppendMemBufferReadRectExp(
      UrCommandBuffer, UrBuffer, UrBufferOffset, UrHostOffset, UrRegion,
      BufferRowPitch, BufferSlicePitch, HostRowPitch, HostSlicePitch, Ptr,
      NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint));

  return PI_SUCCESS;
}

inline pi_result piextCommandBufferMemBufferRead(
    pi_ext_command_buffer CommandBuffer, pi_mem Src, size_t Offset, size_t Size,
    void *Dst, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  PI_ASSERT(Src, PI_ERROR_INVALID_MEM_OBJECT);

  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);
  ur_mem_handle_t UrBuffer = reinterpret_cast<ur_mem_handle_t>(Src);

  HANDLE_ERRORS(urCommandBufferAppendMemBufferReadExp(
      UrCommandBuffer, UrBuffer, Offset, Size, Dst, NumSyncPointsInWaitList,
      SyncPointWaitList, SyncPoint));

  return PI_SUCCESS;
}

inline pi_result piextCommandBufferMemBufferWriteRect(
    pi_ext_command_buffer CommandBuffer, pi_mem Buffer,
    pi_buff_rect_offset BufferOffset, pi_buff_rect_offset HostOffset,
    pi_buff_rect_region Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, const void *Ptr,
    pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {

  PI_ASSERT(Buffer, PI_ERROR_INVALID_MEM_OBJECT);

  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);
  ur_mem_handle_t UrBuffer = reinterpret_cast<ur_mem_handle_t>(Buffer);
  ur_rect_offset_t UrBufferOffset{BufferOffset->x_bytes, BufferOffset->y_scalar,
                                  BufferOffset->z_scalar};
  ur_rect_offset_t UrHostOffset{HostOffset->x_bytes, HostOffset->y_scalar,
                                HostOffset->z_scalar};
  ur_rect_region_t UrRegion{};
  UrRegion.depth = Region->depth_scalar;
  UrRegion.height = Region->height_scalar;
  UrRegion.width = Region->width_bytes;

  HANDLE_ERRORS(urCommandBufferAppendMemBufferWriteRectExp(
      UrCommandBuffer, UrBuffer, UrBufferOffset, UrHostOffset, UrRegion,
      BufferRowPitch, BufferSlicePitch, HostRowPitch, HostSlicePitch,
      const_cast<void *>(Ptr), NumSyncPointsInWaitList, SyncPointWaitList,
      SyncPoint));

  return PI_SUCCESS;
}

inline pi_result piextCommandBufferMemBufferWrite(
    pi_ext_command_buffer CommandBuffer, pi_mem Buffer, size_t Offset,
    size_t Size, const void *Ptr, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {

  PI_ASSERT(Buffer, PI_ERROR_INVALID_MEM_OBJECT);

  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);
  ur_mem_handle_t UrBuffer = reinterpret_cast<ur_mem_handle_t>(Buffer);

  HANDLE_ERRORS(urCommandBufferAppendMemBufferWriteExp(
      UrCommandBuffer, UrBuffer, Offset, Size, const_cast<void *>(Ptr),
      NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint));

  return PI_SUCCESS;
}

inline pi_result piextCommandBufferMemBufferFill(
    pi_ext_command_buffer CommandBuffer, pi_mem Buffer, const void *Pattern,
    size_t PatternSize, size_t Offset, size_t Size,
    pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  PI_ASSERT(Buffer, PI_ERROR_INVALID_MEM_OBJECT);

  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);
  ur_mem_handle_t UrBuffer = reinterpret_cast<ur_mem_handle_t>(Buffer);

  HANDLE_ERRORS(urCommandBufferAppendMemBufferFillExp(
      UrCommandBuffer, UrBuffer, Pattern, PatternSize, Offset, Size,
      NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint));
  return PI_SUCCESS;
}

inline pi_result piextCommandBufferFillUSM(
    pi_ext_command_buffer CommandBuffer, void *Ptr, const void *Pattern,
    size_t PatternSize, size_t Size, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {

  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);

  HANDLE_ERRORS(urCommandBufferAppendUSMFillExp(
      UrCommandBuffer, Ptr, Pattern, PatternSize, Size, NumSyncPointsInWaitList,
      SyncPointWaitList, SyncPoint));
  return PI_SUCCESS;
}

inline pi_result piextCommandBufferPrefetchUSM(
    pi_ext_command_buffer CommandBuffer, const void *Ptr, size_t Size,
    pi_usm_migration_flags Flags, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {

  // flags is currently unused so fail if set
  PI_ASSERT(Flags == 0, PI_ERROR_INVALID_VALUE);

  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);

  // TODO: to map from pi_usm_migration_flags to
  // ur_usm_migration_flags_t
  // once we have those defined
  ur_usm_migration_flags_t UrFlags{};
  HANDLE_ERRORS(urCommandBufferAppendUSMPrefetchExp(
      UrCommandBuffer, Ptr, Size, UrFlags, NumSyncPointsInWaitList,
      SyncPointWaitList, SyncPoint));
  return PI_SUCCESS;
}

inline pi_result piextCommandBufferAdviseUSM(
    pi_ext_command_buffer CommandBuffer, const void *Ptr, size_t Length,
    pi_mem_advice Advice, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  // TODO: Handle advice correctly
  (void)Advice;

  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);

  ur_usm_advice_flags_t UrAdvice{};
  HANDLE_ERRORS(urCommandBufferAppendUSMAdviseExp(
      UrCommandBuffer, Ptr, Length, UrAdvice, NumSyncPointsInWaitList,
      SyncPointWaitList, SyncPoint));
  return PI_SUCCESS;
}

inline pi_result piextEnqueueCommandBuffer(pi_ext_command_buffer CommandBuffer,
                                           pi_queue Queue,
                                           pi_uint32 NumEventsInWaitList,
                                           const pi_event *EventWaitList,
                                           pi_event *Event) {

  ur_exp_command_buffer_handle_t UrCommandBuffer =
      reinterpret_cast<ur_exp_command_buffer_handle_t>(CommandBuffer);

  ur_queue_handle_t UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  const ur_event_handle_t *UrEventWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventWaitList);
  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(Event);

  HANDLE_ERRORS(urCommandBufferEnqueueExp(
      UrCommandBuffer, UrQueue, NumEventsInWaitList, UrEventWaitList, UREvent));

  return PI_SUCCESS;
}

// Command-buffer extension
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// usm-p2p

inline pi_result piextEnablePeerAccess(pi_device command_device,
                                       pi_device peer_device) {
  auto commandDevice = reinterpret_cast<ur_device_handle_t>(command_device);
  auto peerDevice = reinterpret_cast<ur_device_handle_t>(peer_device);

  HANDLE_ERRORS(urUsmP2PEnablePeerAccessExp(commandDevice, peerDevice));

  return PI_SUCCESS;
}

inline pi_result piextDisablePeerAccess(pi_device command_device,
                                        pi_device peer_device) {
  auto commandDevice = reinterpret_cast<ur_device_handle_t>(command_device);
  auto peerDevice = reinterpret_cast<ur_device_handle_t>(peer_device);

  HANDLE_ERRORS(urUsmP2PDisablePeerAccessExp(commandDevice, peerDevice));

  return PI_SUCCESS;
}

inline pi_result
piextPeerAccessGetInfo(pi_device command_device, pi_device peer_device,
                       pi_peer_attr attr, size_t param_value_size,
                       void *param_value, size_t *param_value_size_ret) {
  auto commandDevice = reinterpret_cast<ur_device_handle_t>(command_device);
  auto peerDevice = reinterpret_cast<ur_device_handle_t>(peer_device);

  ur_exp_peer_info_t propName;
  switch (attr) {
  case PI_PEER_ACCESS_SUPPORTED: {
    propName = UR_EXP_PEER_INFO_UR_PEER_ACCESS_SUPPORTED;
    break;
  }
  case PI_PEER_ATOMICS_SUPPORTED: {
    propName = UR_EXP_PEER_INFO_UR_PEER_ATOMICS_SUPPORTED;
    break;
  }
  default: {
    return PI_ERROR_INVALID_VALUE;
  }
  }

  HANDLE_ERRORS(urUsmP2PPeerAccessGetInfoExp(
      commandDevice, peerDevice, propName, param_value_size, param_value,
      param_value_size_ret));

  return PI_SUCCESS;
}

// usm-p2p
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Bindless Images Extension

inline pi_result piextMemImageAllocate(pi_context Context, pi_device Device,
                                       pi_image_format *ImageFormat,
                                       pi_image_desc *ImageDesc,
                                       pi_image_mem_handle *RetMem) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);

  ur_image_format_t UrFormat{};
  ur_image_desc_t UrDesc{};
  pi2urImageDesc(ImageFormat, ImageDesc, &UrFormat, &UrDesc);

  ur_exp_image_mem_handle_t *UrRetMem =
      reinterpret_cast<ur_exp_image_mem_handle_t *>(RetMem);

  HANDLE_ERRORS(urBindlessImagesImageAllocateExp(UrContext, UrDevice, &UrFormat,
                                                 &UrDesc, UrRetMem));

  return PI_SUCCESS;
}

inline pi_result piextMemUnsampledImageCreate(
    pi_context Context, pi_device Device, pi_image_mem_handle ImgMem,
    pi_image_format *ImageFormat, pi_image_desc *ImageDesc, pi_mem *RetMem,
    pi_image_handle *RetHandle) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(RetMem, PI_ERROR_INVALID_MEM_OBJECT);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  auto UrImgMem = reinterpret_cast<ur_exp_image_mem_handle_t>(ImgMem);

  ur_image_format_t UrFormat{};
  ur_image_desc_t UrDesc{};
  pi2urImageDesc(ImageFormat, ImageDesc, &UrFormat, &UrDesc);

  ur_mem_handle_t *UrRetMem = reinterpret_cast<ur_mem_handle_t *>(RetMem);
  ur_exp_image_handle_t *UrRetHandle =
      reinterpret_cast<ur_exp_image_handle_t *>(RetHandle);

  HANDLE_ERRORS(urBindlessImagesUnsampledImageCreateExp(
      UrContext, UrDevice, UrImgMem, &UrFormat, &UrDesc, UrRetMem,
      UrRetHandle));

  return PI_SUCCESS;
}

inline pi_result piextMemSampledImageCreate(
    pi_context Context, pi_device Device, pi_image_mem_handle ImgMem,
    pi_image_format *ImageFormat, pi_image_desc *ImageDesc, pi_sampler Sampler,
    pi_mem *RetMem, pi_image_handle *RetHandle) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(RetMem, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Sampler, PI_ERROR_INVALID_SAMPLER);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  auto UrImgMem = reinterpret_cast<ur_exp_image_mem_handle_t>(ImgMem);

  ur_image_format_t UrFormat{};
  ur_image_desc_t UrDesc{};
  pi2urImageDesc(ImageFormat, ImageDesc, &UrFormat, &UrDesc);

  auto UrSampler = reinterpret_cast<ur_sampler_handle_t>(Sampler);
  ur_mem_handle_t *UrRetMem = reinterpret_cast<ur_mem_handle_t *>(RetMem);
  ur_exp_image_handle_t *UrRetHandle =
      reinterpret_cast<ur_exp_image_handle_t *>(RetHandle);

  HANDLE_ERRORS(urBindlessImagesSampledImageCreateExp(
      UrContext, UrDevice, UrImgMem, &UrFormat, &UrDesc, UrSampler, UrRetMem,
      UrRetHandle));

  return PI_SUCCESS;
}

inline pi_result piextBindlessImageSamplerCreate(
    pi_context Context, const pi_sampler_properties *SamplerProperties,
    float MinMipmapLevelClamp, float MaxMipmapLevelClamp, float MaxAnisotropy,
    pi_sampler *RetSampler) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(RetSampler, PI_ERROR_INVALID_VALUE);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  ur_sampler_desc_t UrProps{};
  UrProps.stype = UR_STRUCTURE_TYPE_SAMPLER_DESC;

  ur_exp_sampler_mip_properties_t UrMipProps{};
  UrMipProps.stype = UR_STRUCTURE_TYPE_EXP_SAMPLER_MIP_PROPERTIES;
  UrMipProps.minMipmapLevelClamp = MinMipmapLevelClamp;
  UrMipProps.maxMipmapLevelClamp = MaxMipmapLevelClamp;
  UrMipProps.maxAnisotropy = MaxAnisotropy;
  UrProps.pNext = &UrMipProps;

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

    case PI_SAMPLER_PROPERTIES_MIP_FILTER_MODE: {
      pi_sampler_filter_mode CurValueFilterMode =
          ur_cast<pi_sampler_filter_mode>(ur_cast<pi_uint32>(*(++CurProperty)));

      if (CurValueFilterMode == PI_SAMPLER_FILTER_MODE_NEAREST)
        UrMipProps.mipFilterMode = UR_SAMPLER_FILTER_MODE_NEAREST;
      else if (CurValueFilterMode == PI_SAMPLER_FILTER_MODE_LINEAR)
        UrMipProps.mipFilterMode = UR_SAMPLER_FILTER_MODE_LINEAR;
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

inline pi_result piextMemMipmapGetLevel(pi_context Context, pi_device Device,
                                        pi_image_mem_handle MipMem,
                                        unsigned int Level,
                                        pi_image_mem_handle *RetMem) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  auto UrMipMem = reinterpret_cast<ur_exp_image_mem_handle_t>(MipMem);
  ur_exp_image_mem_handle_t *UrRetMem =
      reinterpret_cast<ur_exp_image_mem_handle_t *>(RetMem);

  HANDLE_ERRORS(urBindlessImagesMipmapGetLevelExp(UrContext, UrDevice, UrMipMem,
                                                  Level, UrRetMem));

  return PI_SUCCESS;
}

inline pi_result piextMemImageFree(pi_context Context, pi_device Device,
                                   pi_image_mem_handle MemoryHandle) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  auto UrMemoryHandle =
      reinterpret_cast<ur_exp_image_mem_handle_t>(MemoryHandle);

  HANDLE_ERRORS(
      urBindlessImagesImageFreeExp(UrContext, UrDevice, UrMemoryHandle));

  return PI_SUCCESS;
}

inline pi_result piextMemMipmapFree(pi_context Context, pi_device Device,
                                    pi_image_mem_handle MemoryHandle) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  auto UrMemoryHandle =
      reinterpret_cast<ur_exp_image_mem_handle_t>(MemoryHandle);

  HANDLE_ERRORS(
      urBindlessImagesMipmapFreeExp(UrContext, UrDevice, UrMemoryHandle));

  return PI_SUCCESS;
}

static void pi2urImageCopyFlags(const pi_image_copy_flags PiFlags,
                                ur_exp_image_copy_flags_t *UrFlags) {
  switch (PiFlags) {
  case PI_IMAGE_COPY_HOST_TO_DEVICE:
    *UrFlags = UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE;
    break;
  case PI_IMAGE_COPY_DEVICE_TO_HOST:
    *UrFlags = UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST;
    break;
  case PI_IMAGE_COPY_DEVICE_TO_DEVICE:
    *UrFlags = UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE;
    break;
  default:
    die("pi2urImageCopyFlags: Unsupported use case");
  }
}

inline pi_result
piextMemImageCopy(pi_queue Queue, void *DstPtr, void *SrcPtr,
                  const pi_image_format *ImageFormat,
                  const pi_image_desc *ImageDesc,
                  const pi_image_copy_flags Flags, pi_image_offset SrcOffset,
                  pi_image_offset DstOffset, pi_image_region CopyExtent,
                  pi_image_region HostExtent, pi_uint32 NumEventsInWaitList,
                  const pi_event *EventWaitList, pi_event *Event) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  auto UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);

  ur_image_format_t UrFormat{};
  ur_image_desc_t UrDesc{};
  pi2urImageDesc(ImageFormat, ImageDesc, &UrFormat, &UrDesc);

  ur_exp_image_copy_flags_t UrFlags;
  pi2urImageCopyFlags(Flags, &UrFlags);

  ur_rect_offset_t UrSrcOffset{SrcOffset->x, SrcOffset->y, SrcOffset->z};
  ur_rect_offset_t UrDstOffset{DstOffset->x, DstOffset->y, DstOffset->z};
  ur_rect_region_t UrCopyExtent{};
  UrCopyExtent.depth = CopyExtent->depth;
  UrCopyExtent.height = CopyExtent->height;
  UrCopyExtent.width = CopyExtent->width;
  ur_rect_region_t UrHostExtent{};
  UrHostExtent.depth = HostExtent->depth;
  UrHostExtent.height = HostExtent->height;
  UrHostExtent.width = HostExtent->width;

  const ur_event_handle_t *UrEventWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventWaitList);
  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(Event);

  HANDLE_ERRORS(urBindlessImagesImageCopyExp(
      UrQueue, DstPtr, SrcPtr, &UrFormat, &UrDesc, UrFlags, UrSrcOffset,
      UrDstOffset, UrCopyExtent, UrHostExtent, NumEventsInWaitList,
      UrEventWaitList, UREvent));

  return PI_SUCCESS;
}

inline pi_result piextMemUnsampledImageHandleDestroy(pi_context Context,
                                                     pi_device Device,
                                                     pi_image_handle Handle) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  auto UrHandle = reinterpret_cast<ur_exp_image_handle_t>(Handle);

  HANDLE_ERRORS(urBindlessImagesUnsampledImageHandleDestroyExp(
      UrContext, UrDevice, UrHandle));

  return PI_SUCCESS;
}

inline pi_result piextMemSampledImageHandleDestroy(pi_context Context,
                                                   pi_device Device,
                                                   pi_image_handle Handle) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  auto UrHandle = reinterpret_cast<ur_exp_image_handle_t>(Handle);

  HANDLE_ERRORS(urBindlessImagesSampledImageHandleDestroyExp(
      UrContext, UrDevice, UrHandle));

  return PI_SUCCESS;
}

static void pi2urImageInfoFlags(const pi_image_info PiFlags,
                                ur_image_info_t *UrFlags) {
  switch (PiFlags) {
#define PI_TO_UR_IMAGE_INFO(FROM, TO)                                          \
  case FROM: {                                                                 \
    *UrFlags = TO;                                                             \
    return;                                                                    \
  }
    PI_TO_UR_IMAGE_INFO(PI_IMAGE_INFO_FORMAT, UR_IMAGE_INFO_FORMAT)
    PI_TO_UR_IMAGE_INFO(PI_IMAGE_INFO_ELEMENT_SIZE, UR_IMAGE_INFO_ELEMENT_SIZE)
    PI_TO_UR_IMAGE_INFO(PI_IMAGE_INFO_ROW_PITCH, UR_IMAGE_INFO_ROW_PITCH)
    PI_TO_UR_IMAGE_INFO(PI_IMAGE_INFO_SLICE_PITCH, UR_IMAGE_INFO_SLICE_PITCH)
    PI_TO_UR_IMAGE_INFO(PI_IMAGE_INFO_WIDTH, UR_IMAGE_INFO_WIDTH)
    PI_TO_UR_IMAGE_INFO(PI_IMAGE_INFO_HEIGHT, UR_IMAGE_INFO_HEIGHT)
    PI_TO_UR_IMAGE_INFO(PI_IMAGE_INFO_DEPTH, UR_IMAGE_INFO_DEPTH)
#undef PI_TO_UR_IMAGE_INFO
  default:
    die("pi2urImageInfoFlags: Unsupported use case");
  }
}

inline pi_result piextMemImageGetInfo(pi_image_mem_handle MemHandle,
                                      pi_image_info ParamName, void *ParamValue,
                                      size_t *ParamValueSizeRet) {
  auto UrMemHandle = reinterpret_cast<ur_exp_image_mem_handle_t>(MemHandle);

  ur_image_info_t UrParamName{};
  pi2urImageInfoFlags(ParamName, &UrParamName);

  HANDLE_ERRORS(urBindlessImagesImageGetInfoExp(UrMemHandle, UrParamName,
                                                ParamValue, ParamValueSizeRet));

  if (ParamName == pi_image_info::PI_IMAGE_INFO_FORMAT && ParamValue) {
    pi_image_format PiFormat;
    ur2piImageFormat(reinterpret_cast<ur_image_format_t *>(ParamValue),
                     &PiFormat);
    reinterpret_cast<pi_image_format *>(ParamValue)->image_channel_data_type =
        PiFormat.image_channel_data_type;
    reinterpret_cast<pi_image_format *>(ParamValue)->image_channel_order =
        PiFormat.image_channel_order;
    if (ParamValueSizeRet) {
      *ParamValueSizeRet = sizeof(pi_image_format);
    }
  }

  return PI_SUCCESS;
}

inline pi_result piextMemImportOpaqueFD(pi_context Context, pi_device Device,
                                        size_t Size, int FileDescriptor,
                                        pi_interop_mem_handle *RetHandle) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  ur_exp_interop_mem_handle_t *UrRetHandle =
      reinterpret_cast<ur_exp_interop_mem_handle_t *>(RetHandle);

  ur_exp_file_descriptor_t PosixFD{};
  PosixFD.stype = UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR;
  PosixFD.fd = FileDescriptor;

  ur_exp_interop_mem_desc_t InteropMemDesc{};
  InteropMemDesc.stype = UR_STRUCTURE_TYPE_EXP_INTEROP_MEM_DESC;
  InteropMemDesc.pNext = &PosixFD;

  HANDLE_ERRORS(urBindlessImagesImportOpaqueFDExp(
      UrContext, UrDevice, Size, &InteropMemDesc, UrRetHandle));

  return PI_SUCCESS;
}

inline pi_result piextMemMapExternalArray(pi_context Context, pi_device Device,
                                          pi_image_format *ImageFormat,
                                          pi_image_desc *ImageDesc,
                                          pi_interop_mem_handle MemHandle,
                                          pi_image_mem_handle *RetMem) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);

  ur_image_format_t UrFormat{};
  ur_image_desc_t UrDesc{};
  pi2urImageDesc(ImageFormat, ImageDesc, &UrFormat, &UrDesc);

  auto UrMemHandle = reinterpret_cast<ur_exp_interop_mem_handle_t>(MemHandle);
  ur_exp_image_mem_handle_t *UrRetMem =
      reinterpret_cast<ur_exp_image_mem_handle_t *>(RetMem);

  HANDLE_ERRORS(urBindlessImagesMapExternalArrayExp(
      UrContext, UrDevice, &UrFormat, &UrDesc, UrMemHandle, UrRetMem));

  return PI_SUCCESS;
}

inline pi_result piextMemReleaseInterop(pi_context Context, pi_device Device,
                                        pi_interop_mem_handle ExtMem) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  auto UrExtMem = reinterpret_cast<ur_exp_interop_mem_handle_t>(ExtMem);

  HANDLE_ERRORS(
      urBindlessImagesReleaseInteropExp(UrContext, UrDevice, UrExtMem));

  return PI_SUCCESS;
}

inline pi_result
piextImportExternalSemaphoreOpaqueFD(pi_context Context, pi_device Device,
                                     int FileDescriptor,
                                     pi_interop_semaphore_handle *RetHandle) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  ur_exp_interop_semaphore_handle_t *UrRetHandle =
      reinterpret_cast<ur_exp_interop_semaphore_handle_t *>(RetHandle);

  ur_exp_file_descriptor_t PosixFD{};
  PosixFD.stype = UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR;
  PosixFD.fd = FileDescriptor;

  ur_exp_interop_semaphore_desc_t InteropSemDesc{};
  InteropSemDesc.stype = UR_STRUCTURE_TYPE_EXP_INTEROP_SEMAPHORE_DESC;
  InteropSemDesc.pNext = &PosixFD;

  HANDLE_ERRORS(urBindlessImagesImportExternalSemaphoreOpaqueFDExp(
      UrContext, UrDevice, &InteropSemDesc, UrRetHandle));

  return PI_SUCCESS;
}

inline pi_result
piextDestroyExternalSemaphore(pi_context Context, pi_device Device,
                              pi_interop_semaphore_handle SemHandle) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  auto UrContext = reinterpret_cast<ur_context_handle_t>(Context);
  auto UrDevice = reinterpret_cast<ur_device_handle_t>(Device);
  auto UrSemHandle =
      reinterpret_cast<ur_exp_interop_semaphore_handle_t>(SemHandle);

  HANDLE_ERRORS(urBindlessImagesDestroyExternalSemaphoreExp(UrContext, UrDevice,
                                                            UrSemHandle));

  return PI_SUCCESS;
}

inline pi_result
piextWaitExternalSemaphore(pi_queue Queue,
                           pi_interop_semaphore_handle SemHandle,
                           pi_uint32 NumEventsInWaitList,
                           const pi_event *EventWaitList, pi_event *Event) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  auto UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  auto UrSemHandle =
      reinterpret_cast<ur_exp_interop_semaphore_handle_t>(SemHandle);
  const ur_event_handle_t *UrEventWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventWaitList);
  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(Event);

  HANDLE_ERRORS(urBindlessImagesWaitExternalSemaphoreExp(
      UrQueue, UrSemHandle, NumEventsInWaitList, UrEventWaitList, UREvent));

  return PI_SUCCESS;
}

inline pi_result
piextSignalExternalSemaphore(pi_queue Queue,
                             pi_interop_semaphore_handle SemHandle,
                             pi_uint32 NumEventsInWaitList,
                             const pi_event *EventWaitList, pi_event *Event) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  auto UrQueue = reinterpret_cast<ur_queue_handle_t>(Queue);
  auto UrSemHandle =
      reinterpret_cast<ur_exp_interop_semaphore_handle_t>(SemHandle);
  const ur_event_handle_t *UrEventWaitList =
      reinterpret_cast<const ur_event_handle_t *>(EventWaitList);
  ur_event_handle_t *UREvent = reinterpret_cast<ur_event_handle_t *>(Event);

  HANDLE_ERRORS(urBindlessImagesSignalExternalSemaphoreExp(
      UrQueue, UrSemHandle, NumEventsInWaitList, UrEventWaitList, UREvent));

  return PI_SUCCESS;
}

// Bindless Images Extension
///////////////////////////////////////////////////////////////////////////////

} // namespace pi2ur
