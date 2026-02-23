//===----------- device.cpp - LLVM Offload Adapter  -----------------------===//
//
// Copyright (C) 2025-2026 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>
#include <ur/ur.hpp>
#include <ur_api.h>

#include "adapters/offload/adapter.hpp"
#include "device.hpp"
#include "platform.hpp"
#include "ur2offload.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGet(ur_platform_handle_t hPlatform,
                                                ur_device_type_t,
                                                uint32_t NumEntries,
                                                ur_device_handle_t *phDevices,
                                                uint32_t *pNumDevices) {
  if (pNumDevices) {
    *pNumDevices = static_cast<uint32_t>(hPlatform->Devices.size());
  }

  size_t NumDevices =
      std::min(static_cast<uint32_t>(hPlatform->Devices.size()), NumEntries);

  for (size_t I = 0; I < NumDevices; I++) {
    phDevices[I] = hPlatform->Devices[I].get();
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetInfo(ur_device_handle_t hDevice,
                                                    ur_device_info_t propName,
                                                    size_t propSize,
                                                    void *pPropValue,
                                                    size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  ol_device_info_t olInfo;
  switch (propName) {
  case UR_DEVICE_INFO_NAME:
    olInfo = OL_DEVICE_INFO_NAME;
    break;
  case UR_DEVICE_INFO_PARENT_DEVICE:
    return ReturnValue(nullptr);
  case UR_DEVICE_INFO_VERSION:
    return ReturnValue("");
  case UR_DEVICE_INFO_EXTENSIONS:
    // todo: use offload API to query supported extensions
    return ReturnValue("cl_khr_il_program");
  case UR_DEVICE_INFO_USE_NATIVE_ASSERT:
    return ReturnValue(false);
  case UR_DEVICE_INFO_TYPE:
    olInfo = OL_DEVICE_INFO_TYPE;
    break;
  case UR_DEVICE_INFO_VENDOR:
    olInfo = OL_DEVICE_INFO_VENDOR;
    break;
  case UR_DEVICE_INFO_VENDOR_ID:
    olInfo = OL_DEVICE_INFO_VENDOR_ID;
    break;
  case UR_DEVICE_INFO_DRIVER_VERSION:
    olInfo = OL_DEVICE_INFO_DRIVER_VERSION;
    break;
  case UR_DEVICE_INFO_PLATFORM:
    return ReturnValue(hDevice->Platform);
    break;
  case UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION:
    return ReturnValue((std::to_string(OL_VERSION_MAJOR) + "." +
                        std::to_string(OL_VERSION_MINOR) + "." +
                        std::to_string(OL_VERSION_PATCH))
                           .c_str());
  case UR_DEVICE_INFO_PROFILE:
    // This doesn't make sense for non-opencl devices, copy other backends and
    // just return FULL_PROFILE
    return ReturnValue("FULL_PROFILE");
  case UR_DEVICE_INFO_MAX_COMPUTE_UNITS:
  case UR_DEVICE_INFO_NUM_COMPUTE_UNITS:
    olInfo = OL_DEVICE_INFO_NUM_COMPUTE_UNITS;
    break;
  case UR_DEVICE_INFO_SINGLE_FP_CONFIG:
    olInfo = OL_DEVICE_INFO_SINGLE_FP_CONFIG;
    break;
  case UR_DEVICE_INFO_HALF_FP_CONFIG:
    olInfo = OL_DEVICE_INFO_HALF_FP_CONFIG;
    break;
  case UR_DEVICE_INFO_DOUBLE_FP_CONFIG:
    olInfo = OL_DEVICE_INFO_DOUBLE_FP_CONFIG;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR:
    olInfo = OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT:
    olInfo = OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT:
    olInfo = OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG:
    olInfo = OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT:
    olInfo = OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE:
    olInfo = OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE;
    break;
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF:
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF:
    olInfo = OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF;
    break;
  case UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY:
    olInfo = OL_DEVICE_INFO_MAX_CLOCK_FREQUENCY;
    break;
  case UR_DEVICE_INFO_MEMORY_CLOCK_RATE:
    olInfo = OL_DEVICE_INFO_MEMORY_CLOCK_RATE;
    break;
  case UR_DEVICE_INFO_ADDRESS_BITS:
    olInfo = OL_DEVICE_INFO_ADDRESS_BITS;
    break;
  case UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE:
    olInfo = OL_DEVICE_INFO_MAX_MEM_ALLOC_SIZE;
    break;
  case UR_DEVICE_INFO_GLOBAL_MEM_SIZE:
    olInfo = OL_DEVICE_INFO_GLOBAL_MEM_SIZE;
    break;
  case UR_DEVICE_INFO_USM_DEVICE_SUPPORT:
  case UR_DEVICE_INFO_USM_HOST_SUPPORT:
  case UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
    return ReturnValue(UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS);
  case UR_DEVICE_INFO_BUILD_ON_SUBDEVICE:
    return ReturnValue(false);
  case UR_DEVICE_INFO_REFERENCE_COUNT:
    // Devices are never allocated or freed
    return ReturnValue(uint32_t{1});
  case UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS:
    return ReturnValue(uint32_t{3});
  case UR_DEVICE_INFO_COMPILER_AVAILABLE:
  case UR_DEVICE_INFO_GLOBAL_VARIABLE_SUPPORT:
    return ReturnValue(true);
  case UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL:
    // TODO: Implement subgroups in Offload
    return ReturnValue(1);
  case UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(size_t);
    }

    if (pPropValue) {
      uint32_t as32;
      OL_RETURN_ON_ERR(olGetDeviceInfo(hDevice->OffloadDevice,
                                       OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
                                       sizeof(as32), &as32));

      *reinterpret_cast<size_t *>(pPropValue) = as32;
    }

    return UR_RESULT_SUCCESS;
  case UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    // OL dimensions are uint32_t while UR is size_t, so they need to be mapped
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(size_t) * 3;
    }

    if (pPropValue) {
      ol_dimensions_t olVec;
      size_t *urVec = reinterpret_cast<size_t *>(pPropValue);
      OL_RETURN_ON_ERR(
          olGetDeviceInfo(hDevice->OffloadDevice,
                          OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION,
                          sizeof(olVec), &olVec));

      urVec[0] = olVec.x;
      urVec[1] = olVec.y;
      urVec[2] = olVec.z;
    }

    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_MAX_WORK_GROUPS_3D: {
    // OL dimensions are uint32_t while UR is size_t, so they need to be mapped
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(size_t) * 3;
    }

    if (pPropValue) {
      ol_dimensions_t olVec;
      size_t *urVec = reinterpret_cast<size_t *>(pPropValue);
      OL_RETURN_ON_ERR(olGetDeviceInfo(
          hDevice->OffloadDevice, OL_DEVICE_INFO_MAX_WORK_SIZE_PER_DIMENSION,
          sizeof(olVec), &olVec));

      urVec[0] = olVec.x;
      urVec[1] = olVec.y;
      urVec[2] = olVec.z;
    }

    return UR_RESULT_SUCCESS;
  }

  // Unimplemented features
  case UR_DEVICE_INFO_PROGRAM_SET_SPECIALIZATION_CONSTANTS:
  case UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS:
  case UR_DEVICE_INFO_USM_POOL_SUPPORT:
  case UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP:
  case UR_DEVICE_INFO_IMAGE_SUPPORT:
  case UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT:
  case UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT:
  case UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORT:
  case UR_DEVICE_INFO_ASYNC_BARRIER:
  case UR_DEVICE_INFO_ESIMD_SUPPORT:
  case UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS:
  case UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT:
  case UR_DEVICE_INFO_USM_P2P_SUPPORT_EXP:
  case UR_DEVICE_INFO_USM_CONTEXT_MEMCPY_SUPPORT_EXP:
  // TODO: Atomic queries in Offload
  case UR_DEVICE_INFO_ATOMIC_64:
  case UR_DEVICE_INFO_IMAGE_SRGB:
  case UR_DEVICE_INFO_HOST_UNIFIED_MEMORY:
  case UR_DEVICE_INFO_LINKER_AVAILABLE:
  case UR_DEVICE_INFO_TIMESTAMP_RECORDING_SUPPORT_EXP:
    return ReturnValue(false);
  case UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT:
  case UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT:
    return ReturnValue(uint32_t{0});
  case UR_DEVICE_INFO_QUEUE_PROPERTIES:
  case UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES:
    return ReturnValue(
        ur_queue_flags_t{UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE});
  case UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES:
    return ReturnValue(0);
  case UR_DEVICE_INFO_DYNAMIC_LINK_SUPPORT_EXP:
    return ReturnValue(false);
  case UR_DEVICE_INFO_KERNEL_LAUNCH_CAPABILITIES:
    return ReturnValue(0);
  case UR_DEVICE_INFO_SUPPORTED_PARTITIONS: {
    if (pPropSizeRet) {
      *pPropSizeRet = 0;
    }
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES: {
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: {
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_PARTITION_TYPE: {
    if (pPropSizeRet) {
      *pPropSizeRet = 0;
    }
    return UR_RESULT_SUCCESS;
  }
  case UR_DEVICE_INFO_AVAILABLE: {
    return ReturnValue(ur_bool_t{true});
  }
  case UR_DEVICE_INFO_BUILT_IN_KERNELS: {
    // An empty string is returned if no built-in kernels are supported by the
    // device.
    return ReturnValue("");
  }
  case UR_DEVICE_INFO_ENDIAN_LITTLE: {
    return ReturnValue(ur_bool_t{true});
  }
  case UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION: {
    // Liboffload has no profiling timers yet
    return ReturnValue(size_t{0});
  }
  case UR_DEVICE_INFO_LOCAL_MEM_TYPE: {
    return ReturnValue(UR_DEVICE_LOCAL_MEM_TYPE_NONE);
  }
  case UR_DEVICE_INFO_LOCAL_MEM_SIZE: {
    return ReturnValue(size_t{0});
  }
  case UR_DEVICE_INFO_DEVICE_WAIT_SUPPORT_EXP: {
    return ReturnValue(ur_bool_t{false});
  }

  // The following properties are lifted from the minimum supported
  // intersection of the HIP and CUDA backends until liboffload adds a specific
  // query
  case UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES: {
    ur_memory_order_capability_flags_t Capabilities =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL;

    return ReturnValue(Capabilities);
  }
  case UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    ur_memory_scope_capability_flags_t Capabilities =
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP;
    return ReturnValue(Capabilities);
  }
  case UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: {
    ur_memory_order_capability_flags_t Capabilities =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE;
    return ReturnValue(Capabilities);
  }
  case UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES: {
    ur_memory_scope_capability_flags_t Capabilities =
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
        UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP;
    return ReturnValue(Capabilities);
  }
  case UR_DEVICE_INFO_BFLOAT16_CONVERSIONS_NATIVE:
    return ReturnValue(false);
  case UR_DEVICE_INFO_EXECUTION_CAPABILITIES: {
    auto Capability = ur_device_exec_capability_flags_t{
        UR_DEVICE_EXEC_CAPABILITY_FLAG_KERNEL};
    return ReturnValue(Capability);
  }
  case UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES: {
    return ReturnValue(int32_t{1});
  }
  case UR_DEVICE_INFO_MAX_PARAMETER_SIZE: {
    return ReturnValue(size_t{4000});
  }
  case UR_DEVICE_INFO_MAX_CONSTANT_ARGS: {
    return ReturnValue(uint32_t{9});
  }
  case UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC: {
    return ReturnValue(ur_bool_t{true});
  }
  case UR_DEVICE_INFO_PRINTF_BUFFER_SIZE: {
    // The minimum value for the FULL profile is 1 MB.
    return ReturnValue(size_t(1024));
  }

  // No image support in liboffload yet, so just return 0 for these properties
  case UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
  case UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
  case UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT:
  case UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH:
  case UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH:
  case UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE:
  case UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE:
    return ReturnValue(size_t{0});
  case UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS:
  case UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS:
  case UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS:
  case UR_DEVICE_INFO_MAX_SAMPLERS:
    return ReturnValue(uint32_t{0});
  case UR_DEVICE_INFO_ENQUEUE_HOST_TASK_SUPPORT_EXP:
    return ReturnValue(false);
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }

  if (pPropSizeRet) {
    OL_RETURN_ON_ERR(
        olGetDeviceInfoSize(hDevice->OffloadDevice, olInfo, pPropSizeRet));
  }

  if (pPropValue) {
    OL_RETURN_ON_ERR(
        olGetDeviceInfo(hDevice->OffloadDevice, olInfo, propSize, pPropValue));

    // Need to explicitly map these types
    switch (olInfo) {
    case OL_DEVICE_INFO_TYPE: {
      auto *urPropPtr = reinterpret_cast<ur_device_type_t *>(pPropValue);
      auto *olPropPtr = reinterpret_cast<ol_device_type_t *>(pPropValue);

      switch (*olPropPtr) {
      case OL_DEVICE_TYPE_CPU:
        *urPropPtr = UR_DEVICE_TYPE_CPU;
        break;
      case OL_DEVICE_TYPE_GPU:
        *urPropPtr = UR_DEVICE_TYPE_GPU;
        break;
      default:
        break;
      }
      break;
    }
    case OL_DEVICE_INFO_SINGLE_FP_CONFIG:
    case OL_DEVICE_INFO_HALF_FP_CONFIG:
    case OL_DEVICE_INFO_DOUBLE_FP_CONFIG: {
      auto olValue =
          *reinterpret_cast<ol_device_fp_capability_flags_t *>(pPropValue);
      ur_device_fp_capability_flags_t urValue{0};
      if (olValue & OL_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT)
        urValue |= UR_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
      if (olValue & OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST)
        urValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
      if (olValue & OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO)
        urValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
      if (olValue & OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF)
        urValue |= UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF;
      if (olValue & OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN)
        urValue |= UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN;
      if (olValue & OL_DEVICE_FP_CAPABILITY_FLAG_DENORM)
        urValue |= UR_DEVICE_FP_CAPABILITY_FLAG_DENORM;
      if (olValue & OL_DEVICE_FP_CAPABILITY_FLAG_FMA)
        urValue |= UR_DEVICE_FP_CAPABILITY_FLAG_FMA;
      if (olValue & OL_DEVICE_FP_CAPABILITY_FLAG_SOFT_FLOAT)
        urValue |= UR_DEVICE_FP_CAPABILITY_FLAG_SOFT_FLOAT;
      auto *urPropPtr =
          reinterpret_cast<ur_device_fp_capability_flags_t *>(pPropValue);
      *urPropPtr = urValue;
    }
    default:
      break;
    }
  }

  return UR_RESULT_SUCCESS;
}

// Device partitioning is not supported in Offload, and won't be for some time.
// This means urDeviceRetain/Release are no-ops because all devices are root
// devices.

UR_APIEXPORT ur_result_t UR_APICALL urDeviceRetain(ur_device_handle_t) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceRelease(ur_device_handle_t) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urDevicePartition(ur_device_handle_t, const ur_device_partition_properties_t *,
                  uint32_t, ur_device_handle_t *, uint32_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceSelectBinary(
    ur_device_handle_t hDevice, const ur_device_binary_t *pBinaries,
    uint32_t NumBinaries, uint32_t *pSelectedBinary) {

  ol_platform_backend_t Backend;
  OL_RETURN_ON_ERR(olGetPlatformInfo(hDevice->Platform->OffloadPlatform,
                                     OL_PLATFORM_INFO_BACKEND, sizeof(Backend),
                                     &Backend));

  const char *ImageTarget = UR_DEVICE_BINARY_TARGET_UNKNOWN;
  if (Backend == OL_PLATFORM_BACKEND_CUDA) {
    ImageTarget = UR_DEVICE_BINARY_TARGET_NVPTX64;
  } else if (Backend == OL_PLATFORM_BACKEND_AMDGPU) {
    ImageTarget = UR_DEVICE_BINARY_TARGET_AMDGCN;
  } else if (Backend == OL_PLATFORM_BACKEND_LEVEL_ZERO) {
    ImageTarget = UR_DEVICE_BINARY_TARGET_SPIRV64;
  }

  for (uint32_t i = 0; i < NumBinaries; ++i) {
    if (strcmp(pBinaries[i].pDeviceTargetSpec, ImageTarget) == 0) {
      *pSelectedBinary = i;
      return UR_RESULT_SUCCESS;
    }
  }

  // No image can be loaded for the given device
  return UR_RESULT_ERROR_INVALID_BINARY;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetNativeHandle(
    ur_device_handle_t UrDevice, ur_native_handle_t *Handle) {
  *Handle = reinterpret_cast<ur_native_handle_t>(UrDevice->OffloadDevice);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceCreateWithNativeHandle(
    ur_native_handle_t hNativeDevice, ur_adapter_handle_t hAdapter,
    const ur_device_native_properties_t *, ur_device_handle_t *phDevice) {
  ol_device_handle_t OlDevice =
      reinterpret_cast<ol_device_handle_t>(hNativeDevice);

  // Currently, all devices are found at initialization, there is no way to
  // create sub devices yet
  for (auto &P : hAdapter->Platforms) {
    auto Found =
        std::find_if(P->Devices.begin(), P->Devices.end(),
                     [&](std::unique_ptr<ur_device_handle_t_> &PDevice) {
                       return PDevice->OffloadDevice == OlDevice;
                     });
    if (Found != P->Devices.end()) {
      *phDevice = Found->get();
      return UR_RESULT_SUCCESS;
    }
  }

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urDeviceGetGlobalTimestamps(ur_device_handle_t, uint64_t *, uint64_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceWaitExp(ur_device_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
