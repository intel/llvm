/*
 *
 * Copyright (C) 2019-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_null.cpp
 *
 */
#include "ur_mock.hpp"
#include "ur_mock_helpers.hpp"

namespace driver {
//////////////////////////////////////////////////////////////////////////
context_t d_context;

ur_result_t mock_urPlatformGetApiVersion(void *pParams) {
  const auto &params =
      *static_cast<ur_platform_get_api_version_params_t *>(pParams);
  **params.ppVersion = d_context.version;
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urPlatformGetInfo(void *pParams) {
  const auto &params = *static_cast<ur_platform_get_info_params_t *>(pParams);
  if (!*params.phPlatform) {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }

  if (*params.ppropName == UR_PLATFORM_INFO_NAME) {
    const char mock_platform_name[] = "Mock Platform";
    if (*params.ppPropSizeRet) {
      **params.ppPropSizeRet = sizeof(mock_platform_name);
    }
    if (*params.ppPropValue) {
#if defined(_WIN32)
      strncpy_s(reinterpret_cast<char *>(*params.ppPropValue),
                *params.ppropSize, mock_platform_name,
                sizeof(mock_platform_name));
#else
      strncpy(reinterpret_cast<char *>(*params.ppPropValue), mock_platform_name,
              *params.ppropSize);
#endif
    }
  }
  return UR_RESULT_SUCCESS;
}

//////////////////////////////////////////////////////////////////////////
ur_result_t mock_urDeviceGetInfo(void *pParams) {
  const auto &params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_TYPE:
    if (*params.ppPropValue != nullptr) {
      *reinterpret_cast<ur_device_type_t *>(*params.ppPropValue) =
          UR_DEVICE_TYPE_GPU;
    }
    if (*params.ppPropSizeRet != nullptr) {
      **params.ppPropSizeRet = sizeof(ur_device_type_t);
    }
    break;

  case UR_DEVICE_INFO_NAME: {
    char deviceName[] = "Mock Device";
    if (*params.ppPropValue && *params.ppropSize < sizeof(deviceName)) {
      return UR_RESULT_ERROR_INVALID_SIZE;
    }
    if (*params.ppPropValue != nullptr) {
#if defined(_WIN32)
      strncpy_s(reinterpret_cast<char *>(*params.ppPropValue),
                *params.ppropSize, deviceName, sizeof(deviceName));
#else
      strncpy(reinterpret_cast<char *>(*params.ppPropValue), deviceName,
              *params.ppropSize);
#endif
    }
    if (*params.ppPropSizeRet != nullptr) {
      **params.ppPropSizeRet = sizeof(deviceName);
    }
  } break;
  case UR_DEVICE_INFO_PLATFORM:
    if (*params.ppPropValue != nullptr) {
      *reinterpret_cast<ur_platform_handle_t *>(*params.ppPropValue) =
          driver::d_context.platform;
    }
    if (*params.ppPropSizeRet != nullptr) {
      **params.ppPropSizeRet = sizeof(ur_platform_handle_t);
    }
    break;
  default:
    return UR_RESULT_SUCCESS;
  }
  return UR_RESULT_SUCCESS;
}

//////////////////////////////////////////////////////////////////////////
context_t::context_t() {
  urGetGlobalProcAddrTable(version, &urDdiTable.Global);
  urGetBindlessImagesExpProcAddrTable(version, &urDdiTable.BindlessImagesExp);
  urGetCommandBufferExpProcAddrTable(version, &urDdiTable.CommandBufferExp);
  urGetContextProcAddrTable(version, &urDdiTable.Context);
  urGetEnqueueProcAddrTable(version, &urDdiTable.Enqueue);
  urGetEnqueueExpProcAddrTable(version, &urDdiTable.EnqueueExp);
  urGetEventProcAddrTable(version, &urDdiTable.Event);
  urGetKernelProcAddrTable(version, &urDdiTable.Kernel);
  urGetMemProcAddrTable(version, &urDdiTable.Mem);
  urGetPhysicalMemProcAddrTable(version, &urDdiTable.PhysicalMem);
  urGetPlatformProcAddrTable(version, &urDdiTable.Platform);
  urGetProgramProcAddrTable(version, &urDdiTable.Program);
  urGetProgramExpProcAddrTable(version, &urDdiTable.ProgramExp);
  urGetQueueProcAddrTable(version, &urDdiTable.Queue);
  urGetSamplerProcAddrTable(version, &urDdiTable.Sampler);
  urGetUSMProcAddrTable(version, &urDdiTable.USM);
  urGetUSMExpProcAddrTable(version, &urDdiTable.USMExp);
  urGetUsmP2PExpProcAddrTable(version, &urDdiTable.UsmP2PExp);
  urGetVirtualMemProcAddrTable(version, &urDdiTable.VirtualMem);
  urGetDeviceProcAddrTable(version, &urDdiTable.Device);

  mock::getCallbacks().set_replace_callback("urPlatformGetApiVersion",
                                            &mock_urPlatformGetApiVersion);
  // Set the default info stuff as before overrides, this way any application
  // passing in an override for them in any slot will take precedence.
  mock::getCallbacks().set_before_callback("urPlatformGetInfo",
                                           &mock_urPlatformGetInfo);
  mock::getCallbacks().set_before_callback("urDeviceGetInfo",
                                           &mock_urDeviceGetInfo);
}
} // namespace driver
