//===----------- context.cpp - LLVM Offload Adapter  ----------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "context.hpp"
#include <unified-runtime/ur_api.h>
#include <vector>

UR_APIEXPORT ur_result_t UR_APICALL urContextCreate(
    uint32_t DeviceCount, const ur_device_handle_t *phDevices,
    const ur_context_properties_t *, ur_context_handle_t *phContext) {
  if (DeviceCount > 1) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  std::vector<ol_device_handle_t> OffloadDevices;
  OffloadDevices.reserve(DeviceCount);
  for (uint32_t I = 0; I < DeviceCount; ++I) {
    OffloadDevices.push_back(phDevices[I]->OffloadDevice);
  }

  ol_context_handle_t OffloadContext;
  OL_RETURN_ON_ERR(
      olCreateContext(DeviceCount, OffloadDevices.data(), &OffloadContext));

  auto Ctx = new ur_context_handle_t_(*phDevices, OffloadContext);
  *phContext = Ctx;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextGetInfo(ur_context_handle_t hContext, ur_context_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_CONTEXT_INFO_NUM_DEVICES:
    return ReturnValue(uint32_t{1});
  case UR_CONTEXT_INFO_DEVICES:
    return ReturnValue(&hContext->Device, 1);
  case UR_CONTEXT_INFO_REFERENCE_COUNT:
    return ReturnValue(hContext->RefCount.load());
  case UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT:
  case UR_CONTEXT_INFO_USM_FILL2D_SUPPORT:
    return ReturnValue(false);
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextRetain(ur_context_handle_t hContext) {
  hContext->RefCount++;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextRelease(ur_context_handle_t hContext) {
  if (--hContext->RefCount == 0) {
    OL_RETURN_ON_ERR(olDestroyContext(hContext->OffloadContext));
    delete hContext;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextGetNativeHandle(ur_context_handle_t, ur_native_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextCreateWithNativeHandle(
    ur_native_handle_t, ur_adapter_handle_t, uint32_t,
    const ur_device_handle_t *, const ur_context_native_properties_t *,
    ur_context_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextSetExtendedDeleter(
    ur_context_handle_t, ur_context_extended_deleter_t, void *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
