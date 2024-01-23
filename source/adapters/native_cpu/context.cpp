//===--------- context.cpp - Native CPU Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <tuple>

#include "ur/ur.hpp"
#include "ur_api.h"

#include "common.hpp"
#include "context.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urContextCreate(
    [[maybe_unused]] uint32_t DeviceCount, const ur_device_handle_t *phDevices,
    const ur_context_properties_t *pProperties,
    ur_context_handle_t *phContext) {
  std::ignore = pProperties;
  assert(DeviceCount == 1);

  // TODO: Proper error checking.
  auto ctx = new ur_context_handle_t_(*phDevices);
  *phContext = ctx;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextRetain(ur_context_handle_t hContext) {
  hContext->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextRelease(ur_context_handle_t hContext) {
  decrementOrDelete(hContext);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextGetInfo(ur_context_handle_t hContext, ur_context_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper returnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_CONTEXT_INFO_NUM_DEVICES:
    return returnValue(1);
  case UR_CONTEXT_INFO_DEVICES:
    return returnValue(hContext->_device);
  case UR_CONTEXT_INFO_REFERENCE_COUNT:
    return returnValue(nullptr);
  case UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT:
    return returnValue(true);
  case UR_CONTEXT_INFO_USM_FILL2D_SUPPORT:
    // case UR_CONTEXT_INFO_USM_MEMSET2D_SUPPORT:
    // 2D USM operations currently not supported.
    return returnValue(false);
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  }
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urContextGetNativeHandle(
    ur_context_handle_t hContext, ur_native_handle_t *phNativeContext) {
  std::ignore = hContext;
  std::ignore = phNativeContext;
  DIE_NO_IMPLEMENTATION
}

UR_APIEXPORT ur_result_t UR_APICALL urContextCreateWithNativeHandle(
    ur_native_handle_t hNativeContext, uint32_t numDevices,
    const ur_device_handle_t *phDevices,
    const ur_context_native_properties_t *pProperties,
    ur_context_handle_t *phContext) {
  std::ignore = hNativeContext;
  std::ignore = numDevices;
  std::ignore = phDevices;
  std::ignore = pProperties;
  std::ignore = phContext;

  DIE_NO_IMPLEMENTATION
}

UR_APIEXPORT ur_result_t UR_APICALL urContextSetExtendedDeleter(
    ur_context_handle_t hContext, ur_context_extended_deleter_t pfnDeleter,
    void *pUserData) {
  std::ignore = hContext;
  std::ignore = pfnDeleter;
  std::ignore = pUserData;

  DIE_NO_IMPLEMENTATION
}
