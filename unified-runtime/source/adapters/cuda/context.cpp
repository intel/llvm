//===--------- context.cpp - CUDA Adapter ---------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "context.hpp"
#include "platform.hpp"
#include "usm.hpp"

#include <cassert>

void ur_context_handle_t_::addPool(ur_usm_pool_handle_t Pool) {
  std::lock_guard<std::mutex> Lock(Mutex);
  PoolHandles.insert(Pool);
}

void ur_context_handle_t_::removePool(ur_usm_pool_handle_t Pool) {
  std::lock_guard<std::mutex> Lock(Mutex);
  PoolHandles.erase(Pool);
}

ur_usm_pool_handle_t
ur_context_handle_t_::getOwningURPool(umf_memory_pool_t *UMFPool) {
  std::lock_guard<std::mutex> Lock(Mutex);
  for (auto &Pool : PoolHandles) {
    if (Pool->hasUMFPool(UMFPool)) {
      return Pool;
    }
  }
  return nullptr;
}

/// Create a UR CUDA context.
UR_APIEXPORT ur_result_t UR_APICALL
urContextCreate(uint32_t DeviceCount, const ur_device_handle_t *phDevices,
                const ur_context_properties_t * /*pProperties*/,
                ur_context_handle_t *phContext) {

  std::unique_ptr<ur_context_handle_t_> ContextPtr{nullptr};
  try {
    ContextPtr = std::unique_ptr<ur_context_handle_t_>(
        new ur_context_handle_t_{phDevices, DeviceCount});
    *phContext = ContextPtr.release();
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextGetInfo(
    ur_context_handle_t hContext, ur_context_info_t ContextInfoType,
    size_t propSize, void *pContextInfo, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pContextInfo, pPropSizeRet);

  switch (static_cast<uint32_t>(ContextInfoType)) {
  case UR_CONTEXT_INFO_NUM_DEVICES:
    return ReturnValue(static_cast<uint32_t>(hContext->getDevices().size()));
  case UR_CONTEXT_INFO_DEVICES:
    return ReturnValue(hContext->getDevices().data(),
                       hContext->getDevices().size());
  case UR_CONTEXT_INFO_REFERENCE_COUNT:
    return ReturnValue(hContext->getReferenceCount());
  case UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT:
    // 2D USM memcpy is supported.
    return ReturnValue(true);
  case UR_CONTEXT_INFO_USM_FILL2D_SUPPORT:
    // 2D USM operations currently not supported.
    return ReturnValue(false);

  default:
    break;
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextRelease(ur_context_handle_t hContext) {
  if (hContext->decrementReferenceCount() > 0) {
    return UR_RESULT_SUCCESS;
  }
  hContext->invokeExtendedDeleters();
  delete hContext;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextRetain(ur_context_handle_t hContext) {
  assert(hContext->getReferenceCount() > 0);

  hContext->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextGetNativeHandle(
    ur_context_handle_t hContext, ur_native_handle_t *phNativeContext) {
  // FIXME: this entry point has been deprecated in the SYCL RT and should be
  // changed to unsupoorted once deprecation period has elapsed.
  *phNativeContext = reinterpret_cast<ur_native_handle_t>(
      hContext->getDevices()[0]->getNativeContext());
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextCreateWithNativeHandle(
    [[maybe_unused]] ur_native_handle_t hNativeContext,
    [[maybe_unused]] ur_adapter_handle_t hAdapter,
    [[maybe_unused]] uint32_t numDevices,
    [[maybe_unused]] const ur_device_handle_t *phDevices,
    [[maybe_unused]] const ur_context_native_properties_t *pProperties,
    [[maybe_unused]] ur_context_handle_t *phContext) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextSetExtendedDeleter(
    ur_context_handle_t hContext, ur_context_extended_deleter_t pfnDeleter,
    void *pUserData) {
  hContext->setExtendedDeleter(pfnDeleter, pUserData);
  return UR_RESULT_SUCCESS;
}
