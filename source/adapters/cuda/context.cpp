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
///
/// By default creates a scoped context and keeps the last active CUDA context
/// on top of the CUDA context stack.
/// With the __SYCL_PI_CONTEXT_PROPERTIES_CUDA_PRIMARY key/id and a value of
/// PI_TRUE creates a primary CUDA context and activates it on the CUDA context
/// stack.
///
UR_APIEXPORT ur_result_t UR_APICALL
urContextCreate(uint32_t DeviceCount, const ur_device_handle_t *phDevices,
                const ur_context_properties_t *pProperties,
                ur_context_handle_t *phContext) {
  std::ignore = pProperties;

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
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: {
    uint32_t Capabilities = UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
                            UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
                            UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE |
                            UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL;
    return ReturnValue(Capabilities);
  }
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES: {
    int Major = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        hContext->getDevices()[0]->get()));
    uint32_t Capabilities =
        (Major >= 7) ? UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM
                     : UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP |
                           UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE;
    return ReturnValue(Capabilities);
  }
  case UR_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
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

  std::unique_ptr<ur_context_handle_t_> Context{hContext};

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
