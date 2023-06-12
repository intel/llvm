//===--------- context.cpp - CUDA Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "context.hpp"

#include <cassert>

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
  std::ignore = DeviceCount;
  std::ignore = pProperties;
  UR_ASSERT(phDevices, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(phContext, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  assert(DeviceCount == 1);
  ur_result_t RetErr = UR_RESULT_SUCCESS;

  std::unique_ptr<ur_context_handle_t_> ContextPtr{nullptr};
  try {
    ContextPtr = std::unique_ptr<ur_context_handle_t_>(
        new ur_context_handle_t_{*phDevices});
    *phContext = ContextPtr.release();
  } catch (ur_result_t Err) {
    RetErr = Err;
  } catch (...) {
    RetErr = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
  return RetErr;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextGetInfo(
    ur_context_handle_t hContext, ur_context_info_t ContextInfoType,
    size_t propSize, void *pContextInfo, size_t *pPropSizeRet) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  UrReturnHelper ReturnValue(propSize, pContextInfo, pPropSizeRet);

  switch (uint32_t{ContextInfoType}) {
  case UR_CONTEXT_INFO_NUM_DEVICES:
    return ReturnValue(1);
  case UR_CONTEXT_INFO_DEVICES:
    return ReturnValue(hContext->getDevice());
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
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&Major,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                             hContext->getDevice()->get()) == CUDA_SUCCESS);
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
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  if (hContext->decrementReferenceCount() > 0) {
    return UR_RESULT_SUCCESS;
  }
  hContext->invokeExtendedDeleters();

  std::unique_ptr<ur_context_handle_t_> Context{hContext};

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextRetain(ur_context_handle_t hContext) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  assert(hContext->getReferenceCount() > 0);

  hContext->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextGetNativeHandle(
    ur_context_handle_t hContext, ur_native_handle_t *phNativeContext) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phNativeContext, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  *phNativeContext = reinterpret_cast<ur_native_handle_t>(hContext->get());
  return UR_RESULT_SUCCESS;
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

  return UR_RESULT_ERROR_INVALID_OPERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextSetExtendedDeleter(
    ur_context_handle_t hContext, ur_context_extended_deleter_t pfnDeleter,
    void *pUserData) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pfnDeleter, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  hContext->setExtendedDeleter(pfnDeleter, pUserData);
  return UR_RESULT_SUCCESS;
}
