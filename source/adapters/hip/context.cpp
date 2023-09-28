//===--------- context.cpp - HIP Adapter ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "context.hpp"
#include "usm.hpp"

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

/// Create a UR HIP context.
///
/// By default creates a scoped context and keeps the last active HIP context
/// on top of the HIP context stack.
///
UR_APIEXPORT ur_result_t UR_APICALL urContextCreate(
    uint32_t DeviceCount, const ur_device_handle_t *phDevices,
    const ur_context_properties_t *, ur_context_handle_t *phContext) {
  std::ignore = DeviceCount;
  assert(DeviceCount == 1);
  ur_result_t RetErr = UR_RESULT_SUCCESS;

  std::unique_ptr<ur_context_handle_t_> ContextPtr{nullptr};
  try {
    // Create a scoped context.
    ContextPtr = std::unique_ptr<ur_context_handle_t_>(
        new ur_context_handle_t_{*phDevices});

    static std::once_flag InitFlag;
    std::call_once(
        InitFlag,
        [](ur_result_t &) {
          // Use default stream to record base event counter
          UR_CHECK_ERROR(hipEventCreateWithFlags(&ur_platform_handle_t_::EvBase,
                                                 hipEventDefault));
          UR_CHECK_ERROR(hipEventRecord(ur_platform_handle_t_::EvBase, 0));
        },
        RetErr);

    *phContext = ContextPtr.release();
  } catch (ur_result_t Err) {
    RetErr = Err;
  } catch (...) {
    RetErr = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
  return RetErr;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextGetInfo(ur_context_handle_t hContext, ur_context_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (uint32_t{propName}) {
  case UR_CONTEXT_INFO_NUM_DEVICES:
    return ReturnValue(1);
  case UR_CONTEXT_INFO_DEVICES:
    return ReturnValue(hContext->getDevice());
  case UR_CONTEXT_INFO_REFERENCE_COUNT:
    return ReturnValue(hContext->getReferenceCount());
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    // These queries should be dealt with in context_impl.cpp by calling the
    // queries of each device separately and building the intersection set.
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
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
  if (hContext->decrementReferenceCount() == 0) {
    delete hContext;
  }
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
  *phNativeContext = reinterpret_cast<ur_native_handle_t>(
      hContext->getDevice()->getNativeContext());
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextCreateWithNativeHandle(
    ur_native_handle_t, uint32_t, const ur_device_handle_t *,
    const ur_context_native_properties_t *, ur_context_handle_t *) {
  return UR_RESULT_ERROR_INVALID_OPERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextSetExtendedDeleter(
    ur_context_handle_t hContext, ur_context_extended_deleter_t pfnDeleter,
    void *pUserData) {
  hContext->setExtendedDeleter(pfnDeleter, pUserData);
  return UR_RESULT_SUCCESS;
}
