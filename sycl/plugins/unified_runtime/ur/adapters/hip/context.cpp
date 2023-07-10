//===--------- context.cpp - HIP Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "context.hpp"

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
    hipCtx_t Current = nullptr;

    // Create a scoped context.
    hipCtx_t NewContext;
    UR_CHECK_ERROR(hipCtxGetCurrent(&Current));
    RetErr = UR_CHECK_ERROR(
        hipCtxCreate(&NewContext, hipDeviceMapHost, phDevices[0]->get()));
    ContextPtr = std::unique_ptr<ur_context_handle_t_>(new ur_context_handle_t_{
        ur_context_handle_t_::kind::UserDefined, NewContext, *phDevices});

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

    // For non-primary scoped contexts keep the last active on top of the stack
    // as `hipCtxCreate` replaces it implicitly otherwise.
    // Primary contexts are kept on top of the stack, so the previous context
    // is not queried and therefore not recovered.
    if (Current != nullptr) {
      UR_CHECK_ERROR(hipCtxSetCurrent(Current));
    }

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
  if (hContext->decrementReferenceCount() > 0) {
    return UR_RESULT_SUCCESS;
  }
  hContext->invokeExtendedDeleters();

  std::unique_ptr<ur_context_handle_t_> context{hContext};

  if (!hContext->isPrimary()) {
    hipCtx_t HIPCtxt = hContext->get();
    // hipCtxSynchronize is not supported for AMD platform so we can just
    // destroy the context, for NVIDIA make sure it's synchronized.
#if defined(__HIP_PLATFORM_NVIDIA__)
    hipCtx_t Current = nullptr;
    UR_CHECK_ERROR(hipCtxGetCurrent(&Current));
    if (HIPCtxt != Current) {
      UR_CHECK_ERROR(hipCtxPushCurrent(HIPCtxt));
    }
    UR_CHECK_ERROR(hipCtxSynchronize());
    UR_CHECK_ERROR(hipCtxGetCurrent(&Current));
    if (HIPCtxt == Current) {
      UR_CHECK_ERROR(hipCtxPopCurrent(&Current));
    }
#endif
    return UR_CHECK_ERROR(hipCtxDestroy(HIPCtxt));
  } else {
    // Primary context is not destroyed, but released
    hipDevice_t HIPDev = hContext->getDevice()->get();
    hipCtx_t Current;
    UR_CHECK_ERROR(hipCtxPopCurrent(&Current));
    return UR_CHECK_ERROR(hipDevicePrimaryCtxRelease(HIPDev));
  }

  hipCtx_t HIPCtxt = hContext->get();
  return UR_CHECK_ERROR(hipCtxDestroy(HIPCtxt));
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextRetain(ur_context_handle_t hContext) {
  assert(hContext->getReferenceCount() > 0);

  hContext->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextGetNativeHandle(
    ur_context_handle_t hContext, ur_native_handle_t *phNativeContext) {
  *phNativeContext = reinterpret_cast<ur_native_handle_t>(hContext->get());
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
