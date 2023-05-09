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
UR_APIEXPORT ur_result_t UR_APICALL
urContextCreate(uint32_t DeviceCount, const ur_device_handle_t *phDevices,
                const ur_context_properties_t *pProperties,
                ur_context_handle_t *phContext) {
  UR_ASSERT(phDevices, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(phContext, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  assert(DeviceCount == 1);
  ur_result_t errcode_ret = UR_RESULT_SUCCESS;

  std::unique_ptr<ur_context_handle_t_> urContextPtr{nullptr};
  try {
    hipCtx_t current = nullptr;

    // Create a scoped context.
    hipCtx_t newContext;
    UR_CHECK_ERROR(hipCtxGetCurrent(&current));
    errcode_ret = UR_CHECK_ERROR(
        hipCtxCreate(&newContext, hipDeviceMapHost, phDevices[0]->get()));
    urContextPtr =
        std::unique_ptr<ur_context_handle_t_>(new ur_context_handle_t_{
            ur_context_handle_t_::kind::user_defined, newContext, *phDevices});

    static std::once_flag initFlag;
    std::call_once(
        initFlag,
        [](ur_result_t &err) {
          // Use default stream to record base event counter
          UR_CHECK_ERROR(hipEventCreateWithFlags(
              &ur_platform_handle_t_::evBase_, hipEventDefault));
          UR_CHECK_ERROR(hipEventRecord(ur_platform_handle_t_::evBase_, 0));
        },
        errcode_ret);

    // For non-primary scoped contexts keep the last active on top of the stack
    // as `cuCtxCreate` replaces it implicitly otherwise.
    // Primary contexts are kept on top of the stack, so the previous context
    // is not queried and therefore not recovered.
    if (current != nullptr) {
      UR_CHECK_ERROR(hipCtxSetCurrent(current));
    }

    *phContext = urContextPtr.release();
  } catch (ur_result_t err) {
    errcode_ret = err;
  } catch (...) {
    errcode_ret = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
  return errcode_ret;
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
    return ReturnValue(hContext->get_device());
  case UR_CONTEXT_INFO_REFERENCE_COUNT:
    return ReturnValue(hContext->get_reference_count());
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    // These queries should be dealt with in context_impl.cpp by calling the
    // queries of each device separately and building the intersection set.
    setErrorMessage("These queries should have never come here.",
                    UR_RESULT_ERROR_INVALID_ARGUMENT);
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
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

UR_APIEXPORT ur_result_t UR_APICALL urContextRelease(ur_context_handle_t ctxt) {
  UR_ASSERT(ctxt, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  if (ctxt->decrement_reference_count() > 0) {
    return UR_RESULT_SUCCESS;
  }
  ctxt->invoke_extended_deleters();

  std::unique_ptr<ur_context_handle_t_> context{ctxt};

  if (!ctxt->is_primary()) {
    hipCtx_t hipCtxt = ctxt->get();
    // hipCtxSynchronize is not supported for AMD platform so we can just
    // destroy the context, for NVIDIA make sure it's synchronized.
#if defined(__HIP_PLATFORM_NVIDIA__)
    hipCtx_t current = nullptr;
    UR_CHECK_ERROR(hipCtxGetCurrent(&current));
    if (hipCtxt != current) {
      UR_CHECK_ERROR(hipCtxPushCurrent(hipCtxt));
    }
    UR_CHECK_ERROR(hipCtxSynchronize());
    UR_CHECK_ERROR(hipCtxGetCurrent(&current));
    if (hipCtxt == current) {
      UR_CHECK_ERROR(hipCtxPopCurrent(&current));
    }
#endif
    return UR_CHECK_ERROR(hipCtxDestroy(hipCtxt));
  } else {
    // Primary context is not destroyed, but released
    hipDevice_t hipDev = ctxt->get_device()->get();
    hipCtx_t current;
    UR_CHECK_ERROR(hipCtxPopCurrent(&current));
    return UR_CHECK_ERROR(hipDevicePrimaryCtxRelease(hipDev));
  }

  hipCtx_t hipCtxt = ctxt->get();
  return UR_CHECK_ERROR(hipCtxDestroy(hipCtxt));
}

UR_APIEXPORT ur_result_t UR_APICALL urContextRetain(ur_context_handle_t ctxt) {
  UR_ASSERT(ctxt, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  assert(ctxt->get_reference_count() > 0);

  ctxt->increment_reference_count();
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
  (void)hNativeContext;
  (void)phContext;

  // TODO(ur): Needed for the conformance test to pass, but it may be valid
  // to have a null CUDA context
  UR_ASSERT(hNativeContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  return UR_RESULT_ERROR_INVALID_OPERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urContextSetExtendedDeleter(
    ur_context_handle_t hContext, ur_context_extended_deleter_t pfnDeleter,
    void *pUserData) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pfnDeleter, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  hContext->set_extended_deleter(pfnDeleter, pUserData);
  return UR_RESULT_SUCCESS;
}
