//===--------- ur_loader_interface.cpp - Level Zero Adapter----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <ur_api.h>
#include <ur_ddi.h>

ur_result_t validateProcInputs(ur_api_version_t version, void *pDdiTable) {
  if (nullptr == pDdiTable) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }
  // Pre 1.0 we enforce loader and adapter must have same version.
  // Post 1.0 only major version match should be required.
  if (version != UR_API_VERSION_CURRENT) {
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetGlobalProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_global_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }

  pDdiTable->pfnInit = urInit;
  pDdiTable->pfnGetLastResult = nullptr;
  pDdiTable->pfnTearDown = urTearDown;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetContextProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_context_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }

  pDdiTable->pfnCreate = nullptr;
  pDdiTable->pfnRetain = nullptr;
  pDdiTable->pfnRelease = nullptr;
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnSetExtendedDeleter = nullptr;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_enqueue_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }

  pDdiTable->pfnKernelLaunch = nullptr;
  pDdiTable->pfnEventsWait = nullptr;
  pDdiTable->pfnEventsWaitWithBarrier = nullptr;
  pDdiTable->pfnMemBufferRead = nullptr;
  pDdiTable->pfnMemBufferWrite = nullptr;
  pDdiTable->pfnMemBufferReadRect = nullptr;
  pDdiTable->pfnMemBufferWriteRect = nullptr;
  pDdiTable->pfnMemBufferCopy = nullptr;
  pDdiTable->pfnMemBufferCopyRect = nullptr;
  pDdiTable->pfnMemBufferFill = nullptr;
  pDdiTable->pfnMemImageRead = nullptr;
  pDdiTable->pfnMemImageWrite = nullptr;
  pDdiTable->pfnMemImageCopy = nullptr;
  pDdiTable->pfnMemBufferMap = nullptr;
  pDdiTable->pfnMemUnmap = nullptr;
  pDdiTable->pfnUSMMemcpy = nullptr;
  pDdiTable->pfnUSMPrefetch = nullptr;
  pDdiTable->pfnUSMAdvise = nullptr;
  pDdiTable->pfnUSMFill2D = nullptr;
  pDdiTable->pfnUSMMemcpy2D = nullptr;
  pDdiTable->pfnDeviceGlobalVariableWrite = nullptr;
  pDdiTable->pfnDeviceGlobalVariableRead = nullptr;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetEventProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_event_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnGetProfilingInfo = nullptr;
  pDdiTable->pfnWait = nullptr;
  pDdiTable->pfnRetain = nullptr;
  pDdiTable->pfnRelease = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnSetCallback = nullptr;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetKernelProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_kernel_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnCreate = nullptr;
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnGetGroupInfo = nullptr;
  pDdiTable->pfnGetSubGroupInfo = nullptr;
  pDdiTable->pfnRetain = nullptr;
  pDdiTable->pfnRelease = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnSetArgValue = nullptr;
  pDdiTable->pfnSetArgLocal = nullptr;
  pDdiTable->pfnSetArgPointer = nullptr;
  pDdiTable->pfnSetExecInfo = nullptr;
  pDdiTable->pfnSetArgSampler = nullptr;
  pDdiTable->pfnSetArgMemObj = nullptr;
  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetMemProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_mem_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnImageCreate = nullptr;
  pDdiTable->pfnBufferCreate = nullptr;
  pDdiTable->pfnRetain = nullptr;
  pDdiTable->pfnRelease = nullptr;
  pDdiTable->pfnBufferPartition = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnImageGetInfo = nullptr;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_platform_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnGet = urPlatformGet;
  pDdiTable->pfnGetInfo = urPlatformGetInfo;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnGetApiVersion = nullptr;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetProgramProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_program_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {

  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnCreateWithBinary = nullptr;
  pDdiTable->pfnRetain = nullptr;
  pDdiTable->pfnRelease = nullptr;
  pDdiTable->pfnGetFunctionPointer = nullptr;
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnGetBuildInfo = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetQueueProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_queue_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }

  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnCreate = nullptr;
  pDdiTable->pfnRetain = nullptr;
  pDdiTable->pfnRelease = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnFinish = nullptr;
  pDdiTable->pfnFlush = nullptr;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_sampler_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnCreate = nullptr;
  pDdiTable->pfnRetain = nullptr;
  pDdiTable->pfnRelease = nullptr;
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetUSMProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_usm_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnHostAlloc = nullptr;
  pDdiTable->pfnDeviceAlloc = nullptr;
  pDdiTable->pfnSharedAlloc = nullptr;
  pDdiTable->pfnFree = nullptr;
  pDdiTable->pfnGetMemAllocInfo = nullptr;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_device_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnGet = urDeviceGet;
  pDdiTable->pfnGetInfo = urDeviceGetInfo;
  pDdiTable->pfnRetain = urDeviceRetain;
  pDdiTable->pfnRelease = urDeviceRelease;
  pDdiTable->pfnPartition = urDevicePartition;
  pDdiTable->pfnSelectBinary = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnGetGlobalTimestamps = nullptr;

  return retVal;
}
