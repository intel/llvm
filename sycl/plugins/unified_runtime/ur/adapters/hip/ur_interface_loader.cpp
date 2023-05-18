//===--------- ur_interface_loader.cpp - Unified Runtime  ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <ur_api.h>
#include <ur_ddi.h>

namespace {

// TODO - this is a duplicate of what is in the L0 plugin
// We should move this to somewhere common
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
} // namespace

#if defined(__cplusplus)
extern "C" {
#endif

UR_DLLEXPORT ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    ur_api_version_t version, ur_platform_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnGet = urPlatformGet;
  pDdiTable->pfnGetApiVersion = urPlatformGetApiVersion;
  pDdiTable->pfnGetInfo = urPlatformGetInfo;
  pDdiTable->pfnGetNativeHandle = nullptr;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetContextProcAddrTable(
    ur_api_version_t version, ur_context_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreate = urContextCreate;
  pDdiTable->pfnCreateWithNativeHandle = urContextCreateWithNativeHandle;
  pDdiTable->pfnGetInfo = urContextGetInfo;
  pDdiTable->pfnGetNativeHandle = urContextGetNativeHandle;
  pDdiTable->pfnRelease = urContextRelease;
  pDdiTable->pfnRetain = urContextRetain;
  pDdiTable->pfnSetExtendedDeleter = urContextSetExtendedDeleter;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetEventProcAddrTable(
    ur_api_version_t version, ur_event_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreateWithNativeHandle = urEventCreateWithNativeHandle;
  pDdiTable->pfnGetInfo = urEventGetInfo;
  pDdiTable->pfnGetNativeHandle = urEventGetNativeHandle;
  pDdiTable->pfnGetProfilingInfo = urEventGetProfilingInfo;
  pDdiTable->pfnRelease = urEventRelease;
  pDdiTable->pfnRetain = urEventRetain;
  pDdiTable->pfnSetCallback = urEventSetCallback;
  pDdiTable->pfnWait = urEventWait;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetProgramProcAddrTable(
    ur_api_version_t version, ur_program_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnBuild = urProgramBuild;
  pDdiTable->pfnCompile = urProgramCompile;
  pDdiTable->pfnCreateWithBinary = urProgramCreateWithBinary;
  pDdiTable->pfnCreateWithIL = urProgramCreateWithIL;
  pDdiTable->pfnCreateWithNativeHandle = urProgramCreateWithNativeHandle;
  pDdiTable->pfnGetBuildInfo = urProgramGetBuildInfo;
  pDdiTable->pfnGetFunctionPointer = urProgramGetFunctionPointer;
  pDdiTable->pfnGetInfo = urProgramGetInfo;
  pDdiTable->pfnGetNativeHandle = urProgramGetNativeHandle;
  pDdiTable->pfnLink = urProgramLink;
  pDdiTable->pfnRelease = urProgramRelease;
  pDdiTable->pfnRetain = urProgramRetain;
  pDdiTable->pfnSetSpecializationConstants =
      urProgramSetSpecializationConstants;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetKernelProcAddrTable(
    ur_api_version_t version, ur_kernel_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreate = urKernelCreate;
  pDdiTable->pfnCreateWithNativeHandle = urKernelCreateWithNativeHandle;
  pDdiTable->pfnGetGroupInfo = urKernelGetGroupInfo;
  pDdiTable->pfnGetInfo = urKernelGetInfo;
  pDdiTable->pfnGetNativeHandle = urKernelGetNativeHandle;
  pDdiTable->pfnGetSubGroupInfo = urKernelGetSubGroupInfo;
  pDdiTable->pfnRelease = urKernelRelease;
  pDdiTable->pfnRetain = urKernelRetain;
  pDdiTable->pfnSetArgLocal = nullptr;
  pDdiTable->pfnSetArgMemObj = nullptr;
  pDdiTable->pfnSetArgPointer = urKernelSetArgPointer;
  pDdiTable->pfnSetArgSampler = nullptr;
  pDdiTable->pfnSetArgValue = urKernelSetArgValue;
  pDdiTable->pfnSetExecInfo = urKernelSetExecInfo;
  pDdiTable->pfnSetSpecializationConstants = nullptr;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    ur_api_version_t version, ur_sampler_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreate = urSamplerCreate;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnGetInfo = urSamplerGetInfo;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnRelease = urSamplerRelease;
  pDdiTable->pfnRetain = urSamplerRetain;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL
urGetMemProcAddrTable(ur_api_version_t version, ur_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnBufferCreate = urMemBufferCreate;
  pDdiTable->pfnBufferPartition = urMemBufferPartition;
  pDdiTable->pfnBufferCreateWithNativeHandle =
      urMemBufferCreateWithNativeHandle;
  pDdiTable->pfnImageCreateWithNativeHandle = urMemImageCreateWithNativeHandle;
  pDdiTable->pfnGetInfo = urMemGetInfo;
  pDdiTable->pfnGetNativeHandle = urMemGetNativeHandle;
  pDdiTable->pfnImageCreate = urMemImageCreate;
  pDdiTable->pfnImageGetInfo = urMemImageGetInfo;
  pDdiTable->pfnRelease = urMemRelease;
  pDdiTable->pfnRetain = urMemRetain;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    ur_api_version_t version, ur_enqueue_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnDeviceGlobalVariableRead = nullptr;
  pDdiTable->pfnDeviceGlobalVariableWrite = nullptr;
  pDdiTable->pfnEventsWait = nullptr;
  pDdiTable->pfnEventsWaitWithBarrier = nullptr;
  pDdiTable->pfnKernelLaunch = nullptr;
  pDdiTable->pfnMemBufferCopy = nullptr;
  pDdiTable->pfnMemBufferCopyRect = nullptr;
  pDdiTable->pfnMemBufferFill = nullptr;
  pDdiTable->pfnMemBufferMap = nullptr;
  pDdiTable->pfnMemBufferRead = nullptr;
  pDdiTable->pfnMemBufferReadRect = nullptr;
  pDdiTable->pfnMemBufferWrite = nullptr;
  pDdiTable->pfnMemBufferWriteRect = nullptr;
  pDdiTable->pfnMemImageCopy = nullptr;
  pDdiTable->pfnMemImageRead = nullptr;
  pDdiTable->pfnMemImageWrite = nullptr;
  pDdiTable->pfnMemUnmap = nullptr;
  pDdiTable->pfnUSMFill2D = nullptr;
  pDdiTable->pfnUSMFill = nullptr;
  pDdiTable->pfnUSMAdvise = nullptr;
  pDdiTable->pfnUSMMemcpy2D = nullptr;
  pDdiTable->pfnUSMMemcpy = nullptr;
  pDdiTable->pfnUSMPrefetch = nullptr;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetGlobalProcAddrTable(
    ur_api_version_t version, ur_global_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnGetLastResult = nullptr;
  pDdiTable->pfnInit = urInit;
  pDdiTable->pfnTearDown = urTearDown;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetQueueProcAddrTable(
    ur_api_version_t version, ur_queue_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreate = urQueueCreate;
  pDdiTable->pfnCreateWithNativeHandle = urQueueCreateWithNativeHandle;
  pDdiTable->pfnFinish = urQueueFinish;
  pDdiTable->pfnFlush = urQueueFlush;
  pDdiTable->pfnGetInfo = urQueueGetInfo;
  pDdiTable->pfnGetNativeHandle = urQueueGetNativeHandle;
  pDdiTable->pfnRelease = urQueueRelease;
  pDdiTable->pfnRetain = urQueueRetain;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL
urGetUSMProcAddrTable(ur_api_version_t version, ur_usm_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnDeviceAlloc = urUSMDeviceAlloc;
  pDdiTable->pfnFree = urUSMFree;
  pDdiTable->pfnGetMemAllocInfo = urUSMGetMemAllocInfo;
  pDdiTable->pfnHostAlloc = urUSMHostAlloc;
  pDdiTable->pfnPoolCreate = nullptr;
  pDdiTable->pfnPoolDestroy = nullptr;
  pDdiTable->pfnPoolDestroy = nullptr;
  pDdiTable->pfnSharedAlloc = urUSMSharedAlloc;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    ur_api_version_t version, ur_device_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreateWithNativeHandle = urDeviceCreateWithNativeHandle;
  pDdiTable->pfnGet = urDeviceGet;
  pDdiTable->pfnGetGlobalTimestamps = urDeviceGetGlobalTimestamps;
  pDdiTable->pfnGetInfo = urDeviceGetInfo;
  pDdiTable->pfnGetNativeHandle = urDeviceGetNativeHandle;
  pDdiTable->pfnPartition = urDevicePartition;
  pDdiTable->pfnRelease = urDeviceRelease;
  pDdiTable->pfnRetain = urDeviceRetain;
  pDdiTable->pfnSelectBinary = urDeviceSelectBinary;
  return UR_RESULT_SUCCESS;
}

#if defined(__cplusplus)
} // extern "C"
#endif
