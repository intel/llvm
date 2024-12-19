//===--------- ur_interface_loader.cpp - Level Zero Adapter ------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <ur_api.h>
#include <ur_ddi.h>

#include "ur_interface_loader.hpp"

static ur_result_t validateProcInputs(ur_api_version_t version,
                                      void *pDdiTable) {
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

#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO
namespace ur::level_zero {
#elif defined(__cplusplus)
extern "C" {
#endif

UR_APIEXPORT ur_result_t UR_APICALL urGetGlobalProcAddrTable(
    ur_api_version_t version, ur_global_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnAdapterGet = ur::level_zero::urAdapterGet;
  pDdiTable->pfnAdapterRelease = ur::level_zero::urAdapterRelease;
  pDdiTable->pfnAdapterRetain = ur::level_zero::urAdapterRetain;
  pDdiTable->pfnAdapterGetLastError = ur::level_zero::urAdapterGetLastError;
  pDdiTable->pfnAdapterGetInfo = ur::level_zero::urAdapterGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetBindlessImagesExpProcAddrTable(
    ur_api_version_t version, ur_bindless_images_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnUnsampledImageHandleDestroyExp =
      ur::level_zero::urBindlessImagesUnsampledImageHandleDestroyExp;
  pDdiTable->pfnSampledImageHandleDestroyExp =
      ur::level_zero::urBindlessImagesSampledImageHandleDestroyExp;
  pDdiTable->pfnImageAllocateExp =
      ur::level_zero::urBindlessImagesImageAllocateExp;
  pDdiTable->pfnImageFreeExp = ur::level_zero::urBindlessImagesImageFreeExp;
  pDdiTable->pfnUnsampledImageCreateExp =
      ur::level_zero::urBindlessImagesUnsampledImageCreateExp;
  pDdiTable->pfnSampledImageCreateExp =
      ur::level_zero::urBindlessImagesSampledImageCreateExp;
  pDdiTable->pfnImageCopyExp = ur::level_zero::urBindlessImagesImageCopyExp;
  pDdiTable->pfnImageGetInfoExp =
      ur::level_zero::urBindlessImagesImageGetInfoExp;
  pDdiTable->pfnMipmapGetLevelExp =
      ur::level_zero::urBindlessImagesMipmapGetLevelExp;
  pDdiTable->pfnMipmapFreeExp = ur::level_zero::urBindlessImagesMipmapFreeExp;
  pDdiTable->pfnImportExternalMemoryExp =
      ur::level_zero::urBindlessImagesImportExternalMemoryExp;
  pDdiTable->pfnMapExternalArrayExp =
      ur::level_zero::urBindlessImagesMapExternalArrayExp;
  pDdiTable->pfnMapExternalLinearMemoryExp =
      ur::level_zero::urBindlessImagesMapExternalLinearMemoryExp;
  pDdiTable->pfnReleaseExternalMemoryExp =
      ur::level_zero::urBindlessImagesReleaseExternalMemoryExp;
  pDdiTable->pfnImportExternalSemaphoreExp =
      ur::level_zero::urBindlessImagesImportExternalSemaphoreExp;
  pDdiTable->pfnReleaseExternalSemaphoreExp =
      ur::level_zero::urBindlessImagesReleaseExternalSemaphoreExp;
  pDdiTable->pfnWaitExternalSemaphoreExp =
      ur::level_zero::urBindlessImagesWaitExternalSemaphoreExp;
  pDdiTable->pfnSignalExternalSemaphoreExp =
      ur::level_zero::urBindlessImagesSignalExternalSemaphoreExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetCommandBufferExpProcAddrTable(
    ur_api_version_t version, ur_command_buffer_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateExp = ur::level_zero::urCommandBufferCreateExp;
  pDdiTable->pfnRetainExp = ur::level_zero::urCommandBufferRetainExp;
  pDdiTable->pfnReleaseExp = ur::level_zero::urCommandBufferReleaseExp;
  pDdiTable->pfnFinalizeExp = ur::level_zero::urCommandBufferFinalizeExp;
  pDdiTable->pfnAppendKernelLaunchExp =
      ur::level_zero::urCommandBufferAppendKernelLaunchExp;
  pDdiTable->pfnAppendUSMMemcpyExp =
      ur::level_zero::urCommandBufferAppendUSMMemcpyExp;
  pDdiTable->pfnAppendUSMFillExp =
      ur::level_zero::urCommandBufferAppendUSMFillExp;
  pDdiTable->pfnAppendMemBufferCopyExp =
      ur::level_zero::urCommandBufferAppendMemBufferCopyExp;
  pDdiTable->pfnAppendMemBufferWriteExp =
      ur::level_zero::urCommandBufferAppendMemBufferWriteExp;
  pDdiTable->pfnAppendMemBufferReadExp =
      ur::level_zero::urCommandBufferAppendMemBufferReadExp;
  pDdiTable->pfnAppendMemBufferCopyRectExp =
      ur::level_zero::urCommandBufferAppendMemBufferCopyRectExp;
  pDdiTable->pfnAppendMemBufferWriteRectExp =
      ur::level_zero::urCommandBufferAppendMemBufferWriteRectExp;
  pDdiTable->pfnAppendMemBufferReadRectExp =
      ur::level_zero::urCommandBufferAppendMemBufferReadRectExp;
  pDdiTable->pfnAppendMemBufferFillExp =
      ur::level_zero::urCommandBufferAppendMemBufferFillExp;
  pDdiTable->pfnAppendUSMPrefetchExp =
      ur::level_zero::urCommandBufferAppendUSMPrefetchExp;
  pDdiTable->pfnAppendUSMAdviseExp =
      ur::level_zero::urCommandBufferAppendUSMAdviseExp;
  pDdiTable->pfnEnqueueExp = ur::level_zero::urCommandBufferEnqueueExp;
  pDdiTable->pfnRetainCommandExp =
      ur::level_zero::urCommandBufferRetainCommandExp;
  pDdiTable->pfnReleaseCommandExp =
      ur::level_zero::urCommandBufferReleaseCommandExp;
  pDdiTable->pfnUpdateKernelLaunchExp =
      ur::level_zero::urCommandBufferUpdateKernelLaunchExp;
  pDdiTable->pfnUpdateSignalEventExp =
      ur::level_zero::urCommandBufferUpdateSignalEventExp;
  pDdiTable->pfnUpdateWaitEventsExp =
      ur::level_zero::urCommandBufferUpdateWaitEventsExp;
  pDdiTable->pfnGetInfoExp = ur::level_zero::urCommandBufferGetInfoExp;
  pDdiTable->pfnCommandGetInfoExp =
      ur::level_zero::urCommandBufferCommandGetInfoExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetContextProcAddrTable(
    ur_api_version_t version, ur_context_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::level_zero::urContextCreate;
  pDdiTable->pfnRetain = ur::level_zero::urContextRetain;
  pDdiTable->pfnRelease = ur::level_zero::urContextRelease;
  pDdiTable->pfnGetInfo = ur::level_zero::urContextGetInfo;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::urContextGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::urContextCreateWithNativeHandle;
  pDdiTable->pfnSetExtendedDeleter =
      ur::level_zero::urContextSetExtendedDeleter;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    ur_api_version_t version, ur_enqueue_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnKernelLaunch = ur::level_zero::urEnqueueKernelLaunch;
  pDdiTable->pfnEventsWait = ur::level_zero::urEnqueueEventsWait;
  pDdiTable->pfnEventsWaitWithBarrier =
      ur::level_zero::urEnqueueEventsWaitWithBarrier;
  pDdiTable->pfnMemBufferRead = ur::level_zero::urEnqueueMemBufferRead;
  pDdiTable->pfnMemBufferWrite = ur::level_zero::urEnqueueMemBufferWrite;
  pDdiTable->pfnMemBufferReadRect = ur::level_zero::urEnqueueMemBufferReadRect;
  pDdiTable->pfnMemBufferWriteRect =
      ur::level_zero::urEnqueueMemBufferWriteRect;
  pDdiTable->pfnMemBufferCopy = ur::level_zero::urEnqueueMemBufferCopy;
  pDdiTable->pfnMemBufferCopyRect = ur::level_zero::urEnqueueMemBufferCopyRect;
  pDdiTable->pfnMemBufferFill = ur::level_zero::urEnqueueMemBufferFill;
  pDdiTable->pfnMemImageRead = ur::level_zero::urEnqueueMemImageRead;
  pDdiTable->pfnMemImageWrite = ur::level_zero::urEnqueueMemImageWrite;
  pDdiTable->pfnMemImageCopy = ur::level_zero::urEnqueueMemImageCopy;
  pDdiTable->pfnMemBufferMap = ur::level_zero::urEnqueueMemBufferMap;
  pDdiTable->pfnMemUnmap = ur::level_zero::urEnqueueMemUnmap;
  pDdiTable->pfnUSMFill = ur::level_zero::urEnqueueUSMFill;
  pDdiTable->pfnUSMMemcpy = ur::level_zero::urEnqueueUSMMemcpy;
  pDdiTable->pfnUSMPrefetch = ur::level_zero::urEnqueueUSMPrefetch;
  pDdiTable->pfnUSMAdvise = ur::level_zero::urEnqueueUSMAdvise;
  pDdiTable->pfnUSMFill2D = ur::level_zero::urEnqueueUSMFill2D;
  pDdiTable->pfnUSMMemcpy2D = ur::level_zero::urEnqueueUSMMemcpy2D;
  pDdiTable->pfnDeviceGlobalVariableWrite =
      ur::level_zero::urEnqueueDeviceGlobalVariableWrite;
  pDdiTable->pfnDeviceGlobalVariableRead =
      ur::level_zero::urEnqueueDeviceGlobalVariableRead;
  pDdiTable->pfnReadHostPipe = ur::level_zero::urEnqueueReadHostPipe;
  pDdiTable->pfnWriteHostPipe = ur::level_zero::urEnqueueWriteHostPipe;
  pDdiTable->pfnEventsWaitWithBarrierExt =
      ur::level_zero::urEnqueueEventsWaitWithBarrierExt;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    ur_api_version_t version, ur_enqueue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnKernelLaunchCustomExp =
      ur::level_zero::urEnqueueKernelLaunchCustomExp;
  pDdiTable->pfnCooperativeKernelLaunchExp =
      ur::level_zero::urEnqueueCooperativeKernelLaunchExp;
  pDdiTable->pfnTimestampRecordingExp =
      ur::level_zero::urEnqueueTimestampRecordingExp;
  pDdiTable->pfnNativeCommandExp = ur::level_zero::urEnqueueNativeCommandExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEventProcAddrTable(
    ur_api_version_t version, ur_event_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetInfo = ur::level_zero::urEventGetInfo;
  pDdiTable->pfnGetProfilingInfo = ur::level_zero::urEventGetProfilingInfo;
  pDdiTable->pfnWait = ur::level_zero::urEventWait;
  pDdiTable->pfnRetain = ur::level_zero::urEventRetain;
  pDdiTable->pfnRelease = ur::level_zero::urEventRelease;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::urEventGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::urEventCreateWithNativeHandle;
  pDdiTable->pfnSetCallback = ur::level_zero::urEventSetCallback;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetKernelProcAddrTable(
    ur_api_version_t version, ur_kernel_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::level_zero::urKernelCreate;
  pDdiTable->pfnGetInfo = ur::level_zero::urKernelGetInfo;
  pDdiTable->pfnGetGroupInfo = ur::level_zero::urKernelGetGroupInfo;
  pDdiTable->pfnGetSubGroupInfo = ur::level_zero::urKernelGetSubGroupInfo;
  pDdiTable->pfnRetain = ur::level_zero::urKernelRetain;
  pDdiTable->pfnRelease = ur::level_zero::urKernelRelease;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::urKernelGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::urKernelCreateWithNativeHandle;
  pDdiTable->pfnGetSuggestedLocalWorkSize =
      ur::level_zero::urKernelGetSuggestedLocalWorkSize;
  pDdiTable->pfnSetArgValue = ur::level_zero::urKernelSetArgValue;
  pDdiTable->pfnSetArgLocal = ur::level_zero::urKernelSetArgLocal;
  pDdiTable->pfnSetArgPointer = ur::level_zero::urKernelSetArgPointer;
  pDdiTable->pfnSetExecInfo = ur::level_zero::urKernelSetExecInfo;
  pDdiTable->pfnSetArgSampler = ur::level_zero::urKernelSetArgSampler;
  pDdiTable->pfnSetArgMemObj = ur::level_zero::urKernelSetArgMemObj;
  pDdiTable->pfnSetSpecializationConstants =
      ur::level_zero::urKernelSetSpecializationConstants;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetKernelExpProcAddrTable(
    ur_api_version_t version, ur_kernel_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnSuggestMaxCooperativeGroupCountExp =
      ur::level_zero::urKernelSuggestMaxCooperativeGroupCountExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGetMemProcAddrTable(ur_api_version_t version, ur_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnImageCreate = ur::level_zero::urMemImageCreate;
  pDdiTable->pfnBufferCreate = ur::level_zero::urMemBufferCreate;
  pDdiTable->pfnRetain = ur::level_zero::urMemRetain;
  pDdiTable->pfnRelease = ur::level_zero::urMemRelease;
  pDdiTable->pfnBufferPartition = ur::level_zero::urMemBufferPartition;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::urMemGetNativeHandle;
  pDdiTable->pfnBufferCreateWithNativeHandle =
      ur::level_zero::urMemBufferCreateWithNativeHandle;
  pDdiTable->pfnImageCreateWithNativeHandle =
      ur::level_zero::urMemImageCreateWithNativeHandle;
  pDdiTable->pfnGetInfo = ur::level_zero::urMemGetInfo;
  pDdiTable->pfnImageGetInfo = ur::level_zero::urMemImageGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    ur_api_version_t version, ur_physical_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::level_zero::urPhysicalMemCreate;
  pDdiTable->pfnRetain = ur::level_zero::urPhysicalMemRetain;
  pDdiTable->pfnRelease = ur::level_zero::urPhysicalMemRelease;
  pDdiTable->pfnGetInfo = ur::level_zero::urPhysicalMemGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    ur_api_version_t version, ur_platform_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGet = ur::level_zero::urPlatformGet;
  pDdiTable->pfnGetInfo = ur::level_zero::urPlatformGetInfo;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::urPlatformGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::urPlatformCreateWithNativeHandle;
  pDdiTable->pfnGetApiVersion = ur::level_zero::urPlatformGetApiVersion;
  pDdiTable->pfnGetBackendOption = ur::level_zero::urPlatformGetBackendOption;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetProgramProcAddrTable(
    ur_api_version_t version, ur_program_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateWithIL = ur::level_zero::urProgramCreateWithIL;
  pDdiTable->pfnCreateWithBinary = ur::level_zero::urProgramCreateWithBinary;
  pDdiTable->pfnBuild = ur::level_zero::urProgramBuild;
  pDdiTable->pfnCompile = ur::level_zero::urProgramCompile;
  pDdiTable->pfnLink = ur::level_zero::urProgramLink;
  pDdiTable->pfnRetain = ur::level_zero::urProgramRetain;
  pDdiTable->pfnRelease = ur::level_zero::urProgramRelease;
  pDdiTable->pfnGetFunctionPointer =
      ur::level_zero::urProgramGetFunctionPointer;
  pDdiTable->pfnGetGlobalVariablePointer =
      ur::level_zero::urProgramGetGlobalVariablePointer;
  pDdiTable->pfnGetInfo = ur::level_zero::urProgramGetInfo;
  pDdiTable->pfnGetBuildInfo = ur::level_zero::urProgramGetBuildInfo;
  pDdiTable->pfnSetSpecializationConstants =
      ur::level_zero::urProgramSetSpecializationConstants;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::urProgramGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::urProgramCreateWithNativeHandle;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetProgramExpProcAddrTable(
    ur_api_version_t version, ur_program_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnBuildExp = ur::level_zero::urProgramBuildExp;
  pDdiTable->pfnCompileExp = ur::level_zero::urProgramCompileExp;
  pDdiTable->pfnLinkExp = ur::level_zero::urProgramLinkExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetQueueProcAddrTable(
    ur_api_version_t version, ur_queue_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetInfo = ur::level_zero::urQueueGetInfo;
  pDdiTable->pfnCreate = ur::level_zero::urQueueCreate;
  pDdiTable->pfnRetain = ur::level_zero::urQueueRetain;
  pDdiTable->pfnRelease = ur::level_zero::urQueueRelease;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::urQueueGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::urQueueCreateWithNativeHandle;
  pDdiTable->pfnFinish = ur::level_zero::urQueueFinish;
  pDdiTable->pfnFlush = ur::level_zero::urQueueFlush;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    ur_api_version_t version, ur_sampler_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::level_zero::urSamplerCreate;
  pDdiTable->pfnRetain = ur::level_zero::urSamplerRetain;
  pDdiTable->pfnRelease = ur::level_zero::urSamplerRelease;
  pDdiTable->pfnGetInfo = ur::level_zero::urSamplerGetInfo;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::urSamplerGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::urSamplerCreateWithNativeHandle;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetTensorMapExpProcAddrTable(
    ur_api_version_t version, ur_tensor_map_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnEncodeIm2ColExp = ur::level_zero::urTensorMapEncodeIm2ColExp;
  pDdiTable->pfnEncodeTiledExp = ur::level_zero::urTensorMapEncodeTiledExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGetUSMProcAddrTable(ur_api_version_t version, ur_usm_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnHostAlloc = ur::level_zero::urUSMHostAlloc;
  pDdiTable->pfnDeviceAlloc = ur::level_zero::urUSMDeviceAlloc;
  pDdiTable->pfnSharedAlloc = ur::level_zero::urUSMSharedAlloc;
  pDdiTable->pfnFree = ur::level_zero::urUSMFree;
  pDdiTable->pfnGetMemAllocInfo = ur::level_zero::urUSMGetMemAllocInfo;
  pDdiTable->pfnPoolCreate = ur::level_zero::urUSMPoolCreate;
  pDdiTable->pfnPoolRetain = ur::level_zero::urUSMPoolRetain;
  pDdiTable->pfnPoolRelease = ur::level_zero::urUSMPoolRelease;
  pDdiTable->pfnPoolGetInfo = ur::level_zero::urUSMPoolGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetUSMExpProcAddrTable(
    ur_api_version_t version, ur_usm_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnPitchedAllocExp = ur::level_zero::urUSMPitchedAllocExp;
  pDdiTable->pfnImportExp = ur::level_zero::urUSMImportExp;
  pDdiTable->pfnReleaseExp = ur::level_zero::urUSMReleaseExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetUsmP2PExpProcAddrTable(
    ur_api_version_t version, ur_usm_p2p_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnEnablePeerAccessExp =
      ur::level_zero::urUsmP2PEnablePeerAccessExp;
  pDdiTable->pfnDisablePeerAccessExp =
      ur::level_zero::urUsmP2PDisablePeerAccessExp;
  pDdiTable->pfnPeerAccessGetInfoExp =
      ur::level_zero::urUsmP2PPeerAccessGetInfoExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetVirtualMemProcAddrTable(
    ur_api_version_t version, ur_virtual_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGranularityGetInfo =
      ur::level_zero::urVirtualMemGranularityGetInfo;
  pDdiTable->pfnReserve = ur::level_zero::urVirtualMemReserve;
  pDdiTable->pfnFree = ur::level_zero::urVirtualMemFree;
  pDdiTable->pfnMap = ur::level_zero::urVirtualMemMap;
  pDdiTable->pfnUnmap = ur::level_zero::urVirtualMemUnmap;
  pDdiTable->pfnSetAccess = ur::level_zero::urVirtualMemSetAccess;
  pDdiTable->pfnGetInfo = ur::level_zero::urVirtualMemGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    ur_api_version_t version, ur_device_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGet = ur::level_zero::urDeviceGet;
  pDdiTable->pfnGetInfo = ur::level_zero::urDeviceGetInfo;
  pDdiTable->pfnRetain = ur::level_zero::urDeviceRetain;
  pDdiTable->pfnRelease = ur::level_zero::urDeviceRelease;
  pDdiTable->pfnPartition = ur::level_zero::urDevicePartition;
  pDdiTable->pfnSelectBinary = ur::level_zero::urDeviceSelectBinary;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::urDeviceGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::urDeviceCreateWithNativeHandle;
  pDdiTable->pfnGetGlobalTimestamps =
      ur::level_zero::urDeviceGetGlobalTimestamps;

  return result;
}

#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO
} // namespace ur::level_zero
#elif defined(__cplusplus)
} // extern "C"
#endif

#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO
namespace ur::level_zero {
ur_result_t urAdapterGetDdiTables(ur_dditable_t *ddi) {
  if (ddi == nullptr) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  ur_result_t result;

  result = ur::level_zero::urGetGlobalProcAddrTable(UR_API_VERSION_CURRENT,
                                                    &ddi->Global);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetBindlessImagesExpProcAddrTable(
      UR_API_VERSION_CURRENT, &ddi->BindlessImagesExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetCommandBufferExpProcAddrTable(
      UR_API_VERSION_CURRENT, &ddi->CommandBufferExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetContextProcAddrTable(UR_API_VERSION_CURRENT,
                                                     &ddi->Context);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetEnqueueProcAddrTable(UR_API_VERSION_CURRENT,
                                                     &ddi->Enqueue);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetEnqueueExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                        &ddi->EnqueueExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetEventProcAddrTable(UR_API_VERSION_CURRENT,
                                                   &ddi->Event);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetKernelProcAddrTable(UR_API_VERSION_CURRENT,
                                                    &ddi->Kernel);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetKernelExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                       &ddi->KernelExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result =
      ur::level_zero::urGetMemProcAddrTable(UR_API_VERSION_CURRENT, &ddi->Mem);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetPhysicalMemProcAddrTable(UR_API_VERSION_CURRENT,
                                                         &ddi->PhysicalMem);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetPlatformProcAddrTable(UR_API_VERSION_CURRENT,
                                                      &ddi->Platform);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetProgramProcAddrTable(UR_API_VERSION_CURRENT,
                                                     &ddi->Program);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetProgramExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                        &ddi->ProgramExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetQueueProcAddrTable(UR_API_VERSION_CURRENT,
                                                   &ddi->Queue);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetSamplerProcAddrTable(UR_API_VERSION_CURRENT,
                                                     &ddi->Sampler);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetTensorMapExpProcAddrTable(
      UR_API_VERSION_CURRENT, &ddi->TensorMapExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result =
      ur::level_zero::urGetUSMProcAddrTable(UR_API_VERSION_CURRENT, &ddi->USM);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetUSMExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                    &ddi->USMExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetUsmP2PExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                       &ddi->UsmP2PExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetVirtualMemProcAddrTable(UR_API_VERSION_CURRENT,
                                                        &ddi->VirtualMem);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = ur::level_zero::urGetDeviceProcAddrTable(UR_API_VERSION_CURRENT,
                                                    &ddi->Device);
  if (result != UR_RESULT_SUCCESS)
    return result;

  return result;
}
} // namespace ur::level_zero
#endif
