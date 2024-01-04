//===--------- ur_interface_loader.cpp - Level Zero Adapter----------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur_api.h>
#include <ur_ddi.h>

namespace {

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

UR_DLLEXPORT ur_result_t UR_APICALL urGetGlobalProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_global_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnAdapterGet = urAdapterGet;
  pDdiTable->pfnAdapterRelease = urAdapterRelease;
  pDdiTable->pfnAdapterRetain = urAdapterRetain;
  pDdiTable->pfnAdapterGetLastError = urAdapterGetLastError;
  pDdiTable->pfnAdapterGetInfo = urAdapterGetInfo;

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

  pDdiTable->pfnCreate = urContextCreate;
  pDdiTable->pfnRetain = urContextRetain;
  pDdiTable->pfnRelease = urContextRelease;
  pDdiTable->pfnGetInfo = urContextGetInfo;
  pDdiTable->pfnGetNativeHandle = urContextGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urContextCreateWithNativeHandle;
  pDdiTable->pfnSetExtendedDeleter = urContextSetExtendedDeleter;

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

  pDdiTable->pfnKernelLaunch = urEnqueueKernelLaunch;
  pDdiTable->pfnEventsWait = urEnqueueEventsWait;
  pDdiTable->pfnEventsWaitWithBarrier = urEnqueueEventsWaitWithBarrier;
  pDdiTable->pfnMemBufferRead = urEnqueueMemBufferRead;
  pDdiTable->pfnMemBufferWrite = urEnqueueMemBufferWrite;
  pDdiTable->pfnMemBufferReadRect = urEnqueueMemBufferReadRect;
  pDdiTable->pfnMemBufferWriteRect = urEnqueueMemBufferWriteRect;
  pDdiTable->pfnMemBufferCopy = urEnqueueMemBufferCopy;
  pDdiTable->pfnMemBufferCopyRect = urEnqueueMemBufferCopyRect;
  pDdiTable->pfnMemBufferFill = urEnqueueMemBufferFill;
  pDdiTable->pfnMemImageRead = urEnqueueMemImageRead;
  pDdiTable->pfnMemImageWrite = urEnqueueMemImageWrite;
  pDdiTable->pfnMemImageCopy = urEnqueueMemImageCopy;
  pDdiTable->pfnMemBufferMap = urEnqueueMemBufferMap;
  pDdiTable->pfnMemUnmap = urEnqueueMemUnmap;
  pDdiTable->pfnUSMFill = urEnqueueUSMFill;
  pDdiTable->pfnUSMMemcpy = urEnqueueUSMMemcpy;
  pDdiTable->pfnUSMPrefetch = urEnqueueUSMPrefetch;
  pDdiTable->pfnUSMAdvise = urEnqueueUSMAdvise;
  pDdiTable->pfnUSMFill2D = urEnqueueUSMFill2D;
  pDdiTable->pfnUSMMemcpy2D = urEnqueueUSMMemcpy2D;
  pDdiTable->pfnDeviceGlobalVariableWrite = urEnqueueDeviceGlobalVariableWrite;
  pDdiTable->pfnDeviceGlobalVariableRead = urEnqueueDeviceGlobalVariableRead;

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
  pDdiTable->pfnGetInfo = urEventGetInfo;
  pDdiTable->pfnGetProfilingInfo = urEventGetProfilingInfo;
  pDdiTable->pfnWait = urEventWait;
  pDdiTable->pfnRetain = urEventRetain;
  pDdiTable->pfnRelease = urEventRelease;
  pDdiTable->pfnGetNativeHandle = urEventGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urEventCreateWithNativeHandle;
  pDdiTable->pfnSetCallback = urEventSetCallback;

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
  pDdiTable->pfnCreate = urKernelCreate;
  pDdiTable->pfnGetInfo = urKernelGetInfo;
  pDdiTable->pfnGetGroupInfo = urKernelGetGroupInfo;
  pDdiTable->pfnGetSubGroupInfo = urKernelGetSubGroupInfo;
  pDdiTable->pfnRetain = urKernelRetain;
  pDdiTable->pfnRelease = urKernelRelease;
  pDdiTable->pfnGetNativeHandle = urKernelGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urKernelCreateWithNativeHandle;
  pDdiTable->pfnSetArgValue = urKernelSetArgValue;
  pDdiTable->pfnSetArgLocal = urKernelSetArgLocal;
  pDdiTable->pfnSetArgPointer = urKernelSetArgPointer;
  pDdiTable->pfnSetExecInfo = urKernelSetExecInfo;
  pDdiTable->pfnSetArgSampler = urKernelSetArgSampler;
  pDdiTable->pfnSetArgMemObj = urKernelSetArgMemObj;
  pDdiTable->pfnSetSpecializationConstants = urKernelSetSpecializationConstants;
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
  pDdiTable->pfnImageCreate = urMemImageCreate;
  pDdiTable->pfnBufferCreate = urMemBufferCreate;
  pDdiTable->pfnRetain = urMemRetain;
  pDdiTable->pfnRelease = urMemRelease;
  pDdiTable->pfnBufferPartition = urMemBufferPartition;
  pDdiTable->pfnGetNativeHandle = urMemGetNativeHandle;
  pDdiTable->pfnBufferCreateWithNativeHandle =
      urMemBufferCreateWithNativeHandle;
  pDdiTable->pfnImageCreateWithNativeHandle = urMemImageCreateWithNativeHandle;
  pDdiTable->pfnGetInfo = urMemGetInfo;
  pDdiTable->pfnImageGetInfo = urMemImageGetInfo;

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
  pDdiTable->pfnGetNativeHandle = urPlatformGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urPlatformCreateWithNativeHandle;
  pDdiTable->pfnGetApiVersion = urPlatformGetApiVersion;
  pDdiTable->pfnGetBackendOption = urPlatformGetBackendOption;

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
  pDdiTable->pfnCreateWithIL = urProgramCreateWithIL;
  pDdiTable->pfnCreateWithBinary = urProgramCreateWithBinary;
  pDdiTable->pfnBuild = urProgramBuild;
  pDdiTable->pfnCompile = urProgramCompile;
  pDdiTable->pfnLink = urProgramLink;
  pDdiTable->pfnRetain = urProgramRetain;
  pDdiTable->pfnRelease = urProgramRelease;
  pDdiTable->pfnGetFunctionPointer = urProgramGetFunctionPointer;
  pDdiTable->pfnGetInfo = urProgramGetInfo;
  pDdiTable->pfnGetBuildInfo = urProgramGetBuildInfo;
  pDdiTable->pfnSetSpecializationConstants =
      urProgramSetSpecializationConstants;
  pDdiTable->pfnGetNativeHandle = urProgramGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urProgramCreateWithNativeHandle;

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

  pDdiTable->pfnGetInfo = urQueueGetInfo;
  pDdiTable->pfnCreate = urQueueCreate;
  pDdiTable->pfnRetain = urQueueRetain;
  pDdiTable->pfnRelease = urQueueRelease;
  pDdiTable->pfnGetNativeHandle = urQueueGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urQueueCreateWithNativeHandle;
  pDdiTable->pfnFinish = urQueueFinish;
  pDdiTable->pfnFlush = urQueueFlush;

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
  pDdiTable->pfnCreate = urSamplerCreate;
  pDdiTable->pfnRetain = urSamplerRetain;
  pDdiTable->pfnRelease = urSamplerRelease;
  pDdiTable->pfnGetInfo = urSamplerGetInfo;
  pDdiTable->pfnGetNativeHandle = urSamplerGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urSamplerCreateWithNativeHandle;

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

  pDdiTable->pfnHostAlloc = urUSMHostAlloc;
  pDdiTable->pfnDeviceAlloc = urUSMDeviceAlloc;
  pDdiTable->pfnSharedAlloc = urUSMSharedAlloc;
  pDdiTable->pfnFree = urUSMFree;
  pDdiTable->pfnGetMemAllocInfo = urUSMGetMemAllocInfo;
  pDdiTable->pfnPoolCreate = urUSMPoolCreate;
  pDdiTable->pfnPoolRetain = urUSMPoolRetain;
  pDdiTable->pfnPoolRelease = urUSMPoolRelease;
  pDdiTable->pfnPoolGetInfo = urUSMPoolGetInfo;

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
  pDdiTable->pfnSelectBinary = urDeviceSelectBinary;
  pDdiTable->pfnGetNativeHandle = urDeviceGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urDeviceCreateWithNativeHandle;
  pDdiTable->pfnGetGlobalTimestamps = urDeviceGetGlobalTimestamps;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetCommandBufferExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_command_buffer_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnCreateExp = urCommandBufferCreateExp;
  pDdiTable->pfnRetainExp = urCommandBufferRetainExp;
  pDdiTable->pfnReleaseExp = urCommandBufferReleaseExp;
  pDdiTable->pfnFinalizeExp = urCommandBufferFinalizeExp;
  pDdiTable->pfnAppendKernelLaunchExp = urCommandBufferAppendKernelLaunchExp;
  pDdiTable->pfnAppendUSMMemcpyExp = urCommandBufferAppendUSMMemcpyExp;
  pDdiTable->pfnAppendUSMFillExp = urCommandBufferAppendUSMFillExp;
  pDdiTable->pfnAppendMemBufferCopyExp = urCommandBufferAppendMemBufferCopyExp;
  pDdiTable->pfnAppendMemBufferCopyRectExp =
      urCommandBufferAppendMemBufferCopyRectExp;
  pDdiTable->pfnAppendMemBufferReadExp = urCommandBufferAppendMemBufferReadExp;
  pDdiTable->pfnAppendMemBufferReadRectExp =
      urCommandBufferAppendMemBufferReadRectExp;
  pDdiTable->pfnAppendMemBufferWriteExp =
      urCommandBufferAppendMemBufferWriteExp;
  pDdiTable->pfnAppendMemBufferWriteRectExp =
      urCommandBufferAppendMemBufferWriteRectExp;
  pDdiTable->pfnAppendUSMPrefetchExp = urCommandBufferAppendUSMPrefetchExp;
  pDdiTable->pfnAppendUSMAdviseExp = urCommandBufferAppendUSMAdviseExp;
  pDdiTable->pfnAppendMemBufferFillExp = urCommandBufferAppendMemBufferFillExp;
  pDdiTable->pfnEnqueueExp = urCommandBufferEnqueueExp;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetUsmP2PExpProcAddrTable(
    ur_api_version_t version, ur_usm_p2p_exp_dditable_t *pDdiTable) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnEnablePeerAccessExp = urUsmP2PEnablePeerAccessExp;
  pDdiTable->pfnDisablePeerAccessExp = urUsmP2PDisablePeerAccessExp;
  pDdiTable->pfnPeerAccessGetInfoExp = urUsmP2PPeerAccessGetInfoExp;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetBindlessImagesExpProcAddrTable(
    ur_api_version_t version, ur_bindless_images_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnUnsampledImageHandleDestroyExp =
      urBindlessImagesUnsampledImageHandleDestroyExp;
  pDdiTable->pfnSampledImageHandleDestroyExp =
      urBindlessImagesSampledImageHandleDestroyExp;
  pDdiTable->pfnImageAllocateExp = urBindlessImagesImageAllocateExp;
  pDdiTable->pfnImageFreeExp = urBindlessImagesImageFreeExp;
  pDdiTable->pfnUnsampledImageCreateExp =
      urBindlessImagesUnsampledImageCreateExp;
  pDdiTable->pfnSampledImageCreateExp = urBindlessImagesSampledImageCreateExp;
  pDdiTable->pfnImageCopyExp = urBindlessImagesImageCopyExp;
  pDdiTable->pfnImageGetInfoExp = urBindlessImagesImageGetInfoExp;
  pDdiTable->pfnMipmapGetLevelExp = urBindlessImagesMipmapGetLevelExp;
  pDdiTable->pfnMipmapFreeExp = urBindlessImagesMipmapFreeExp;
  pDdiTable->pfnImportOpaqueFDExp = urBindlessImagesImportOpaqueFDExp;
  pDdiTable->pfnMapExternalArrayExp = urBindlessImagesMapExternalArrayExp;
  pDdiTable->pfnReleaseInteropExp = urBindlessImagesReleaseInteropExp;
  pDdiTable->pfnImportExternalSemaphoreOpaqueFDExp =
      urBindlessImagesImportExternalSemaphoreOpaqueFDExp;
  pDdiTable->pfnDestroyExternalSemaphoreExp =
      urBindlessImagesDestroyExternalSemaphoreExp;
  pDdiTable->pfnWaitExternalSemaphoreExp =
      urBindlessImagesWaitExternalSemaphoreExp;
  pDdiTable->pfnSignalExternalSemaphoreExp =
      urBindlessImagesSignalExternalSemaphoreExp;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetUSMExpProcAddrTable(
    ur_api_version_t version, ur_usm_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnPitchedAllocExp = urUSMPitchedAllocExp;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetVirtualMemProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_virtual_mem_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }

  pDdiTable->pfnFree = urVirtualMemFree;
  pDdiTable->pfnGetInfo = urVirtualMemGetInfo;
  pDdiTable->pfnGranularityGetInfo = urVirtualMemGranularityGetInfo;
  pDdiTable->pfnMap = urVirtualMemMap;
  pDdiTable->pfnReserve = urVirtualMemReserve;
  pDdiTable->pfnSetAccess = urVirtualMemSetAccess;
  pDdiTable->pfnUnmap = urVirtualMemUnmap;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_physical_mem_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }

  pDdiTable->pfnCreate = urPhysicalMemCreate;
  pDdiTable->pfnRelease = urPhysicalMemRelease;
  pDdiTable->pfnRetain = urPhysicalMemRetain;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    ur_api_version_t version, ur_enqueue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCooperativeKernelLaunchExp = nullptr;

  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetKernelExpProcAddrTable(
    ur_api_version_t version, ur_kernel_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnSuggestMaxCooperativeGroupCountExp = nullptr;

  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetProgramExpProcAddrTable(
    ur_api_version_t version, ur_program_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnBuildExp = urProgramBuildExp;
  pDdiTable->pfnCompileExp = urProgramCompileExp;
  pDdiTable->pfnLinkExp = urProgramLinkExp;

  return UR_RESULT_SUCCESS;
}
#if defined(__cplusplus)
} // extern "C"
#endif
