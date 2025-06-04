//===----------- ur_interface_loader.cpp - Native CPU Plugin  -------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include <ur_api.h>
#include <ur_ddi.h>

namespace {

// TODO - this is a duplicate of what is in the L0 plugin
// We should move this to somewhere common
ur_result_t validateProcInputs(ur_api_version_t version, void *pDdiTable) {
  if (pDdiTable == nullptr) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }
  // Pre 1.0 we enforce that loader and adapter must have the same version.
  // Post 1.0 only a major version match should be required.
  if (version != UR_API_VERSION_CURRENT) {
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }
  return UR_RESULT_SUCCESS;
}
} // namespace

extern "C" {

UR_DLLEXPORT ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    ur_api_version_t version, ur_platform_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreateWithNativeHandle = urPlatformCreateWithNativeHandle;
  pDdiTable->pfnGet = urPlatformGet;
  pDdiTable->pfnGetApiVersion = urPlatformGetApiVersion;
  pDdiTable->pfnGetInfo = urPlatformGetInfo;
  pDdiTable->pfnGetNativeHandle = urPlatformGetNativeHandle;
  pDdiTable->pfnGetBackendOption = urPlatformGetBackendOption;
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
  pDdiTable->pfnGetGlobalVariablePointer = urProgramGetGlobalVariablePointer;
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
  pDdiTable->pfnSetArgLocal = urKernelSetArgLocal;
  pDdiTable->pfnSetArgMemObj = urKernelSetArgMemObj;
  pDdiTable->pfnSetArgPointer = urKernelSetArgPointer;
  pDdiTable->pfnSetArgSampler = urKernelSetArgSampler;
  pDdiTable->pfnSetArgValue = urKernelSetArgValue;
  pDdiTable->pfnSetExecInfo = urKernelSetExecInfo;
  pDdiTable->pfnSetSpecializationConstants = urKernelSetSpecializationConstants;
  pDdiTable->pfnGetSuggestedLocalWorkSize = urKernelGetSuggestedLocalWorkSize;
  pDdiTable->pfnSuggestMaxCooperativeGroupCount =
      urKernelSuggestMaxCooperativeGroupCount;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    ur_api_version_t version, ur_sampler_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreate = urSamplerCreate;
  pDdiTable->pfnCreateWithNativeHandle = urSamplerCreateWithNativeHandle;
  pDdiTable->pfnGetInfo = urSamplerGetInfo;
  pDdiTable->pfnGetNativeHandle = urSamplerGetNativeHandle;
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
  pDdiTable->pfnDeviceGlobalVariableRead = urEnqueueDeviceGlobalVariableRead;
  pDdiTable->pfnDeviceGlobalVariableWrite = urEnqueueDeviceGlobalVariableWrite;
  pDdiTable->pfnEventsWait = urEnqueueEventsWait;
  pDdiTable->pfnEventsWaitWithBarrier = urEnqueueEventsWaitWithBarrier;
  pDdiTable->pfnEventsWaitWithBarrierExt = urEnqueueEventsWaitWithBarrierExt;
  pDdiTable->pfnKernelLaunch = urEnqueueKernelLaunch;
  pDdiTable->pfnMemBufferCopy = urEnqueueMemBufferCopy;
  pDdiTable->pfnMemBufferCopyRect = urEnqueueMemBufferCopyRect;
  pDdiTable->pfnMemBufferFill = urEnqueueMemBufferFill;
  pDdiTable->pfnMemBufferMap = urEnqueueMemBufferMap;
  pDdiTable->pfnMemBufferRead = urEnqueueMemBufferRead;
  pDdiTable->pfnMemBufferReadRect = urEnqueueMemBufferReadRect;
  pDdiTable->pfnMemBufferWrite = urEnqueueMemBufferWrite;
  pDdiTable->pfnMemBufferWriteRect = urEnqueueMemBufferWriteRect;
  pDdiTable->pfnMemImageCopy = urEnqueueMemImageCopy;
  pDdiTable->pfnMemImageRead = urEnqueueMemImageRead;
  pDdiTable->pfnMemImageWrite = urEnqueueMemImageWrite;
  pDdiTable->pfnMemUnmap = urEnqueueMemUnmap;
  pDdiTable->pfnUSMFill2D = urEnqueueUSMFill2D;
  pDdiTable->pfnUSMFill = urEnqueueUSMFill;
  pDdiTable->pfnUSMAdvise = urEnqueueUSMAdvise;
  pDdiTable->pfnUSMMemcpy2D = urEnqueueUSMMemcpy2D;
  pDdiTable->pfnUSMMemcpy = urEnqueueUSMMemcpy;
  pDdiTable->pfnUSMPrefetch = urEnqueueUSMPrefetch;
  pDdiTable->pfnReadHostPipe = urEnqueueReadHostPipe;
  pDdiTable->pfnWriteHostPipe = urEnqueueWriteHostPipe;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetAdapterProcAddrTable(
    ur_api_version_t version, ur_adapter_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnGet = urAdapterGet;
  pDdiTable->pfnGetInfo = urAdapterGetInfo;
  pDdiTable->pfnRelease = urAdapterRelease;
  pDdiTable->pfnRetain = urAdapterRetain;
  pDdiTable->pfnGetLastError = urAdapterGetLastError;
  pDdiTable->pfnSetLoggerCallback = urAdapterSetLoggerCallback;
  pDdiTable->pfnSetLoggerCallbackLevel = urAdapterSetLoggerCallbackLevel;

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
  pDdiTable->pfnPoolCreate = urUSMPoolCreate;
  pDdiTable->pfnPoolRetain = urUSMPoolRetain;
  pDdiTable->pfnPoolRelease = urUSMPoolRelease;
  pDdiTable->pfnPoolGetInfo = urUSMPoolGetInfo;
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

UR_DLLEXPORT ur_result_t UR_APICALL urGetCommandBufferExpProcAddrTable(
    ur_api_version_t version, ur_command_buffer_exp_dditable_t *pDdiTable) {
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
  pDdiTable->pfnUpdateKernelLaunchExp = urCommandBufferUpdateKernelLaunchExp;
  pDdiTable->pfnGetInfoExp = urCommandBufferGetInfoExp;
  pDdiTable->pfnUpdateWaitEventsExp = urCommandBufferUpdateWaitEventsExp;
  pDdiTable->pfnUpdateSignalEventExp = urCommandBufferUpdateSignalEventExp;
  pDdiTable->pfnAppendNativeCommandExp = urCommandBufferAppendNativeCommandExp;
  pDdiTable->pfnGetNativeHandleExp = urCommandBufferGetNativeHandleExp;

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
  pDdiTable->pfnImportExternalMemoryExp =
      urBindlessImagesImportExternalMemoryExp;
  pDdiTable->pfnMapExternalArrayExp = urBindlessImagesMapExternalArrayExp;
  pDdiTable->pfnMapExternalLinearMemoryExp =
      urBindlessImagesMapExternalLinearMemoryExp;
  pDdiTable->pfnReleaseExternalMemoryExp =
      urBindlessImagesReleaseExternalMemoryExp;
  pDdiTable->pfnFreeMappedLinearMemoryExp =
      urBindlessImagesFreeMappedLinearMemoryExp;
  pDdiTable->pfnImportExternalSemaphoreExp =
      urBindlessImagesImportExternalSemaphoreExp;
  pDdiTable->pfnReleaseExternalSemaphoreExp =
      urBindlessImagesReleaseExternalSemaphoreExp;
  pDdiTable->pfnWaitExternalSemaphoreExp =
      urBindlessImagesWaitExternalSemaphoreExp;
  pDdiTable->pfnSignalExternalSemaphoreExp =
      urBindlessImagesSignalExternalSemaphoreExp;
  pDdiTable->pfnGetImageMemoryHandleTypeSupportExp =
      urBindlessImagesGetImageMemoryHandleTypeSupportExp;
  pDdiTable->pfnGetImageUnsampledHandleSupportExp =
      urBindlessImagesGetImageUnsampledHandleSupportExp;
  pDdiTable->pfnGetImageSampledHandleSupportExp =
      urBindlessImagesGetImageSampledHandleSupportExp;

  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    ur_api_version_t version, ur_physical_mem_dditable_t *pDdiTable) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }

  pDdiTable->pfnCreate = urPhysicalMemCreate;
  pDdiTable->pfnRelease = urPhysicalMemRelease;
  pDdiTable->pfnRetain = urPhysicalMemRetain;
  pDdiTable->pfnGetInfo = urPhysicalMemGetInfo;

  return retVal;
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
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_virtual_mem_dditable_t *pDdiTable) {
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

UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    ur_api_version_t version, ur_enqueue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnTimestampRecordingExp = urEnqueueTimestampRecordingExp;
  pDdiTable->pfnNativeCommandExp = urEnqueueNativeCommandExp;
  pDdiTable->pfnCommandBufferExp = urEnqueueCommandBufferExp;
  pDdiTable->pfnKernelLaunchWithArgsExp = urEnqueueKernelLaunchWithArgsExp;

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

UR_DLLEXPORT ur_result_t UR_APICALL urAllAddrTable(ur_api_version_t version,
                                                   ur_dditable_t *pDdiTable) {
  urGetAdapterProcAddrTable(version, &pDdiTable->Adapter);
  urGetBindlessImagesExpProcAddrTable(version, &pDdiTable->BindlessImagesExp);
  urGetCommandBufferExpProcAddrTable(version, &pDdiTable->CommandBufferExp);
  urGetContextProcAddrTable(version, &pDdiTable->Context);
  urGetEnqueueProcAddrTable(version, &pDdiTable->Enqueue);
  urGetEnqueueExpProcAddrTable(version, &pDdiTable->EnqueueExp);
  urGetEventProcAddrTable(version, &pDdiTable->Event);
  urGetKernelProcAddrTable(version, &pDdiTable->Kernel);
  urGetMemProcAddrTable(version, &pDdiTable->Mem);
  urGetPhysicalMemProcAddrTable(version, &pDdiTable->PhysicalMem);
  urGetPlatformProcAddrTable(version, &pDdiTable->Platform);
  urGetProgramProcAddrTable(version, &pDdiTable->Program);
  urGetProgramExpProcAddrTable(version, &pDdiTable->ProgramExp);
  urGetQueueProcAddrTable(version, &pDdiTable->Queue);
  urGetSamplerProcAddrTable(version, &pDdiTable->Sampler);
  urGetUSMProcAddrTable(version, &pDdiTable->USM);
  urGetUSMExpProcAddrTable(version, &pDdiTable->USMExp);
  urGetUsmP2PExpProcAddrTable(version, &pDdiTable->UsmP2PExp);
  urGetVirtualMemProcAddrTable(version, &pDdiTable->VirtualMem);
  urGetDeviceProcAddrTable(version, &pDdiTable->Device);

  return UR_RESULT_SUCCESS;
}
} // extern "C"

const ur_dditable_t *ur::native_cpu::ddi_getter::value() {
  static std::once_flag flag;
  static ur_dditable_t table;

  std::call_once(flag,
                 []() { urAllAddrTable(UR_API_VERSION_CURRENT, &table); });
  return &table;
}
