//===--------- ur_interface_loader.cpp - Level Zero Adapter ------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <mutex>
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

#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO_V2
namespace ur::level_zero_v2 {
#else
extern "C" {
#endif

UR_APIEXPORT ur_result_t UR_APICALL urGetAdapterProcAddrTable(
    ur_api_version_t version, ur_adapter_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGet = ur::level_zero::urAdapterGet;
  pDdiTable->pfnRelease = ur::level_zero::urAdapterRelease;
  pDdiTable->pfnRetain = ur::level_zero::urAdapterRetain;
  pDdiTable->pfnGetLastError = ur::level_zero::urAdapterGetLastError;
  pDdiTable->pfnGetInfo = ur::level_zero::urAdapterGetInfo;
  pDdiTable->pfnSetLoggerCallback =
      ur::level_zero::urAdapterSetLoggerCallback;
  pDdiTable->pfnSetLoggerCallbackLevel =
      ur::level_zero::urAdapterSetLoggerCallbackLevel;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetBindlessImagesExpProcAddrTable(
    ur_api_version_t version, ur_bindless_images_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnUnsampledImageHandleDestroyExp =
      ur::level_zero_v2::urBindlessImagesUnsampledImageHandleDestroyExp;
  pDdiTable->pfnSampledImageHandleDestroyExp =
      ur::level_zero_v2::urBindlessImagesSampledImageHandleDestroyExp;
  pDdiTable->pfnImageAllocateExp =
      ur::level_zero_v2::urBindlessImagesImageAllocateExp;
  pDdiTable->pfnImageFreeExp = ur::level_zero_v2::urBindlessImagesImageFreeExp;
  pDdiTable->pfnUnsampledImageCreateExp =
      ur::level_zero_v2::urBindlessImagesUnsampledImageCreateExp;
  pDdiTable->pfnSampledImageCreateExp =
      ur::level_zero_v2::urBindlessImagesSampledImageCreateExp;
  pDdiTable->pfnImageCopyExp = ur::level_zero_v2::urBindlessImagesImageCopyExp;
  pDdiTable->pfnImageGetInfoExp =
      ur::level_zero_v2::urBindlessImagesImageGetInfoExp;
  pDdiTable->pfnGetImageMemoryHandleTypeSupportExp =
      ur::level_zero_v2::urBindlessImagesGetImageMemoryHandleTypeSupportExp;
  pDdiTable->pfnGetImageUnsampledHandleSupportExp =
      ur::level_zero_v2::urBindlessImagesGetImageUnsampledHandleSupportExp;
  pDdiTable->pfnGetImageSampledHandleSupportExp =
      ur::level_zero_v2::urBindlessImagesGetImageSampledHandleSupportExp;
  pDdiTable->pfnMipmapGetLevelExp =
      ur::level_zero_v2::urBindlessImagesMipmapGetLevelExp;
  pDdiTable->pfnMipmapFreeExp =
      ur::level_zero_v2::urBindlessImagesMipmapFreeExp;
  pDdiTable->pfnImportExternalMemoryExp =
      ur::level_zero_v2::urBindlessImagesImportExternalMemoryExp;
  pDdiTable->pfnMapExternalArrayExp =
      ur::level_zero_v2::urBindlessImagesMapExternalArrayExp;
  pDdiTable->pfnMapExternalLinearMemoryExp =
      ur::level_zero_v2::urBindlessImagesMapExternalLinearMemoryExp;
  pDdiTable->pfnReleaseExternalMemoryExp =
      ur::level_zero_v2::urBindlessImagesReleaseExternalMemoryExp;
  pDdiTable->pfnFreeMappedLinearMemoryExp =
      ur::level_zero_v2::urBindlessImagesFreeMappedLinearMemoryExp;
  pDdiTable->pfnSupportsImportingHandleTypeExp =
      ur::level_zero_v2::urBindlessImagesSupportsImportingHandleTypeExp;
  pDdiTable->pfnImportExternalSemaphoreExp =
      ur::level_zero_v2::urBindlessImagesImportExternalSemaphoreExp;
  pDdiTable->pfnReleaseExternalSemaphoreExp =
      ur::level_zero_v2::urBindlessImagesReleaseExternalSemaphoreExp;
  pDdiTable->pfnWaitExternalSemaphoreExp =
      ur::level_zero_v2::urBindlessImagesWaitExternalSemaphoreExp;
  pDdiTable->pfnSignalExternalSemaphoreExp =
      ur::level_zero_v2::urBindlessImagesSignalExternalSemaphoreExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetCommandBufferExpProcAddrTable(
    ur_api_version_t version, ur_command_buffer_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateExp = ur::level_zero_v2::urCommandBufferCreateExp;
  pDdiTable->pfnRetainExp = ur::level_zero_v2::urCommandBufferRetainExp;
  pDdiTable->pfnReleaseExp = ur::level_zero_v2::urCommandBufferReleaseExp;
  pDdiTable->pfnFinalizeExp = ur::level_zero_v2::urCommandBufferFinalizeExp;
  pDdiTable->pfnAppendKernelLaunchExp =
      ur::level_zero_v2::urCommandBufferAppendKernelLaunchExp;
  pDdiTable->pfnAppendUSMMemcpyExp =
      ur::level_zero_v2::urCommandBufferAppendUSMMemcpyExp;
  pDdiTable->pfnAppendUSMFillExp =
      ur::level_zero_v2::urCommandBufferAppendUSMFillExp;
  pDdiTable->pfnAppendMemBufferCopyExp =
      ur::level_zero_v2::urCommandBufferAppendMemBufferCopyExp;
  pDdiTable->pfnAppendMemBufferWriteExp =
      ur::level_zero_v2::urCommandBufferAppendMemBufferWriteExp;
  pDdiTable->pfnAppendMemBufferReadExp =
      ur::level_zero_v2::urCommandBufferAppendMemBufferReadExp;
  pDdiTable->pfnAppendMemBufferCopyRectExp =
      ur::level_zero_v2::urCommandBufferAppendMemBufferCopyRectExp;
  pDdiTable->pfnAppendMemBufferWriteRectExp =
      ur::level_zero_v2::urCommandBufferAppendMemBufferWriteRectExp;
  pDdiTable->pfnAppendMemBufferReadRectExp =
      ur::level_zero_v2::urCommandBufferAppendMemBufferReadRectExp;
  pDdiTable->pfnAppendMemBufferFillExp =
      ur::level_zero_v2::urCommandBufferAppendMemBufferFillExp;
  pDdiTable->pfnAppendUSMPrefetchExp =
      ur::level_zero_v2::urCommandBufferAppendUSMPrefetchExp;
  pDdiTable->pfnAppendUSMAdviseExp =
      ur::level_zero_v2::urCommandBufferAppendUSMAdviseExp;
  pDdiTable->pfnAppendNativeCommandExp =
      ur::level_zero_v2::urCommandBufferAppendNativeCommandExp;
  pDdiTable->pfnUpdateKernelLaunchExp =
      ur::level_zero_v2::urCommandBufferUpdateKernelLaunchExp;
  pDdiTable->pfnUpdateSignalEventExp =
      ur::level_zero_v2::urCommandBufferUpdateSignalEventExp;
  pDdiTable->pfnUpdateWaitEventsExp =
      ur::level_zero_v2::urCommandBufferUpdateWaitEventsExp;
  pDdiTable->pfnGetInfoExp = ur::level_zero_v2::urCommandBufferGetInfoExp;
  pDdiTable->pfnGetNativeHandleExp =
      ur::level_zero_v2::urCommandBufferGetNativeHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetContextProcAddrTable(
    ur_api_version_t version, ur_context_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::level_zero_v2::urContextCreate;
  pDdiTable->pfnRetain = ur::level_zero_v2::urContextRetain;
  pDdiTable->pfnRelease = ur::level_zero_v2::urContextRelease;
  pDdiTable->pfnGetInfo = ur::level_zero_v2::urContextGetInfo;
  pDdiTable->pfnGetNativeHandle = ur::level_zero_v2::urContextGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero_v2::urContextCreateWithNativeHandle;
  pDdiTable->pfnSetExtendedDeleter =
      ur::level_zero_v2::urContextSetExtendedDeleter;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    ur_api_version_t version, ur_enqueue_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnKernelLaunch = ur::level_zero_v2::urEnqueueKernelLaunch;
  pDdiTable->pfnEventsWait = ur::level_zero_v2::urEnqueueEventsWait;
  pDdiTable->pfnEventsWaitWithBarrier =
      ur::level_zero_v2::urEnqueueEventsWaitWithBarrier;
  pDdiTable->pfnMemBufferRead = ur::level_zero_v2::urEnqueueMemBufferRead;
  pDdiTable->pfnMemBufferWrite = ur::level_zero_v2::urEnqueueMemBufferWrite;
  pDdiTable->pfnMemBufferReadRect =
      ur::level_zero_v2::urEnqueueMemBufferReadRect;
  pDdiTable->pfnMemBufferWriteRect =
      ur::level_zero_v2::urEnqueueMemBufferWriteRect;
  pDdiTable->pfnMemBufferCopy = ur::level_zero_v2::urEnqueueMemBufferCopy;
  pDdiTable->pfnMemBufferCopyRect =
      ur::level_zero_v2::urEnqueueMemBufferCopyRect;
  pDdiTable->pfnMemBufferFill = ur::level_zero_v2::urEnqueueMemBufferFill;
  pDdiTable->pfnMemImageRead = ur::level_zero_v2::urEnqueueMemImageRead;
  pDdiTable->pfnMemImageWrite = ur::level_zero_v2::urEnqueueMemImageWrite;
  pDdiTable->pfnMemImageCopy = ur::level_zero_v2::urEnqueueMemImageCopy;
  pDdiTable->pfnMemBufferMap = ur::level_zero_v2::urEnqueueMemBufferMap;
  pDdiTable->pfnMemUnmap = ur::level_zero_v2::urEnqueueMemUnmap;
  pDdiTable->pfnUSMFill = ur::level_zero_v2::urEnqueueUSMFill;
  pDdiTable->pfnUSMMemcpy = ur::level_zero_v2::urEnqueueUSMMemcpy;
  pDdiTable->pfnUSMPrefetch = ur::level_zero_v2::urEnqueueUSMPrefetch;
  pDdiTable->pfnUSMAdvise = ur::level_zero_v2::urEnqueueUSMAdvise;
  pDdiTable->pfnUSMFill2D = ur::level_zero_v2::urEnqueueUSMFill2D;
  pDdiTable->pfnUSMMemcpy2D = ur::level_zero_v2::urEnqueueUSMMemcpy2D;
  pDdiTable->pfnDeviceGlobalVariableWrite =
      ur::level_zero_v2::urEnqueueDeviceGlobalVariableWrite;
  pDdiTable->pfnDeviceGlobalVariableRead =
      ur::level_zero_v2::urEnqueueDeviceGlobalVariableRead;
  pDdiTable->pfnReadHostPipe = ur::level_zero_v2::urEnqueueReadHostPipe;
  pDdiTable->pfnWriteHostPipe = ur::level_zero_v2::urEnqueueWriteHostPipe;
  pDdiTable->pfnEventsWaitWithBarrierExt =
      ur::level_zero_v2::urEnqueueEventsWaitWithBarrierExt;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    ur_api_version_t version, ur_enqueue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnKernelLaunchWithArgsExp =
      ur::level_zero_v2::urEnqueueKernelLaunchWithArgsExp;
  pDdiTable->pfnUSMDeviceAllocExp =
      ur::level_zero_v2::urEnqueueUSMDeviceAllocExp;
  pDdiTable->pfnUSMSharedAllocExp =
      ur::level_zero_v2::urEnqueueUSMSharedAllocExp;
  pDdiTable->pfnUSMHostAllocExp = ur::level_zero_v2::urEnqueueUSMHostAllocExp;
  pDdiTable->pfnUSMFreeExp = ur::level_zero_v2::urEnqueueUSMFreeExp;
  pDdiTable->pfnCommandBufferExp = ur::level_zero_v2::urEnqueueCommandBufferExp;
  pDdiTable->pfnTimestampRecordingExp =
      ur::level_zero_v2::urEnqueueTimestampRecordingExp;
  pDdiTable->pfnHostTaskExp = ur::level_zero_v2::urEnqueueHostTaskExp;
  pDdiTable->pfnNativeCommandExp = ur::level_zero_v2::urEnqueueNativeCommandExp;
  pDdiTable->pfnGraphExp = ur::level_zero_v2::urEnqueueGraphExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEventProcAddrTable(
    ur_api_version_t version, ur_event_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetInfo = ur::level_zero_v2::urEventGetInfo;
  pDdiTable->pfnGetProfilingInfo = ur::level_zero_v2::urEventGetProfilingInfo;
  pDdiTable->pfnWait = ur::level_zero_v2::urEventWait;
  pDdiTable->pfnRetain = ur::level_zero_v2::urEventRetain;
  pDdiTable->pfnRelease = ur::level_zero_v2::urEventRelease;
  pDdiTable->pfnGetNativeHandle = ur::level_zero_v2::urEventGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero_v2::urEventCreateWithNativeHandle;
  pDdiTable->pfnSetCallback = ur::level_zero_v2::urEventSetCallback;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetGraphExpProcAddrTable(
    ur_api_version_t version, ur_graph_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateExp = ur::level_zero_v2::urGraphCreateExp;
  pDdiTable->pfnInstantiateGraphExp =
      ur::level_zero_v2::urGraphInstantiateGraphExp;
  pDdiTable->pfnDestroyExp = ur::level_zero_v2::urGraphDestroyExp;
  pDdiTable->pfnExecutableGraphDestroyExp =
      ur::level_zero_v2::urGraphExecutableGraphDestroyExp;
  pDdiTable->pfnIsEmptyExp = ur::level_zero_v2::urGraphIsEmptyExp;
  pDdiTable->pfnDumpContentsExp = ur::level_zero_v2::urGraphDumpContentsExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetIPCExpProcAddrTable(
    ur_api_version_t version, ur_ipc_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetMemHandleExp = ur::level_zero_v2::urIPCGetMemHandleExp;
  pDdiTable->pfnPutMemHandleExp = ur::level_zero_v2::urIPCPutMemHandleExp;
  pDdiTable->pfnOpenMemHandleExp = ur::level_zero_v2::urIPCOpenMemHandleExp;
  pDdiTable->pfnCloseMemHandleExp = ur::level_zero_v2::urIPCCloseMemHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetKernelProcAddrTable(
    ur_api_version_t version, ur_kernel_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::level_zero_v2::urKernelCreate;
  pDdiTable->pfnGetInfo = ur::level_zero_v2::urKernelGetInfo;
  pDdiTable->pfnGetGroupInfo = ur::level_zero_v2::urKernelGetGroupInfo;
  pDdiTable->pfnGetSubGroupInfo = ur::level_zero_v2::urKernelGetSubGroupInfo;
  pDdiTable->pfnRetain = ur::level_zero_v2::urKernelRetain;
  pDdiTable->pfnRelease = ur::level_zero_v2::urKernelRelease;
  pDdiTable->pfnGetNativeHandle = ur::level_zero_v2::urKernelGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero_v2::urKernelCreateWithNativeHandle;
  pDdiTable->pfnGetSuggestedLocalWorkSize =
      ur::level_zero_v2::urKernelGetSuggestedLocalWorkSize;
  pDdiTable->pfnSetArgValue = ur::level_zero_v2::urKernelSetArgValue;
  pDdiTable->pfnSetArgLocal = ur::level_zero_v2::urKernelSetArgLocal;
  pDdiTable->pfnSetArgPointer = ur::level_zero_v2::urKernelSetArgPointer;
  pDdiTable->pfnSetExecInfo = ur::level_zero_v2::urKernelSetExecInfo;
  pDdiTable->pfnSetArgSampler = ur::level_zero_v2::urKernelSetArgSampler;
  pDdiTable->pfnSetArgMemObj = ur::level_zero_v2::urKernelSetArgMemObj;
  pDdiTable->pfnSetSpecializationConstants =
      ur::level_zero_v2::urKernelSetSpecializationConstants;
  pDdiTable->pfnSuggestMaxCooperativeGroupCount =
      ur::level_zero_v2::urKernelSuggestMaxCooperativeGroupCount;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGetMemProcAddrTable(ur_api_version_t version, ur_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnImageCreate = ur::level_zero_v2::urMemImageCreate;
  pDdiTable->pfnBufferCreate = ur::level_zero_v2::urMemBufferCreate;
  pDdiTable->pfnRetain = ur::level_zero_v2::urMemRetain;
  pDdiTable->pfnRelease = ur::level_zero_v2::urMemRelease;
  pDdiTable->pfnBufferPartition = ur::level_zero_v2::urMemBufferPartition;
  pDdiTable->pfnGetNativeHandle = ur::level_zero_v2::urMemGetNativeHandle;
  pDdiTable->pfnBufferCreateWithNativeHandle =
      ur::level_zero_v2::urMemBufferCreateWithNativeHandle;
  pDdiTable->pfnImageCreateWithNativeHandle =
      ur::level_zero_v2::urMemImageCreateWithNativeHandle;
  pDdiTable->pfnGetInfo = ur::level_zero_v2::urMemGetInfo;
  pDdiTable->pfnImageGetInfo = ur::level_zero_v2::urMemImageGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetMemoryExportExpProcAddrTable(
    ur_api_version_t version, ur_memory_export_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnAllocExportableMemoryExp =
      ur::level_zero_v2::urMemoryExportAllocExportableMemoryExp;
  pDdiTable->pfnFreeExportableMemoryExp =
      ur::level_zero_v2::urMemoryExportFreeExportableMemoryExp;
  pDdiTable->pfnExportMemoryHandleExp =
      ur::level_zero_v2::urMemoryExportExportMemoryHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    ur_api_version_t version, ur_physical_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::level_zero_v2::urPhysicalMemCreate;
  pDdiTable->pfnRetain = ur::level_zero_v2::urPhysicalMemRetain;
  pDdiTable->pfnRelease = ur::level_zero_v2::urPhysicalMemRelease;
  pDdiTable->pfnGetInfo = ur::level_zero_v2::urPhysicalMemGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    ur_api_version_t version, ur_platform_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGet = ur::level_zero_v2::urPlatformGet;
  pDdiTable->pfnGetInfo = ur::level_zero_v2::urPlatformGetInfo;
  pDdiTable->pfnGetNativeHandle = ur::level_zero_v2::urPlatformGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero_v2::urPlatformCreateWithNativeHandle;
  pDdiTable->pfnGetApiVersion = ur::level_zero_v2::urPlatformGetApiVersion;
  pDdiTable->pfnGetBackendOption =
      ur::level_zero_v2::urPlatformGetBackendOption;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetProgramProcAddrTable(
    ur_api_version_t version, ur_program_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateWithIL = ur::level_zero_v2::urProgramCreateWithIL;
  pDdiTable->pfnCreateWithBinary = ur::level_zero_v2::urProgramCreateWithBinary;
  pDdiTable->pfnBuild = ur::level_zero_v2::urProgramBuild;
  pDdiTable->pfnCompile = ur::level_zero_v2::urProgramCompile;
  pDdiTable->pfnLink = ur::level_zero_v2::urProgramLink;
  pDdiTable->pfnRetain = ur::level_zero_v2::urProgramRetain;
  pDdiTable->pfnRelease = ur::level_zero_v2::urProgramRelease;
  pDdiTable->pfnGetFunctionPointer =
      ur::level_zero_v2::urProgramGetFunctionPointer;
  pDdiTable->pfnGetGlobalVariablePointer =
      ur::level_zero_v2::urProgramGetGlobalVariablePointer;
  pDdiTable->pfnGetInfo = ur::level_zero_v2::urProgramGetInfo;
  pDdiTable->pfnGetBuildInfo = ur::level_zero_v2::urProgramGetBuildInfo;
  pDdiTable->pfnSetSpecializationConstants =
      ur::level_zero_v2::urProgramSetSpecializationConstants;
  pDdiTable->pfnGetNativeHandle = ur::level_zero_v2::urProgramGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero_v2::urProgramCreateWithNativeHandle;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetProgramExpProcAddrTable(
    ur_api_version_t version, ur_program_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnDynamicLinkExp = ur::level_zero_v2::urProgramDynamicLinkExp;
  pDdiTable->pfnBuildExp = ur::level_zero_v2::urProgramBuildExp;
  pDdiTable->pfnCompileExp = ur::level_zero_v2::urProgramCompileExp;
  pDdiTable->pfnLinkExp = ur::level_zero_v2::urProgramLinkExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetQueueProcAddrTable(
    ur_api_version_t version, ur_queue_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetInfo = ur::level_zero_v2::urQueueGetInfo;
  pDdiTable->pfnCreate = ur::level_zero_v2::urQueueCreate;
  pDdiTable->pfnRetain = ur::level_zero_v2::urQueueRetain;
  pDdiTable->pfnRelease = ur::level_zero_v2::urQueueRelease;
  pDdiTable->pfnGetNativeHandle = ur::level_zero_v2::urQueueGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero_v2::urQueueCreateWithNativeHandle;
  pDdiTable->pfnFinish = ur::level_zero_v2::urQueueFinish;
  pDdiTable->pfnFlush = ur::level_zero_v2::urQueueFlush;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetQueueExpProcAddrTable(
    ur_api_version_t version, ur_queue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnBeginGraphCaptureExp =
      ur::level_zero_v2::urQueueBeginGraphCaptureExp;
  pDdiTable->pfnBeginCaptureIntoGraphExp =
      ur::level_zero_v2::urQueueBeginCaptureIntoGraphExp;
  pDdiTable->pfnEndGraphCaptureExp =
      ur::level_zero_v2::urQueueEndGraphCaptureExp;
  pDdiTable->pfnIsGraphCaptureEnabledExp =
      ur::level_zero_v2::urQueueIsGraphCaptureEnabledExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    ur_api_version_t version, ur_sampler_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::level_zero_v2::urSamplerCreate;
  pDdiTable->pfnRetain = ur::level_zero_v2::urSamplerRetain;
  pDdiTable->pfnRelease = ur::level_zero_v2::urSamplerRelease;
  pDdiTable->pfnGetInfo = ur::level_zero_v2::urSamplerGetInfo;
  pDdiTable->pfnGetNativeHandle = ur::level_zero_v2::urSamplerGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero_v2::urSamplerCreateWithNativeHandle;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGetUSMProcAddrTable(ur_api_version_t version, ur_usm_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnHostAlloc = ur::level_zero_v2::urUSMHostAlloc;
  pDdiTable->pfnDeviceAlloc = ur::level_zero_v2::urUSMDeviceAlloc;
  pDdiTable->pfnSharedAlloc = ur::level_zero_v2::urUSMSharedAlloc;
  pDdiTable->pfnFree = ur::level_zero_v2::urUSMFree;
  pDdiTable->pfnGetMemAllocInfo = ur::level_zero_v2::urUSMGetMemAllocInfo;
  pDdiTable->pfnPoolCreate = ur::level_zero_v2::urUSMPoolCreate;
  pDdiTable->pfnPoolRetain = ur::level_zero_v2::urUSMPoolRetain;
  pDdiTable->pfnPoolRelease = ur::level_zero_v2::urUSMPoolRelease;
  pDdiTable->pfnPoolGetInfo = ur::level_zero_v2::urUSMPoolGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetUSMExpProcAddrTable(
    ur_api_version_t version, ur_usm_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnPoolCreateExp = ur::level_zero_v2::urUSMPoolCreateExp;
  pDdiTable->pfnPoolDestroyExp = ur::level_zero_v2::urUSMPoolDestroyExp;
  pDdiTable->pfnPoolGetDefaultDevicePoolExp =
      ur::level_zero_v2::urUSMPoolGetDefaultDevicePoolExp;
  pDdiTable->pfnPoolGetInfoExp = ur::level_zero_v2::urUSMPoolGetInfoExp;
  pDdiTable->pfnPoolSetInfoExp = ur::level_zero_v2::urUSMPoolSetInfoExp;
  pDdiTable->pfnPoolSetDevicePoolExp =
      ur::level_zero_v2::urUSMPoolSetDevicePoolExp;
  pDdiTable->pfnPoolGetDevicePoolExp =
      ur::level_zero_v2::urUSMPoolGetDevicePoolExp;
  pDdiTable->pfnPoolTrimToExp = ur::level_zero_v2::urUSMPoolTrimToExp;
  pDdiTable->pfnPitchedAllocExp = ur::level_zero_v2::urUSMPitchedAllocExp;
  pDdiTable->pfnContextMemcpyExp = ur::level_zero_v2::urUSMContextMemcpyExp;
  pDdiTable->pfnImportExp = ur::level_zero_v2::urUSMImportExp;
  pDdiTable->pfnReleaseExp = ur::level_zero_v2::urUSMReleaseExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetUsmP2PExpProcAddrTable(
    ur_api_version_t version, ur_usm_p2p_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnEnablePeerAccessExp =
      ur::level_zero_v2::urUsmP2PEnablePeerAccessExp;
  pDdiTable->pfnDisablePeerAccessExp =
      ur::level_zero_v2::urUsmP2PDisablePeerAccessExp;
  pDdiTable->pfnPeerAccessGetInfoExp =
      ur::level_zero_v2::urUsmP2PPeerAccessGetInfoExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetVirtualMemProcAddrTable(
    ur_api_version_t version, ur_virtual_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGranularityGetInfo =
      ur::level_zero_v2::urVirtualMemGranularityGetInfo;
  pDdiTable->pfnReserve = ur::level_zero_v2::urVirtualMemReserve;
  pDdiTable->pfnFree = ur::level_zero_v2::urVirtualMemFree;
  pDdiTable->pfnMap = ur::level_zero_v2::urVirtualMemMap;
  pDdiTable->pfnUnmap = ur::level_zero_v2::urVirtualMemUnmap;
  pDdiTable->pfnSetAccess = ur::level_zero_v2::urVirtualMemSetAccess;
  pDdiTable->pfnGetInfo = ur::level_zero_v2::urVirtualMemGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    ur_api_version_t version, ur_device_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGet = ur::level_zero_v2::urDeviceGet;
  pDdiTable->pfnGetInfo = ur::level_zero_v2::urDeviceGetInfo;
  pDdiTable->pfnRetain = ur::level_zero_v2::urDeviceRetain;
  pDdiTable->pfnRelease = ur::level_zero_v2::urDeviceRelease;
  pDdiTable->pfnPartition = ur::level_zero_v2::urDevicePartition;
  pDdiTable->pfnSelectBinary = ur::level_zero_v2::urDeviceSelectBinary;
  pDdiTable->pfnGetNativeHandle = ur::level_zero_v2::urDeviceGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero_v2::urDeviceCreateWithNativeHandle;
  pDdiTable->pfnGetGlobalTimestamps =
      ur::level_zero_v2::urDeviceGetGlobalTimestamps;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetDeviceExpProcAddrTable(
    ur_api_version_t version, ur_device_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnWaitExp = ur::level_zero_v2::urDeviceWaitExp;

  return result;
}

#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO_V2
} // namespace ur::level_zero_v2
#else
} // extern "C"
#endif

namespace {
ur_result_t populateDdiTable(ur_dditable_t *ddi) {
  if (ddi == nullptr) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  ur_result_t result;

#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO_V2
#define NAMESPACE_ ::ur::level_zero_v2
#else
#define NAMESPACE_
#endif

  result = NAMESPACE_::urGetAdapterProcAddrTable(UR_API_VERSION_CURRENT,
                                                 &ddi->Adapter);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetBindlessImagesExpProcAddrTable(
      UR_API_VERSION_CURRENT, &ddi->BindlessImagesExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetCommandBufferExpProcAddrTable(
      UR_API_VERSION_CURRENT, &ddi->CommandBufferExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetContextProcAddrTable(UR_API_VERSION_CURRENT,
                                                 &ddi->Context);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetEnqueueProcAddrTable(UR_API_VERSION_CURRENT,
                                                 &ddi->Enqueue);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetEnqueueExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                    &ddi->EnqueueExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result =
      NAMESPACE_::urGetEventProcAddrTable(UR_API_VERSION_CURRENT, &ddi->Event);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetGraphExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                  &ddi->GraphExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetIPCExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                &ddi->IPCExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetKernelProcAddrTable(UR_API_VERSION_CURRENT,
                                                &ddi->Kernel);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetMemProcAddrTable(UR_API_VERSION_CURRENT, &ddi->Mem);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetMemoryExportExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                         &ddi->MemoryExportExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetPhysicalMemProcAddrTable(UR_API_VERSION_CURRENT,
                                                     &ddi->PhysicalMem);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetPlatformProcAddrTable(UR_API_VERSION_CURRENT,
                                                  &ddi->Platform);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetProgramProcAddrTable(UR_API_VERSION_CURRENT,
                                                 &ddi->Program);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetProgramExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                    &ddi->ProgramExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result =
      NAMESPACE_::urGetQueueProcAddrTable(UR_API_VERSION_CURRENT, &ddi->Queue);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetQueueExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                  &ddi->QueueExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetSamplerProcAddrTable(UR_API_VERSION_CURRENT,
                                                 &ddi->Sampler);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetUSMProcAddrTable(UR_API_VERSION_CURRENT, &ddi->USM);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetUSMExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                &ddi->USMExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetUsmP2PExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                   &ddi->UsmP2PExp);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetVirtualMemProcAddrTable(UR_API_VERSION_CURRENT,
                                                    &ddi->VirtualMem);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetDeviceProcAddrTable(UR_API_VERSION_CURRENT,
                                                &ddi->Device);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result = NAMESPACE_::urGetDeviceExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                   &ddi->DeviceExp);
  if (result != UR_RESULT_SUCCESS)
    return result;

#undef NAMESPACE_

  return result;
}
} // namespace

namespace ur::level_zero_v2 {
const ur_dditable_t *ddi_getter::value() {
  static std::once_flag flag;
  static ur_dditable_t table;

  std::call_once(flag, []() { populateDdiTable(&table); });
  return &table;
}

#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO_V2
ur_result_t urAdapterGetDdiTables(ur_dditable_t *ddi) {
  return populateDdiTable(ddi);
}
#endif
} // namespace ur::level_zero_v2
