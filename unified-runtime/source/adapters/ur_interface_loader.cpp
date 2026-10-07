//===--------- ur_interface_loader.cpp - Level Zero Adapter ------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <mutex>
#include <unified-runtime/ur_api.h>
#include <unified-runtime/ur_ddi.h>

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

#ifdef UR_STATIC_ADAPTER_OPENCL
namespace ur::opencl {
#else
extern "C" {
#endif

UR_APIEXPORT ur_result_t UR_APICALL urGetAdapterProcAddrTable(
    ur_api_version_t version, ur_adapter_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGet = urAdapterGet;
  pDdiTable->pfnRelease = urAdapterRelease;
  pDdiTable->pfnRetain = urAdapterRetain;
  pDdiTable->pfnGetLastError = urAdapterGetLastError;
  pDdiTable->pfnGetInfo = urAdapterGetInfo;
  pDdiTable->pfnSetLoggerCallback = urAdapterSetLoggerCallback;
  pDdiTable->pfnSetLoggerCallbackLevel = urAdapterSetLoggerCallbackLevel;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetBindlessImagesExpProcAddrTable(
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
  pDdiTable->pfnGetImageMemoryHandleTypeSupportExp =
      urBindlessImagesGetImageMemoryHandleTypeSupportExp;
  pDdiTable->pfnGetImageUnsampledHandleSupportExp =
      urBindlessImagesGetImageUnsampledHandleSupportExp;
  pDdiTable->pfnGetImageSampledHandleSupportExp =
      urBindlessImagesGetImageSampledHandleSupportExp;
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
  pDdiTable->pfnSupportsImportingHandleTypeExp =
      urBindlessImagesSupportsImportingHandleTypeExp;
  pDdiTable->pfnImportExternalSemaphoreExp =
      urBindlessImagesImportExternalSemaphoreExp;
  pDdiTable->pfnReleaseExternalSemaphoreExp =
      urBindlessImagesReleaseExternalSemaphoreExp;
  pDdiTable->pfnWaitExternalSemaphoreExp =
      urBindlessImagesWaitExternalSemaphoreExp;
  pDdiTable->pfnSignalExternalSemaphoreExp =
      urBindlessImagesSignalExternalSemaphoreExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetCommandBufferExpProcAddrTable(
    ur_api_version_t version, ur_command_buffer_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateExp = urCommandBufferCreateExp;
  pDdiTable->pfnRetainExp = urCommandBufferRetainExp;
  pDdiTable->pfnReleaseExp = urCommandBufferReleaseExp;
  pDdiTable->pfnFinalizeExp = urCommandBufferFinalizeExp;
  pDdiTable->pfnAppendKernelLaunchExp = urCommandBufferAppendKernelLaunchExp;
  pDdiTable->pfnAppendKernelLaunchWithArgsExp =
      urCommandBufferAppendKernelLaunchWithArgsExp;
  pDdiTable->pfnAppendUSMMemcpyExp = urCommandBufferAppendUSMMemcpyExp;
  pDdiTable->pfnAppendUSMFillExp = urCommandBufferAppendUSMFillExp;
  pDdiTable->pfnAppendMemBufferCopyExp = urCommandBufferAppendMemBufferCopyExp;
  pDdiTable->pfnAppendMemBufferWriteExp =
      urCommandBufferAppendMemBufferWriteExp;
  pDdiTable->pfnAppendMemBufferReadExp = urCommandBufferAppendMemBufferReadExp;
  pDdiTable->pfnAppendMemBufferCopyRectExp =
      urCommandBufferAppendMemBufferCopyRectExp;
  pDdiTable->pfnAppendMemBufferWriteRectExp =
      urCommandBufferAppendMemBufferWriteRectExp;
  pDdiTable->pfnAppendMemBufferReadRectExp =
      urCommandBufferAppendMemBufferReadRectExp;
  pDdiTable->pfnAppendMemBufferFillExp = urCommandBufferAppendMemBufferFillExp;
  pDdiTable->pfnAppendUSMPrefetchExp = urCommandBufferAppendUSMPrefetchExp;
  pDdiTable->pfnAppendUSMAdviseExp = urCommandBufferAppendUSMAdviseExp;
  pDdiTable->pfnAppendNativeCommandExp = urCommandBufferAppendNativeCommandExp;
  pDdiTable->pfnUpdateKernelLaunchExp = urCommandBufferUpdateKernelLaunchExp;
  pDdiTable->pfnUpdateSignalEventExp = urCommandBufferUpdateSignalEventExp;
  pDdiTable->pfnUpdateWaitEventsExp = urCommandBufferUpdateWaitEventsExp;
  pDdiTable->pfnGetInfoExp = urCommandBufferGetInfoExp;
  pDdiTable->pfnGetNativeHandleExp = urCommandBufferGetNativeHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetContextProcAddrTable(
    ur_api_version_t version, ur_context_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = urContextCreate;
  pDdiTable->pfnRetain = urContextRetain;
  pDdiTable->pfnRelease = urContextRelease;
  pDdiTable->pfnGetInfo = urContextGetInfo;
  pDdiTable->pfnGetNativeHandle = urContextGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urContextCreateWithNativeHandle;
  pDdiTable->pfnSetExtendedDeleter = urContextSetExtendedDeleter;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    ur_api_version_t version, ur_enqueue_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

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
  pDdiTable->pfnReadHostPipe = urEnqueueReadHostPipe;
  pDdiTable->pfnWriteHostPipe = urEnqueueWriteHostPipe;
  pDdiTable->pfnEventsWaitWithBarrierExt = urEnqueueEventsWaitWithBarrierExt;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    ur_api_version_t version, ur_enqueue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnKernelLaunchWithArgsExp = urEnqueueKernelLaunchWithArgsExp;
  pDdiTable->pfnUSMDeviceAllocExp = urEnqueueUSMDeviceAllocExp;
  pDdiTable->pfnUSMSharedAllocExp = urEnqueueUSMSharedAllocExp;
  pDdiTable->pfnUSMHostAllocExp = urEnqueueUSMHostAllocExp;
  pDdiTable->pfnUSMFreeExp = urEnqueueUSMFreeExp;
  pDdiTable->pfnTimestampRecordingExp = urEnqueueTimestampRecordingExp;
  pDdiTable->pfnCommandBufferExp = urEnqueueCommandBufferExp;
  pDdiTable->pfnHostTaskExp = urEnqueueHostTaskExp;
  pDdiTable->pfnNativeCommandExp = urEnqueueNativeCommandExp;
  pDdiTable->pfnGraphExp = urEnqueueGraphExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEventProcAddrTable(
    ur_api_version_t version, ur_event_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetInfo = urEventGetInfo;
  pDdiTable->pfnGetProfilingInfo = urEventGetProfilingInfo;
  pDdiTable->pfnWait = urEventWait;
  pDdiTable->pfnRetain = urEventRetain;
  pDdiTable->pfnRelease = urEventRelease;
  pDdiTable->pfnGetNativeHandle = urEventGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urEventCreateWithNativeHandle;
  pDdiTable->pfnSetCallback = urEventSetCallback;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEventExpProcAddrTable(
    ur_api_version_t version, ur_event_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateExp = urEventCreateExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetGraphExpProcAddrTable(
    ur_api_version_t version, ur_graph_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateExp = urGraphCreateExp;
  pDdiTable->pfnInstantiateGraphExp = urGraphInstantiateGraphExp;
  pDdiTable->pfnDestroyExp = urGraphDestroyExp;
  pDdiTable->pfnExecutableGraphDestroyExp = urGraphExecutableGraphDestroyExp;
  pDdiTable->pfnIsEmptyExp = urGraphIsEmptyExp;
  pDdiTable->pfnGetIdExp = urGraphGetIdExp;
  pDdiTable->pfnSetDestructionCallbackExp = urGraphSetDestructionCallbackExp;
  pDdiTable->pfnDumpContentsExp = urGraphDumpContentsExp;
  pDdiTable->pfnGetNativeHandleExp = urGraphGetNativeHandleExp;
  pDdiTable->pfnExecutableGraphGetNativeHandleExp =
      urGraphExecutableGraphGetNativeHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetIPCExpProcAddrTable(
    ur_api_version_t version, ur_ipc_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetMemHandleExp = urIPCGetMemHandleExp;
  pDdiTable->pfnPutMemHandleExp = urIPCPutMemHandleExp;
  pDdiTable->pfnOpenMemHandleExp = urIPCOpenMemHandleExp;
  pDdiTable->pfnCloseMemHandleExp = urIPCCloseMemHandleExp;
  pDdiTable->pfnGetPhysMemHandleExp = urIPCGetPhysMemHandleExp;
  pDdiTable->pfnPutPhysMemHandleExp = urIPCPutPhysMemHandleExp;
  pDdiTable->pfnOpenPhysMemHandleExp = urIPCOpenPhysMemHandleExp;
  pDdiTable->pfnClosePhysMemHandleExp = urIPCClosePhysMemHandleExp;
  pDdiTable->pfnGetEventHandleExp = urIPCGetEventHandleExp;
  pDdiTable->pfnPutEventHandleExp = urIPCPutEventHandleExp;
  pDdiTable->pfnOpenEventHandleExp = urIPCOpenEventHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetKernelProcAddrTable(
    ur_api_version_t version, ur_kernel_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = urKernelCreate;
  pDdiTable->pfnGetInfo = urKernelGetInfo;
  pDdiTable->pfnGetGroupInfo = urKernelGetGroupInfo;
  pDdiTable->pfnGetSubGroupInfo = urKernelGetSubGroupInfo;
  pDdiTable->pfnRetain = urKernelRetain;
  pDdiTable->pfnRelease = urKernelRelease;
  pDdiTable->pfnGetNativeHandle = urKernelGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urKernelCreateWithNativeHandle;
  pDdiTable->pfnGetSuggestedLocalWorkSize = urKernelGetSuggestedLocalWorkSize;
  pDdiTable->pfnGetSuggestedLocalWorkSizeWithArgs =
      urKernelGetSuggestedLocalWorkSizeWithArgs;
  pDdiTable->pfnSetExecInfo = urKernelSetExecInfo;
  pDdiTable->pfnSetSpecializationConstants = urKernelSetSpecializationConstants;
  pDdiTable->pfnSuggestMaxCooperativeGroupCount =
      urKernelSuggestMaxCooperativeGroupCount;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGetMemProcAddrTable(ur_api_version_t version, ur_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
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

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetMemoryExportExpProcAddrTable(
    ur_api_version_t version, ur_memory_export_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnAllocExportableMemoryExp =
      urMemoryExportAllocExportableMemoryExp;
  pDdiTable->pfnFreeExportableMemoryExp = urMemoryExportFreeExportableMemoryExp;
  pDdiTable->pfnExportMemoryHandleExp = urMemoryExportExportMemoryHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    ur_api_version_t version, ur_physical_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = urPhysicalMemCreate;
  pDdiTable->pfnRetain = urPhysicalMemRetain;
  pDdiTable->pfnRelease = urPhysicalMemRelease;
  pDdiTable->pfnGetInfo = urPhysicalMemGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    ur_api_version_t version, ur_platform_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGet = urPlatformGet;
  pDdiTable->pfnGetInfo = urPlatformGetInfo;
  pDdiTable->pfnGetNativeHandle = urPlatformGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urPlatformCreateWithNativeHandle;
  pDdiTable->pfnGetApiVersion = urPlatformGetApiVersion;
  pDdiTable->pfnGetBackendOption = urPlatformGetBackendOption;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetProgramProcAddrTable(
    ur_api_version_t version, ur_program_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateWithIL = urProgramCreateWithIL;
  pDdiTable->pfnCreateWithBinary = urProgramCreateWithBinary;
  pDdiTable->pfnBuild = urProgramBuild;
  pDdiTable->pfnCompile = urProgramCompile;
  pDdiTable->pfnLink = urProgramLink;
  pDdiTable->pfnRetain = urProgramRetain;
  pDdiTable->pfnRelease = urProgramRelease;
  pDdiTable->pfnGetFunctionPointer = urProgramGetFunctionPointer;
  pDdiTable->pfnGetGlobalVariablePointer = urProgramGetGlobalVariablePointer;
  pDdiTable->pfnGetInfo = urProgramGetInfo;
  pDdiTable->pfnGetBuildInfo = urProgramGetBuildInfo;
  pDdiTable->pfnSetSpecializationConstants =
      urProgramSetSpecializationConstants;
  pDdiTable->pfnGetNativeHandle = urProgramGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urProgramCreateWithNativeHandle;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetProgramExpProcAddrTable(
    ur_api_version_t version, ur_program_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnDynamicLinkExp = urProgramDynamicLinkExp;
  pDdiTable->pfnBuildExp = urProgramBuildExp;
  pDdiTable->pfnCompileExp = urProgramCompileExp;
  pDdiTable->pfnLinkExp = urProgramLinkExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetQueueProcAddrTable(
    ur_api_version_t version, ur_queue_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetInfo = urQueueGetInfo;
  pDdiTable->pfnCreate = urQueueCreate;
  pDdiTable->pfnRetain = urQueueRetain;
  pDdiTable->pfnRelease = urQueueRelease;
  pDdiTable->pfnGetNativeHandle = urQueueGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urQueueCreateWithNativeHandle;
  pDdiTable->pfnFinish = urQueueFinish;
  pDdiTable->pfnFlush = urQueueFlush;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetQueueExpProcAddrTable(
    ur_api_version_t version, ur_queue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnBeginGraphCaptureExp = urQueueBeginGraphCaptureExp;
  pDdiTable->pfnBeginCaptureIntoGraphExp = urQueueBeginCaptureIntoGraphExp;
  pDdiTable->pfnEndGraphCaptureExp = urQueueEndGraphCaptureExp;
  pDdiTable->pfnIsGraphCaptureEnabledExp = urQueueIsGraphCaptureEnabledExp;
  pDdiTable->pfnGetGraphExp = urQueueGetGraphExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    ur_api_version_t version, ur_sampler_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = urSamplerCreate;
  pDdiTable->pfnRetain = urSamplerRetain;
  pDdiTable->pfnRelease = urSamplerRelease;
  pDdiTable->pfnGetInfo = urSamplerGetInfo;
  pDdiTable->pfnGetNativeHandle = urSamplerGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle = urSamplerCreateWithNativeHandle;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGetUSMProcAddrTable(ur_api_version_t version, ur_usm_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
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

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetUSMExpProcAddrTable(
    ur_api_version_t version, ur_usm_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnPoolCreateExp = urUSMPoolCreateExp;
  pDdiTable->pfnPoolDestroyExp = urUSMPoolDestroyExp;
  pDdiTable->pfnPoolGetDefaultDevicePoolExp = urUSMPoolGetDefaultDevicePoolExp;
  pDdiTable->pfnPoolGetInfoExp = urUSMPoolGetInfoExp;
  pDdiTable->pfnPoolSetInfoExp = urUSMPoolSetInfoExp;
  pDdiTable->pfnPoolSetDevicePoolExp = urUSMPoolSetDevicePoolExp;
  pDdiTable->pfnPoolGetDevicePoolExp = urUSMPoolGetDevicePoolExp;
  pDdiTable->pfnPoolTrimToExp = urUSMPoolTrimToExp;
  pDdiTable->pfnPitchedAllocExp = urUSMPitchedAllocExp;
  pDdiTable->pfnContextMemcpyExp = urUSMContextMemcpyExp;
  pDdiTable->pfnHostAllocUnregisterExp = urUSMHostAllocUnregisterExp;
  pDdiTable->pfnHostAllocRegisterExp = urUSMHostAllocRegisterExp;
  pDdiTable->pfnImportExp = urUSMImportExp;
  pDdiTable->pfnReleaseExp = urUSMReleaseExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetUsmP2PExpProcAddrTable(
    ur_api_version_t version, ur_usm_p2p_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnEnablePeerAccessExp = urUsmP2PEnablePeerAccessExp;
  pDdiTable->pfnDisablePeerAccessExp = urUsmP2PDisablePeerAccessExp;
  pDdiTable->pfnPeerAccessGetInfoExp = urUsmP2PPeerAccessGetInfoExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetVirtualMemProcAddrTable(
    ur_api_version_t version, ur_virtual_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGranularityGetInfo = urVirtualMemGranularityGetInfo;
  pDdiTable->pfnReserve = urVirtualMemReserve;
  pDdiTable->pfnFree = urVirtualMemFree;
  pDdiTable->pfnMap = urVirtualMemMap;
  pDdiTable->pfnUnmap = urVirtualMemUnmap;
  pDdiTable->pfnSetAccess = urVirtualMemSetAccess;
  pDdiTable->pfnGetInfo = urVirtualMemGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    ur_api_version_t version, ur_device_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
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

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetDeviceExpProcAddrTable(
    ur_api_version_t version, ur_device_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnWaitExp = urDeviceWaitExp;

  return result;
}

#ifdef UR_STATIC_ADAPTER_OPENCL
} // namespace ur::opencl
#else
} // extern "C"
#endif

namespace {
ur_result_t populateDdiTable(ur_dditable_t *ddi) {
  if (ddi == nullptr) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  ur_result_t result;

#ifdef UR_STATIC_ADAPTER_OPENCL
#define NAMESPACE_ ::ur::opencl
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
  result = NAMESPACE_::urGetEventExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                  &ddi->EventExp);
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

namespace ur::opencl {
const ur_dditable_t *ddi_getter::value() {
  static std::once_flag flag;
  static ur_dditable_t table;

  std::call_once(flag, []() { populateDdiTable(&table); });
  return &table;
}

#ifdef UR_STATIC_ADAPTER_OPENCL
ur_result_t urAdapterGetDdiTables(ur_dditable_t *ddi) {
  return populateDdiTable(ddi);
}
#endif
} // namespace ur::opencl
