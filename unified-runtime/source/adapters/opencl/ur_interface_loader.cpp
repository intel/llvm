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

  pDdiTable->pfnGet = ur::opencl::urAdapterGet;
  pDdiTable->pfnRelease = ur::opencl::urAdapterRelease;
  pDdiTable->pfnRetain = ur::opencl::urAdapterRetain;
  pDdiTable->pfnGetLastError = ur::opencl::urAdapterGetLastError;
  pDdiTable->pfnGetInfo = ur::opencl::urAdapterGetInfo;
  pDdiTable->pfnSetLoggerCallback = ur::opencl::urAdapterSetLoggerCallback;
  pDdiTable->pfnSetLoggerCallbackLevel =
      ur::opencl::urAdapterSetLoggerCallbackLevel;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetBindlessImagesExpProcAddrTable(
    ur_api_version_t version, ur_bindless_images_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnUnsampledImageHandleDestroyExp =
      ur::opencl::urBindlessImagesUnsampledImageHandleDestroyExp;
  pDdiTable->pfnSampledImageHandleDestroyExp =
      ur::opencl::urBindlessImagesSampledImageHandleDestroyExp;
  pDdiTable->pfnImageAllocateExp = ur::opencl::urBindlessImagesImageAllocateExp;
  pDdiTable->pfnImageFreeExp = ur::opencl::urBindlessImagesImageFreeExp;
  pDdiTable->pfnUnsampledImageCreateExp =
      ur::opencl::urBindlessImagesUnsampledImageCreateExp;
  pDdiTable->pfnSampledImageCreateExp =
      ur::opencl::urBindlessImagesSampledImageCreateExp;
  pDdiTable->pfnImageCopyExp = ur::opencl::urBindlessImagesImageCopyExp;
  pDdiTable->pfnImageGetInfoExp = ur::opencl::urBindlessImagesImageGetInfoExp;
  pDdiTable->pfnGetImageMemoryHandleTypeSupportExp =
      ur::opencl::urBindlessImagesGetImageMemoryHandleTypeSupportExp;
  pDdiTable->pfnGetImageUnsampledHandleSupportExp =
      ur::opencl::urBindlessImagesGetImageUnsampledHandleSupportExp;
  pDdiTable->pfnGetImageSampledHandleSupportExp =
      ur::opencl::urBindlessImagesGetImageSampledHandleSupportExp;
  pDdiTable->pfnMipmapGetLevelExp =
      ur::opencl::urBindlessImagesMipmapGetLevelExp;
  pDdiTable->pfnMipmapFreeExp = ur::opencl::urBindlessImagesMipmapFreeExp;
  pDdiTable->pfnImportExternalMemoryExp =
      ur::opencl::urBindlessImagesImportExternalMemoryExp;
  pDdiTable->pfnMapExternalArrayExp =
      ur::opencl::urBindlessImagesMapExternalArrayExp;
  pDdiTable->pfnMapExternalLinearMemoryExp =
      ur::opencl::urBindlessImagesMapExternalLinearMemoryExp;
  pDdiTable->pfnReleaseExternalMemoryExp =
      ur::opencl::urBindlessImagesReleaseExternalMemoryExp;
  pDdiTable->pfnFreeMappedLinearMemoryExp =
      ur::opencl::urBindlessImagesFreeMappedLinearMemoryExp;
  pDdiTable->pfnSupportsImportingHandleTypeExp =
      ur::opencl::urBindlessImagesSupportsImportingHandleTypeExp;
  pDdiTable->pfnImportExternalSemaphoreExp =
      ur::opencl::urBindlessImagesImportExternalSemaphoreExp;
  pDdiTable->pfnReleaseExternalSemaphoreExp =
      ur::opencl::urBindlessImagesReleaseExternalSemaphoreExp;
  pDdiTable->pfnWaitExternalSemaphoreExp =
      ur::opencl::urBindlessImagesWaitExternalSemaphoreExp;
  pDdiTable->pfnSignalExternalSemaphoreExp =
      ur::opencl::urBindlessImagesSignalExternalSemaphoreExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetCommandBufferExpProcAddrTable(
    ur_api_version_t version, ur_command_buffer_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateExp = ur::opencl::urCommandBufferCreateExp;
  pDdiTable->pfnRetainExp = ur::opencl::urCommandBufferRetainExp;
  pDdiTable->pfnReleaseExp = ur::opencl::urCommandBufferReleaseExp;
  pDdiTable->pfnFinalizeExp = ur::opencl::urCommandBufferFinalizeExp;
  pDdiTable->pfnAppendKernelLaunchExp =
      ur::opencl::urCommandBufferAppendKernelLaunchExp;
  pDdiTable->pfnAppendKernelLaunchWithArgsExp =
      ur::opencl::urCommandBufferAppendKernelLaunchWithArgsExp;
  pDdiTable->pfnAppendUSMMemcpyExp =
      ur::opencl::urCommandBufferAppendUSMMemcpyExp;
  pDdiTable->pfnAppendUSMFillExp = ur::opencl::urCommandBufferAppendUSMFillExp;
  pDdiTable->pfnAppendMemBufferCopyExp =
      ur::opencl::urCommandBufferAppendMemBufferCopyExp;
  pDdiTable->pfnAppendMemBufferWriteExp =
      ur::opencl::urCommandBufferAppendMemBufferWriteExp;
  pDdiTable->pfnAppendMemBufferReadExp =
      ur::opencl::urCommandBufferAppendMemBufferReadExp;
  pDdiTable->pfnAppendMemBufferCopyRectExp =
      ur::opencl::urCommandBufferAppendMemBufferCopyRectExp;
  pDdiTable->pfnAppendMemBufferWriteRectExp =
      ur::opencl::urCommandBufferAppendMemBufferWriteRectExp;
  pDdiTable->pfnAppendMemBufferReadRectExp =
      ur::opencl::urCommandBufferAppendMemBufferReadRectExp;
  pDdiTable->pfnAppendMemBufferFillExp =
      ur::opencl::urCommandBufferAppendMemBufferFillExp;
  pDdiTable->pfnAppendUSMPrefetchExp =
      ur::opencl::urCommandBufferAppendUSMPrefetchExp;
  pDdiTable->pfnAppendUSMAdviseExp =
      ur::opencl::urCommandBufferAppendUSMAdviseExp;
  pDdiTable->pfnAppendNativeCommandExp =
      ur::opencl::urCommandBufferAppendNativeCommandExp;
  pDdiTable->pfnUpdateKernelLaunchExp =
      ur::opencl::urCommandBufferUpdateKernelLaunchExp;
  pDdiTable->pfnUpdateSignalEventExp =
      ur::opencl::urCommandBufferUpdateSignalEventExp;
  pDdiTable->pfnUpdateWaitEventsExp =
      ur::opencl::urCommandBufferUpdateWaitEventsExp;
  pDdiTable->pfnGetInfoExp = ur::opencl::urCommandBufferGetInfoExp;
  pDdiTable->pfnGetNativeHandleExp =
      ur::opencl::urCommandBufferGetNativeHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetContextProcAddrTable(
    ur_api_version_t version, ur_context_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::opencl::urContextCreate;
  pDdiTable->pfnRetain = ur::opencl::urContextRetain;
  pDdiTable->pfnRelease = ur::opencl::urContextRelease;
  pDdiTable->pfnGetInfo = ur::opencl::urContextGetInfo;
  pDdiTable->pfnGetNativeHandle = ur::opencl::urContextGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::opencl::urContextCreateWithNativeHandle;
  pDdiTable->pfnSetExtendedDeleter = ur::opencl::urContextSetExtendedDeleter;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    ur_api_version_t version, ur_enqueue_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnEventsWait = ur::opencl::urEnqueueEventsWait;
  pDdiTable->pfnEventsWaitWithBarrier =
      ur::opencl::urEnqueueEventsWaitWithBarrier;
  pDdiTable->pfnMemBufferRead = ur::opencl::urEnqueueMemBufferRead;
  pDdiTable->pfnMemBufferWrite = ur::opencl::urEnqueueMemBufferWrite;
  pDdiTable->pfnMemBufferReadRect = ur::opencl::urEnqueueMemBufferReadRect;
  pDdiTable->pfnMemBufferWriteRect = ur::opencl::urEnqueueMemBufferWriteRect;
  pDdiTable->pfnMemBufferCopy = ur::opencl::urEnqueueMemBufferCopy;
  pDdiTable->pfnMemBufferCopyRect = ur::opencl::urEnqueueMemBufferCopyRect;
  pDdiTable->pfnMemBufferFill = ur::opencl::urEnqueueMemBufferFill;
  pDdiTable->pfnMemImageRead = ur::opencl::urEnqueueMemImageRead;
  pDdiTable->pfnMemImageWrite = ur::opencl::urEnqueueMemImageWrite;
  pDdiTable->pfnMemImageCopy = ur::opencl::urEnqueueMemImageCopy;
  pDdiTable->pfnMemBufferMap = ur::opencl::urEnqueueMemBufferMap;
  pDdiTable->pfnMemUnmap = ur::opencl::urEnqueueMemUnmap;
  pDdiTable->pfnUSMFill = ur::opencl::urEnqueueUSMFill;
  pDdiTable->pfnUSMMemcpy = ur::opencl::urEnqueueUSMMemcpy;
  pDdiTable->pfnUSMPrefetch = ur::opencl::urEnqueueUSMPrefetch;
  pDdiTable->pfnUSMAdvise = ur::opencl::urEnqueueUSMAdvise;
  pDdiTable->pfnUSMFill2D = ur::opencl::urEnqueueUSMFill2D;
  pDdiTable->pfnUSMMemcpy2D = ur::opencl::urEnqueueUSMMemcpy2D;
  pDdiTable->pfnDeviceGlobalVariableWrite =
      ur::opencl::urEnqueueDeviceGlobalVariableWrite;
  pDdiTable->pfnDeviceGlobalVariableRead =
      ur::opencl::urEnqueueDeviceGlobalVariableRead;
  pDdiTable->pfnReadHostPipe = ur::opencl::urEnqueueReadHostPipe;
  pDdiTable->pfnWriteHostPipe = ur::opencl::urEnqueueWriteHostPipe;
  pDdiTable->pfnEventsWaitWithBarrierExt =
      ur::opencl::urEnqueueEventsWaitWithBarrierExt;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    ur_api_version_t version, ur_enqueue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnKernelLaunchWithArgsExp =
      ur::opencl::urEnqueueKernelLaunchWithArgsExp;
  pDdiTable->pfnUSMDeviceAllocExp = ur::opencl::urEnqueueUSMDeviceAllocExp;
  pDdiTable->pfnUSMSharedAllocExp = ur::opencl::urEnqueueUSMSharedAllocExp;
  pDdiTable->pfnUSMHostAllocExp = ur::opencl::urEnqueueUSMHostAllocExp;
  pDdiTable->pfnUSMFreeExp = ur::opencl::urEnqueueUSMFreeExp;
  pDdiTable->pfnTimestampRecordingExp =
      ur::opencl::urEnqueueTimestampRecordingExp;
  pDdiTable->pfnCommandBufferExp = ur::opencl::urEnqueueCommandBufferExp;
  pDdiTable->pfnHostTaskExp = ur::opencl::urEnqueueHostTaskExp;
  pDdiTable->pfnNativeCommandExp = ur::opencl::urEnqueueNativeCommandExp;
  pDdiTable->pfnGraphExp = ur::opencl::urEnqueueGraphExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEventProcAddrTable(
    ur_api_version_t version, ur_event_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetInfo = ur::opencl::urEventGetInfo;
  pDdiTable->pfnGetProfilingInfo = ur::opencl::urEventGetProfilingInfo;
  pDdiTable->pfnWait = ur::opencl::urEventWait;
  pDdiTable->pfnRetain = ur::opencl::urEventRetain;
  pDdiTable->pfnRelease = ur::opencl::urEventRelease;
  pDdiTable->pfnGetNativeHandle = ur::opencl::urEventGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::opencl::urEventCreateWithNativeHandle;
  pDdiTable->pfnSetCallback = ur::opencl::urEventSetCallback;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEventExpProcAddrTable(
    ur_api_version_t version, ur_event_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateExp = ur::opencl::urEventCreateExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetGraphExpProcAddrTable(
    ur_api_version_t version, ur_graph_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateExp = ur::opencl::urGraphCreateExp;
  pDdiTable->pfnInstantiateGraphExp = ur::opencl::urGraphInstantiateGraphExp;
  pDdiTable->pfnDestroyExp = ur::opencl::urGraphDestroyExp;
  pDdiTable->pfnExecutableGraphDestroyExp =
      ur::opencl::urGraphExecutableGraphDestroyExp;
  pDdiTable->pfnIsEmptyExp = ur::opencl::urGraphIsEmptyExp;
  pDdiTable->pfnSetDestructionCallbackExp =
      ur::opencl::urGraphSetDestructionCallbackExp;
  pDdiTable->pfnDumpContentsExp = ur::opencl::urGraphDumpContentsExp;
  pDdiTable->pfnGetNativeHandleExp = ur::opencl::urGraphGetNativeHandleExp;
  pDdiTable->pfnExecutableGraphGetNativeHandleExp =
      ur::opencl::urGraphExecutableGraphGetNativeHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetIPCExpProcAddrTable(
    ur_api_version_t version, ur_ipc_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetMemHandleExp = ur::opencl::urIPCGetMemHandleExp;
  pDdiTable->pfnPutMemHandleExp = ur::opencl::urIPCPutMemHandleExp;
  pDdiTable->pfnOpenMemHandleExp = ur::opencl::urIPCOpenMemHandleExp;
  pDdiTable->pfnCloseMemHandleExp = ur::opencl::urIPCCloseMemHandleExp;
  pDdiTable->pfnGetPhysMemHandleExp = ur::opencl::urIPCGetPhysMemHandleExp;
  pDdiTable->pfnPutPhysMemHandleExp = ur::opencl::urIPCPutPhysMemHandleExp;
  pDdiTable->pfnOpenPhysMemHandleExp = ur::opencl::urIPCOpenPhysMemHandleExp;
  pDdiTable->pfnClosePhysMemHandleExp = ur::opencl::urIPCClosePhysMemHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetKernelProcAddrTable(
    ur_api_version_t version, ur_kernel_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::opencl::urKernelCreate;
  pDdiTable->pfnGetInfo = ur::opencl::urKernelGetInfo;
  pDdiTable->pfnGetGroupInfo = ur::opencl::urKernelGetGroupInfo;
  pDdiTable->pfnGetSubGroupInfo = ur::opencl::urKernelGetSubGroupInfo;
  pDdiTable->pfnRetain = ur::opencl::urKernelRetain;
  pDdiTable->pfnRelease = ur::opencl::urKernelRelease;
  pDdiTable->pfnGetNativeHandle = ur::opencl::urKernelGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::opencl::urKernelCreateWithNativeHandle;
  pDdiTable->pfnGetSuggestedLocalWorkSize =
      ur::opencl::urKernelGetSuggestedLocalWorkSize;
  pDdiTable->pfnGetSuggestedLocalWorkSizeWithArgs =
      ur::opencl::urKernelGetSuggestedLocalWorkSizeWithArgs;
  pDdiTable->pfnSetExecInfo = ur::opencl::urKernelSetExecInfo;
  pDdiTable->pfnSetSpecializationConstants =
      ur::opencl::urKernelSetSpecializationConstants;
  pDdiTable->pfnSuggestMaxCooperativeGroupCount =
      ur::opencl::urKernelSuggestMaxCooperativeGroupCount;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGetMemProcAddrTable(ur_api_version_t version, ur_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnImageCreate = ur::opencl::urMemImageCreate;
  pDdiTable->pfnBufferCreate = ur::opencl::urMemBufferCreate;
  pDdiTable->pfnRetain = ur::opencl::urMemRetain;
  pDdiTable->pfnRelease = ur::opencl::urMemRelease;
  pDdiTable->pfnBufferPartition = ur::opencl::urMemBufferPartition;
  pDdiTable->pfnGetNativeHandle = ur::opencl::urMemGetNativeHandle;
  pDdiTable->pfnBufferCreateWithNativeHandle =
      ur::opencl::urMemBufferCreateWithNativeHandle;
  pDdiTable->pfnImageCreateWithNativeHandle =
      ur::opencl::urMemImageCreateWithNativeHandle;
  pDdiTable->pfnGetInfo = ur::opencl::urMemGetInfo;
  pDdiTable->pfnImageGetInfo = ur::opencl::urMemImageGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetMemoryExportExpProcAddrTable(
    ur_api_version_t version, ur_memory_export_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnAllocExportableMemoryExp =
      ur::opencl::urMemoryExportAllocExportableMemoryExp;
  pDdiTable->pfnFreeExportableMemoryExp =
      ur::opencl::urMemoryExportFreeExportableMemoryExp;
  pDdiTable->pfnExportMemoryHandleExp =
      ur::opencl::urMemoryExportExportMemoryHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    ur_api_version_t version, ur_physical_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::opencl::urPhysicalMemCreate;
  pDdiTable->pfnRetain = ur::opencl::urPhysicalMemRetain;
  pDdiTable->pfnRelease = ur::opencl::urPhysicalMemRelease;
  pDdiTable->pfnGetInfo = ur::opencl::urPhysicalMemGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    ur_api_version_t version, ur_platform_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGet = ur::opencl::urPlatformGet;
  pDdiTable->pfnGetInfo = ur::opencl::urPlatformGetInfo;
  pDdiTable->pfnGetNativeHandle = ur::opencl::urPlatformGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::opencl::urPlatformCreateWithNativeHandle;
  pDdiTable->pfnGetApiVersion = ur::opencl::urPlatformGetApiVersion;
  pDdiTable->pfnGetBackendOption = ur::opencl::urPlatformGetBackendOption;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetProgramProcAddrTable(
    ur_api_version_t version, ur_program_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateWithIL = ur::opencl::urProgramCreateWithIL;
  pDdiTable->pfnCreateWithBinary = ur::opencl::urProgramCreateWithBinary;
  pDdiTable->pfnBuild = ur::opencl::urProgramBuild;
  pDdiTable->pfnCompile = ur::opencl::urProgramCompile;
  pDdiTable->pfnLink = ur::opencl::urProgramLink;
  pDdiTable->pfnRetain = ur::opencl::urProgramRetain;
  pDdiTable->pfnRelease = ur::opencl::urProgramRelease;
  pDdiTable->pfnGetFunctionPointer = ur::opencl::urProgramGetFunctionPointer;
  pDdiTable->pfnGetGlobalVariablePointer =
      ur::opencl::urProgramGetGlobalVariablePointer;
  pDdiTable->pfnGetInfo = ur::opencl::urProgramGetInfo;
  pDdiTable->pfnGetBuildInfo = ur::opencl::urProgramGetBuildInfo;
  pDdiTable->pfnSetSpecializationConstants =
      ur::opencl::urProgramSetSpecializationConstants;
  pDdiTable->pfnGetNativeHandle = ur::opencl::urProgramGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::opencl::urProgramCreateWithNativeHandle;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetProgramExpProcAddrTable(
    ur_api_version_t version, ur_program_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnDynamicLinkExp = ur::opencl::urProgramDynamicLinkExp;
  pDdiTable->pfnBuildExp = ur::opencl::urProgramBuildExp;
  pDdiTable->pfnCompileExp = ur::opencl::urProgramCompileExp;
  pDdiTable->pfnLinkExp = ur::opencl::urProgramLinkExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetQueueProcAddrTable(
    ur_api_version_t version, ur_queue_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetInfo = ur::opencl::urQueueGetInfo;
  pDdiTable->pfnCreate = ur::opencl::urQueueCreate;
  pDdiTable->pfnRetain = ur::opencl::urQueueRetain;
  pDdiTable->pfnRelease = ur::opencl::urQueueRelease;
  pDdiTable->pfnGetNativeHandle = ur::opencl::urQueueGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::opencl::urQueueCreateWithNativeHandle;
  pDdiTable->pfnFinish = ur::opencl::urQueueFinish;
  pDdiTable->pfnFlush = ur::opencl::urQueueFlush;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetQueueExpProcAddrTable(
    ur_api_version_t version, ur_queue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnBeginGraphCaptureExp = ur::opencl::urQueueBeginGraphCaptureExp;
  pDdiTable->pfnBeginCaptureIntoGraphExp =
      ur::opencl::urQueueBeginCaptureIntoGraphExp;
  pDdiTable->pfnEndGraphCaptureExp = ur::opencl::urQueueEndGraphCaptureExp;
  pDdiTable->pfnIsGraphCaptureEnabledExp =
      ur::opencl::urQueueIsGraphCaptureEnabledExp;
  pDdiTable->pfnGetGraphExp = ur::opencl::urQueueGetGraphExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    ur_api_version_t version, ur_sampler_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::opencl::urSamplerCreate;
  pDdiTable->pfnRetain = ur::opencl::urSamplerRetain;
  pDdiTable->pfnRelease = ur::opencl::urSamplerRelease;
  pDdiTable->pfnGetInfo = ur::opencl::urSamplerGetInfo;
  pDdiTable->pfnGetNativeHandle = ur::opencl::urSamplerGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::opencl::urSamplerCreateWithNativeHandle;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGetUSMProcAddrTable(ur_api_version_t version, ur_usm_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnHostAlloc = ur::opencl::urUSMHostAlloc;
  pDdiTable->pfnDeviceAlloc = ur::opencl::urUSMDeviceAlloc;
  pDdiTable->pfnSharedAlloc = ur::opencl::urUSMSharedAlloc;
  pDdiTable->pfnFree = ur::opencl::urUSMFree;
  pDdiTable->pfnGetMemAllocInfo = ur::opencl::urUSMGetMemAllocInfo;
  pDdiTable->pfnPoolCreate = ur::opencl::urUSMPoolCreate;
  pDdiTable->pfnPoolRetain = ur::opencl::urUSMPoolRetain;
  pDdiTable->pfnPoolRelease = ur::opencl::urUSMPoolRelease;
  pDdiTable->pfnPoolGetInfo = ur::opencl::urUSMPoolGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetUSMExpProcAddrTable(
    ur_api_version_t version, ur_usm_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnPoolCreateExp = ur::opencl::urUSMPoolCreateExp;
  pDdiTable->pfnPoolDestroyExp = ur::opencl::urUSMPoolDestroyExp;
  pDdiTable->pfnPoolGetDefaultDevicePoolExp =
      ur::opencl::urUSMPoolGetDefaultDevicePoolExp;
  pDdiTable->pfnPoolGetInfoExp = ur::opencl::urUSMPoolGetInfoExp;
  pDdiTable->pfnPoolSetInfoExp = ur::opencl::urUSMPoolSetInfoExp;
  pDdiTable->pfnPoolSetDevicePoolExp = ur::opencl::urUSMPoolSetDevicePoolExp;
  pDdiTable->pfnPoolGetDevicePoolExp = ur::opencl::urUSMPoolGetDevicePoolExp;
  pDdiTable->pfnPoolTrimToExp = ur::opencl::urUSMPoolTrimToExp;
  pDdiTable->pfnPitchedAllocExp = ur::opencl::urUSMPitchedAllocExp;
  pDdiTable->pfnContextMemcpyExp = ur::opencl::urUSMContextMemcpyExp;
  pDdiTable->pfnHostAllocUnregisterExp =
      ur::opencl::urUSMHostAllocUnregisterExp;
  pDdiTable->pfnHostAllocRegisterExp = ur::opencl::urUSMHostAllocRegisterExp;
  pDdiTable->pfnImportExp = ur::opencl::urUSMImportExp;
  pDdiTable->pfnReleaseExp = ur::opencl::urUSMReleaseExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetUsmP2PExpProcAddrTable(
    ur_api_version_t version, ur_usm_p2p_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnEnablePeerAccessExp = ur::opencl::urUsmP2PEnablePeerAccessExp;
  pDdiTable->pfnDisablePeerAccessExp = ur::opencl::urUsmP2PDisablePeerAccessExp;
  pDdiTable->pfnPeerAccessGetInfoExp = ur::opencl::urUsmP2PPeerAccessGetInfoExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetVirtualMemProcAddrTable(
    ur_api_version_t version, ur_virtual_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGranularityGetInfo = ur::opencl::urVirtualMemGranularityGetInfo;
  pDdiTable->pfnReserve = ur::opencl::urVirtualMemReserve;
  pDdiTable->pfnFree = ur::opencl::urVirtualMemFree;
  pDdiTable->pfnMap = ur::opencl::urVirtualMemMap;
  pDdiTable->pfnUnmap = ur::opencl::urVirtualMemUnmap;
  pDdiTable->pfnSetAccess = ur::opencl::urVirtualMemSetAccess;
  pDdiTable->pfnGetInfo = ur::opencl::urVirtualMemGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    ur_api_version_t version, ur_device_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGet = ur::opencl::urDeviceGet;
  pDdiTable->pfnGetInfo = ur::opencl::urDeviceGetInfo;
  pDdiTable->pfnRetain = ur::opencl::urDeviceRetain;
  pDdiTable->pfnRelease = ur::opencl::urDeviceRelease;
  pDdiTable->pfnPartition = ur::opencl::urDevicePartition;
  pDdiTable->pfnSelectBinary = ur::opencl::urDeviceSelectBinary;
  pDdiTable->pfnGetNativeHandle = ur::opencl::urDeviceGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::opencl::urDeviceCreateWithNativeHandle;
  pDdiTable->pfnGetGlobalTimestamps = ur::opencl::urDeviceGetGlobalTimestamps;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetDeviceExpProcAddrTable(
    ur_api_version_t version, ur_device_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnWaitExp = ur::opencl::urDeviceWaitExp;

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
