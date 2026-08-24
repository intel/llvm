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

#include "ur_interface_loader_common_forwarders.hpp"
#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO
namespace ur::level_zero::v1 {
#else
extern "C" {
#endif

UR_APIEXPORT ur_result_t UR_APICALL urGetAdapterProcAddrTable(
    ur_api_version_t version, ur_adapter_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGet = ur::level_zero::v1::urAdapterGet;
  pDdiTable->pfnRelease = ur::level_zero::v1::urAdapterRelease;
  pDdiTable->pfnRetain = ur::level_zero::v1::urAdapterRetain;
  pDdiTable->pfnGetLastError = ur::level_zero::v1::urAdapterGetLastError;
  pDdiTable->pfnGetInfo = ur::level_zero::v1::urAdapterGetInfo;
  pDdiTable->pfnSetLoggerCallback =
      ur::level_zero::v1::urAdapterSetLoggerCallback;
  pDdiTable->pfnSetLoggerCallbackLevel =
      ur::level_zero::v1::urAdapterSetLoggerCallbackLevel;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetBindlessImagesExpProcAddrTable(
    ur_api_version_t version, ur_bindless_images_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnUnsampledImageHandleDestroyExp =
      ur::level_zero::v1::urBindlessImagesUnsampledImageHandleDestroyExp;
  pDdiTable->pfnSampledImageHandleDestroyExp =
      ur::level_zero::v1::urBindlessImagesSampledImageHandleDestroyExp;
  pDdiTable->pfnImageAllocateExp =
      ur::level_zero::v1::urBindlessImagesImageAllocateExp;
  pDdiTable->pfnImageFreeExp = ur::level_zero::v1::urBindlessImagesImageFreeExp;
  pDdiTable->pfnUnsampledImageCreateExp =
      ur::level_zero::v1::urBindlessImagesUnsampledImageCreateExp;
  pDdiTable->pfnSampledImageCreateExp =
      ur::level_zero::v1::urBindlessImagesSampledImageCreateExp;
  pDdiTable->pfnImageCopyExp = ur::level_zero::v1::urBindlessImagesImageCopyExp;
  pDdiTable->pfnImageGetInfoExp =
      ur::level_zero::v1::urBindlessImagesImageGetInfoExp;
  pDdiTable->pfnGetImageMemoryHandleTypeSupportExp =
      ur::level_zero::v1::urBindlessImagesGetImageMemoryHandleTypeSupportExp;
  pDdiTable->pfnGetImageUnsampledHandleSupportExp =
      ur::level_zero::v1::urBindlessImagesGetImageUnsampledHandleSupportExp;
  pDdiTable->pfnGetImageSampledHandleSupportExp =
      ur::level_zero::v1::urBindlessImagesGetImageSampledHandleSupportExp;
  pDdiTable->pfnMipmapGetLevelExp =
      ur::level_zero::v1::urBindlessImagesMipmapGetLevelExp;
  pDdiTable->pfnMipmapFreeExp =
      ur::level_zero::v1::urBindlessImagesMipmapFreeExp;
  pDdiTable->pfnImportExternalMemoryExp =
      ur::level_zero::v1::urBindlessImagesImportExternalMemoryExp;
  pDdiTable->pfnMapExternalArrayExp =
      ur::level_zero::v1::urBindlessImagesMapExternalArrayExp;
  pDdiTable->pfnMapExternalLinearMemoryExp =
      ur::level_zero::v1::urBindlessImagesMapExternalLinearMemoryExp;
  pDdiTable->pfnReleaseExternalMemoryExp =
      ur::level_zero::v1::urBindlessImagesReleaseExternalMemoryExp;
  pDdiTable->pfnFreeMappedLinearMemoryExp =
      ur::level_zero::v1::urBindlessImagesFreeMappedLinearMemoryExp;
  pDdiTable->pfnSupportsImportingHandleTypeExp =
      ur::level_zero::v1::urBindlessImagesSupportsImportingHandleTypeExp;
  pDdiTable->pfnImportExternalSemaphoreExp =
      ur::level_zero::v1::urBindlessImagesImportExternalSemaphoreExp;
  pDdiTable->pfnReleaseExternalSemaphoreExp =
      ur::level_zero::v1::urBindlessImagesReleaseExternalSemaphoreExp;
  pDdiTable->pfnWaitExternalSemaphoreExp =
      ur::level_zero::v1::urBindlessImagesWaitExternalSemaphoreExp;
  pDdiTable->pfnSignalExternalSemaphoreExp =
      ur::level_zero::v1::urBindlessImagesSignalExternalSemaphoreExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetCommandBufferExpProcAddrTable(
    ur_api_version_t version, ur_command_buffer_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateExp = ur::level_zero::v1::urCommandBufferCreateExp;
  pDdiTable->pfnRetainExp = ur::level_zero::v1::urCommandBufferRetainExp;
  pDdiTable->pfnReleaseExp = ur::level_zero::v1::urCommandBufferReleaseExp;
  pDdiTable->pfnFinalizeExp = ur::level_zero::v1::urCommandBufferFinalizeExp;
  pDdiTable->pfnAppendKernelLaunchExp =
      ur::level_zero::v1::urCommandBufferAppendKernelLaunchExp;
  pDdiTable->pfnAppendKernelLaunchWithArgsExp =
      ur::level_zero::v1::urCommandBufferAppendKernelLaunchWithArgsExp;
  pDdiTable->pfnAppendUSMMemcpyExp =
      ur::level_zero::v1::urCommandBufferAppendUSMMemcpyExp;
  pDdiTable->pfnAppendUSMFillExp =
      ur::level_zero::v1::urCommandBufferAppendUSMFillExp;
  pDdiTable->pfnAppendMemBufferCopyExp =
      ur::level_zero::v1::urCommandBufferAppendMemBufferCopyExp;
  pDdiTable->pfnAppendMemBufferWriteExp =
      ur::level_zero::v1::urCommandBufferAppendMemBufferWriteExp;
  pDdiTable->pfnAppendMemBufferReadExp =
      ur::level_zero::v1::urCommandBufferAppendMemBufferReadExp;
  pDdiTable->pfnAppendMemBufferCopyRectExp =
      ur::level_zero::v1::urCommandBufferAppendMemBufferCopyRectExp;
  pDdiTable->pfnAppendMemBufferWriteRectExp =
      ur::level_zero::v1::urCommandBufferAppendMemBufferWriteRectExp;
  pDdiTable->pfnAppendMemBufferReadRectExp =
      ur::level_zero::v1::urCommandBufferAppendMemBufferReadRectExp;
  pDdiTable->pfnAppendMemBufferFillExp =
      ur::level_zero::v1::urCommandBufferAppendMemBufferFillExp;
  pDdiTable->pfnAppendUSMPrefetchExp =
      ur::level_zero::v1::urCommandBufferAppendUSMPrefetchExp;
  pDdiTable->pfnAppendUSMAdviseExp =
      ur::level_zero::v1::urCommandBufferAppendUSMAdviseExp;
  pDdiTable->pfnAppendNativeCommandExp =
      ur::level_zero::v1::urCommandBufferAppendNativeCommandExp;
  pDdiTable->pfnUpdateKernelLaunchExp =
      ur::level_zero::v1::urCommandBufferUpdateKernelLaunchExp;
  pDdiTable->pfnUpdateSignalEventExp =
      ur::level_zero::v1::urCommandBufferUpdateSignalEventExp;
  pDdiTable->pfnUpdateWaitEventsExp =
      ur::level_zero::v1::urCommandBufferUpdateWaitEventsExp;
  pDdiTable->pfnGetInfoExp = ur::level_zero::v1::urCommandBufferGetInfoExp;
  pDdiTable->pfnGetNativeHandleExp =
      ur::level_zero::v1::urCommandBufferGetNativeHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetContextProcAddrTable(
    ur_api_version_t version, ur_context_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::level_zero::v1::urContextCreate;
  pDdiTable->pfnRetain = ur::level_zero::v1::urContextRetain;
  pDdiTable->pfnRelease = ur::level_zero::v1::urContextRelease;
  pDdiTable->pfnGetInfo = ur::level_zero::v1::urContextGetInfo;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::v1::urContextGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::v1::urContextCreateWithNativeHandle;
  pDdiTable->pfnSetExtendedDeleter =
      ur::level_zero::v1::urContextSetExtendedDeleter;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    ur_api_version_t version, ur_enqueue_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnEventsWait = ur::level_zero::v1::urEnqueueEventsWait;
  pDdiTable->pfnEventsWaitWithBarrier =
      ur::level_zero::v1::urEnqueueEventsWaitWithBarrier;
  pDdiTable->pfnMemBufferRead = ur::level_zero::v1::urEnqueueMemBufferRead;
  pDdiTable->pfnMemBufferWrite = ur::level_zero::v1::urEnqueueMemBufferWrite;
  pDdiTable->pfnMemBufferReadRect =
      ur::level_zero::v1::urEnqueueMemBufferReadRect;
  pDdiTable->pfnMemBufferWriteRect =
      ur::level_zero::v1::urEnqueueMemBufferWriteRect;
  pDdiTable->pfnMemBufferCopy = ur::level_zero::v1::urEnqueueMemBufferCopy;
  pDdiTable->pfnMemBufferCopyRect =
      ur::level_zero::v1::urEnqueueMemBufferCopyRect;
  pDdiTable->pfnMemBufferFill = ur::level_zero::v1::urEnqueueMemBufferFill;
  pDdiTable->pfnMemImageRead = ur::level_zero::v1::urEnqueueMemImageRead;
  pDdiTable->pfnMemImageWrite = ur::level_zero::v1::urEnqueueMemImageWrite;
  pDdiTable->pfnMemImageCopy = ur::level_zero::v1::urEnqueueMemImageCopy;
  pDdiTable->pfnMemBufferMap = ur::level_zero::v1::urEnqueueMemBufferMap;
  pDdiTable->pfnMemUnmap = ur::level_zero::v1::urEnqueueMemUnmap;
  pDdiTable->pfnUSMFill = ur::level_zero::v1::urEnqueueUSMFill;
  pDdiTable->pfnUSMMemcpy = ur::level_zero::v1::urEnqueueUSMMemcpy;
  pDdiTable->pfnUSMPrefetch = ur::level_zero::v1::urEnqueueUSMPrefetch;
  pDdiTable->pfnUSMAdvise = ur::level_zero::v1::urEnqueueUSMAdvise;
  pDdiTable->pfnUSMFill2D = ur::level_zero::v1::urEnqueueUSMFill2D;
  pDdiTable->pfnUSMMemcpy2D = ur::level_zero::v1::urEnqueueUSMMemcpy2D;
  pDdiTable->pfnDeviceGlobalVariableWrite =
      ur::level_zero::v1::urEnqueueDeviceGlobalVariableWrite;
  pDdiTable->pfnDeviceGlobalVariableRead =
      ur::level_zero::v1::urEnqueueDeviceGlobalVariableRead;
  pDdiTable->pfnReadHostPipe = ur::level_zero::v1::urEnqueueReadHostPipe;
  pDdiTable->pfnWriteHostPipe = ur::level_zero::v1::urEnqueueWriteHostPipe;
  pDdiTable->pfnEventsWaitWithBarrierExt =
      ur::level_zero::v1::urEnqueueEventsWaitWithBarrierExt;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    ur_api_version_t version, ur_enqueue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnKernelLaunchWithArgsExp =
      ur::level_zero::v1::urEnqueueKernelLaunchWithArgsExp;
  pDdiTable->pfnUSMDeviceAllocExp =
      ur::level_zero::v1::urEnqueueUSMDeviceAllocExp;
  pDdiTable->pfnUSMSharedAllocExp =
      ur::level_zero::v1::urEnqueueUSMSharedAllocExp;
  pDdiTable->pfnUSMHostAllocExp = ur::level_zero::v1::urEnqueueUSMHostAllocExp;
  pDdiTable->pfnUSMFreeExp = ur::level_zero::v1::urEnqueueUSMFreeExp;
  pDdiTable->pfnTimestampRecordingExp =
      ur::level_zero::v1::urEnqueueTimestampRecordingExp;
  pDdiTable->pfnCommandBufferExp =
      ur::level_zero::v1::urEnqueueCommandBufferExp;
  pDdiTable->pfnHostTaskExp = ur::level_zero::v1::urEnqueueHostTaskExp;
  pDdiTable->pfnNativeCommandExp =
      ur::level_zero::v1::urEnqueueNativeCommandExp;
  pDdiTable->pfnGraphExp = ur::level_zero::v1::urEnqueueGraphExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEventProcAddrTable(
    ur_api_version_t version, ur_event_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetInfo = ur::level_zero::v1::urEventGetInfo;
  pDdiTable->pfnGetProfilingInfo = ur::level_zero::v1::urEventGetProfilingInfo;
  pDdiTable->pfnWait = ur::level_zero::v1::urEventWait;
  pDdiTable->pfnRetain = ur::level_zero::v1::urEventRetain;
  pDdiTable->pfnRelease = ur::level_zero::v1::urEventRelease;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::v1::urEventGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::v1::urEventCreateWithNativeHandle;
  pDdiTable->pfnSetCallback = ur::level_zero::v1::urEventSetCallback;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetEventExpProcAddrTable(
    ur_api_version_t version, ur_event_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateExp = ur::level_zero::v1::urEventCreateExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetGraphExpProcAddrTable(
    ur_api_version_t version, ur_graph_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateExp = ur::level_zero::v1::urGraphCreateExp;
  pDdiTable->pfnInstantiateGraphExp =
      ur::level_zero::v1::urGraphInstantiateGraphExp;
  pDdiTable->pfnDestroyExp = ur::level_zero::v1::urGraphDestroyExp;
  pDdiTable->pfnExecutableGraphDestroyExp =
      ur::level_zero::v1::urGraphExecutableGraphDestroyExp;
  pDdiTable->pfnIsEmptyExp = ur::level_zero::v1::urGraphIsEmptyExp;
  pDdiTable->pfnGetIdExp = ur::level_zero::v1::urGraphGetIdExp;
  pDdiTable->pfnSetDestructionCallbackExp =
      ur::level_zero::v1::urGraphSetDestructionCallbackExp;
  pDdiTable->pfnDumpContentsExp = ur::level_zero::v1::urGraphDumpContentsExp;
  pDdiTable->pfnGetNativeHandleExp =
      ur::level_zero::v1::urGraphGetNativeHandleExp;
  pDdiTable->pfnExecutableGraphGetNativeHandleExp =
      ur::level_zero::v1::urGraphExecutableGraphGetNativeHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetIPCExpProcAddrTable(
    ur_api_version_t version, ur_ipc_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetMemHandleExp = ur::level_zero::v1::urIPCGetMemHandleExp;
  pDdiTable->pfnPutMemHandleExp = ur::level_zero::v1::urIPCPutMemHandleExp;
  pDdiTable->pfnOpenMemHandleExp = ur::level_zero::v1::urIPCOpenMemHandleExp;
  pDdiTable->pfnCloseMemHandleExp = ur::level_zero::v1::urIPCCloseMemHandleExp;
  pDdiTable->pfnGetPhysMemHandleExp =
      ur::level_zero::v1::urIPCGetPhysMemHandleExp;
  pDdiTable->pfnPutPhysMemHandleExp =
      ur::level_zero::v1::urIPCPutPhysMemHandleExp;
  pDdiTable->pfnOpenPhysMemHandleExp =
      ur::level_zero::v1::urIPCOpenPhysMemHandleExp;
  pDdiTable->pfnClosePhysMemHandleExp =
      ur::level_zero::v1::urIPCClosePhysMemHandleExp;
  pDdiTable->pfnGetEventHandleExp = ur::level_zero::v1::urIPCGetEventHandleExp;
  pDdiTable->pfnPutEventHandleExp = ur::level_zero::v1::urIPCPutEventHandleExp;
  pDdiTable->pfnOpenEventHandleExp =
      ur::level_zero::v1::urIPCOpenEventHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetKernelProcAddrTable(
    ur_api_version_t version, ur_kernel_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::level_zero::v1::urKernelCreate;
  pDdiTable->pfnGetInfo = ur::level_zero::v1::urKernelGetInfo;
  pDdiTable->pfnGetGroupInfo = ur::level_zero::v1::urKernelGetGroupInfo;
  pDdiTable->pfnGetSubGroupInfo = ur::level_zero::v1::urKernelGetSubGroupInfo;
  pDdiTable->pfnRetain = ur::level_zero::v1::urKernelRetain;
  pDdiTable->pfnRelease = ur::level_zero::v1::urKernelRelease;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::v1::urKernelGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::v1::urKernelCreateWithNativeHandle;
  pDdiTable->pfnGetSuggestedLocalWorkSize =
      ur::level_zero::v1::urKernelGetSuggestedLocalWorkSize;
  pDdiTable->pfnGetSuggestedLocalWorkSizeWithArgs =
      ur::level_zero::v1::urKernelGetSuggestedLocalWorkSizeWithArgs;
  pDdiTable->pfnSetExecInfo = ur::level_zero::v1::urKernelSetExecInfo;
  pDdiTable->pfnSetSpecializationConstants =
      ur::level_zero::v1::urKernelSetSpecializationConstants;
  pDdiTable->pfnSuggestMaxCooperativeGroupCount =
      ur::level_zero::v1::urKernelSuggestMaxCooperativeGroupCount;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGetMemProcAddrTable(ur_api_version_t version, ur_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnImageCreate = ur::level_zero::v1::urMemImageCreate;
  pDdiTable->pfnBufferCreate = ur::level_zero::v1::urMemBufferCreate;
  pDdiTable->pfnRetain = ur::level_zero::v1::urMemRetain;
  pDdiTable->pfnRelease = ur::level_zero::v1::urMemRelease;
  pDdiTable->pfnBufferPartition = ur::level_zero::v1::urMemBufferPartition;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::v1::urMemGetNativeHandle;
  pDdiTable->pfnBufferCreateWithNativeHandle =
      ur::level_zero::v1::urMemBufferCreateWithNativeHandle;
  pDdiTable->pfnImageCreateWithNativeHandle =
      ur::level_zero::v1::urMemImageCreateWithNativeHandle;
  pDdiTable->pfnGetInfo = ur::level_zero::v1::urMemGetInfo;
  pDdiTable->pfnImageGetInfo = ur::level_zero::v1::urMemImageGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetMemoryExportExpProcAddrTable(
    ur_api_version_t version, ur_memory_export_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnAllocExportableMemoryExp =
      ur::level_zero::v1::urMemoryExportAllocExportableMemoryExp;
  pDdiTable->pfnFreeExportableMemoryExp =
      ur::level_zero::v1::urMemoryExportFreeExportableMemoryExp;
  pDdiTable->pfnExportMemoryHandleExp =
      ur::level_zero::v1::urMemoryExportExportMemoryHandleExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    ur_api_version_t version, ur_physical_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::level_zero::v1::urPhysicalMemCreate;
  pDdiTable->pfnRetain = ur::level_zero::v1::urPhysicalMemRetain;
  pDdiTable->pfnRelease = ur::level_zero::v1::urPhysicalMemRelease;
  pDdiTable->pfnGetInfo = ur::level_zero::v1::urPhysicalMemGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    ur_api_version_t version, ur_platform_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGet = ur::level_zero::v1::urPlatformGet;
  pDdiTable->pfnGetInfo = ur::level_zero::v1::urPlatformGetInfo;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::v1::urPlatformGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::v1::urPlatformCreateWithNativeHandle;
  pDdiTable->pfnGetApiVersion = ur::level_zero::v1::urPlatformGetApiVersion;
  pDdiTable->pfnGetBackendOption =
      ur::level_zero::v1::urPlatformGetBackendOption;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetProgramProcAddrTable(
    ur_api_version_t version, ur_program_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreateWithIL = ur::level_zero::v1::urProgramCreateWithIL;
  pDdiTable->pfnCreateWithBinary =
      ur::level_zero::v1::urProgramCreateWithBinary;
  pDdiTable->pfnBuild = ur::level_zero::v1::urProgramBuild;
  pDdiTable->pfnCompile = ur::level_zero::v1::urProgramCompile;
  pDdiTable->pfnLink = ur::level_zero::v1::urProgramLink;
  pDdiTable->pfnRetain = ur::level_zero::v1::urProgramRetain;
  pDdiTable->pfnRelease = ur::level_zero::v1::urProgramRelease;
  pDdiTable->pfnGetFunctionPointer =
      ur::level_zero::v1::urProgramGetFunctionPointer;
  pDdiTable->pfnGetGlobalVariablePointer =
      ur::level_zero::v1::urProgramGetGlobalVariablePointer;
  pDdiTable->pfnGetInfo = ur::level_zero::v1::urProgramGetInfo;
  pDdiTable->pfnGetBuildInfo = ur::level_zero::v1::urProgramGetBuildInfo;
  pDdiTable->pfnSetSpecializationConstants =
      ur::level_zero::v1::urProgramSetSpecializationConstants;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::v1::urProgramGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::v1::urProgramCreateWithNativeHandle;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetProgramExpProcAddrTable(
    ur_api_version_t version, ur_program_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnDynamicLinkExp = ur::level_zero::v1::urProgramDynamicLinkExp;
  pDdiTable->pfnBuildExp = ur::level_zero::v1::urProgramBuildExp;
  pDdiTable->pfnCompileExp = ur::level_zero::v1::urProgramCompileExp;
  pDdiTable->pfnLinkExp = ur::level_zero::v1::urProgramLinkExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetQueueProcAddrTable(
    ur_api_version_t version, ur_queue_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetInfo = ur::level_zero::v1::urQueueGetInfo;
  pDdiTable->pfnCreate = ur::level_zero::v1::urQueueCreate;
  pDdiTable->pfnRetain = ur::level_zero::v1::urQueueRetain;
  pDdiTable->pfnRelease = ur::level_zero::v1::urQueueRelease;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::v1::urQueueGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::v1::urQueueCreateWithNativeHandle;
  pDdiTable->pfnFinish = ur::level_zero::v1::urQueueFinish;
  pDdiTable->pfnFlush = ur::level_zero::v1::urQueueFlush;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetQueueExpProcAddrTable(
    ur_api_version_t version, ur_queue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnBeginGraphCaptureExp =
      ur::level_zero::v1::urQueueBeginGraphCaptureExp;
  pDdiTable->pfnBeginCaptureIntoGraphExp =
      ur::level_zero::v1::urQueueBeginCaptureIntoGraphExp;
  pDdiTable->pfnEndGraphCaptureExp =
      ur::level_zero::v1::urQueueEndGraphCaptureExp;
  pDdiTable->pfnIsGraphCaptureEnabledExp =
      ur::level_zero::v1::urQueueIsGraphCaptureEnabledExp;
  pDdiTable->pfnGetGraphExp = ur::level_zero::v1::urQueueGetGraphExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    ur_api_version_t version, ur_sampler_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCreate = ur::level_zero::v1::urSamplerCreate;
  pDdiTable->pfnRetain = ur::level_zero::v1::urSamplerRetain;
  pDdiTable->pfnRelease = ur::level_zero::v1::urSamplerRelease;
  pDdiTable->pfnGetInfo = ur::level_zero::v1::urSamplerGetInfo;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::v1::urSamplerGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::v1::urSamplerCreateWithNativeHandle;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urGetUSMProcAddrTable(ur_api_version_t version, ur_usm_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnHostAlloc = ur::level_zero::v1::urUSMHostAlloc;
  pDdiTable->pfnDeviceAlloc = ur::level_zero::v1::urUSMDeviceAlloc;
  pDdiTable->pfnSharedAlloc = ur::level_zero::v1::urUSMSharedAlloc;
  pDdiTable->pfnFree = ur::level_zero::v1::urUSMFree;
  pDdiTable->pfnGetMemAllocInfo = ur::level_zero::v1::urUSMGetMemAllocInfo;
  pDdiTable->pfnPoolCreate = ur::level_zero::v1::urUSMPoolCreate;
  pDdiTable->pfnPoolRetain = ur::level_zero::v1::urUSMPoolRetain;
  pDdiTable->pfnPoolRelease = ur::level_zero::v1::urUSMPoolRelease;
  pDdiTable->pfnPoolGetInfo = ur::level_zero::v1::urUSMPoolGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetUSMExpProcAddrTable(
    ur_api_version_t version, ur_usm_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnPoolCreateExp = ur::level_zero::v1::urUSMPoolCreateExp;
  pDdiTable->pfnPoolDestroyExp = ur::level_zero::v1::urUSMPoolDestroyExp;
  pDdiTable->pfnPoolGetDefaultDevicePoolExp =
      ur::level_zero::v1::urUSMPoolGetDefaultDevicePoolExp;
  pDdiTable->pfnPoolGetInfoExp = ur::level_zero::v1::urUSMPoolGetInfoExp;
  pDdiTable->pfnPoolSetInfoExp = ur::level_zero::v1::urUSMPoolSetInfoExp;
  pDdiTable->pfnPoolSetDevicePoolExp =
      ur::level_zero::v1::urUSMPoolSetDevicePoolExp;
  pDdiTable->pfnPoolGetDevicePoolExp =
      ur::level_zero::v1::urUSMPoolGetDevicePoolExp;
  pDdiTable->pfnPoolTrimToExp = ur::level_zero::v1::urUSMPoolTrimToExp;
  pDdiTable->pfnPitchedAllocExp = ur::level_zero::v1::urUSMPitchedAllocExp;
  pDdiTable->pfnContextMemcpyExp = ur::level_zero::v1::urUSMContextMemcpyExp;
  pDdiTable->pfnHostAllocUnregisterExp =
      ur::level_zero::v1::urUSMHostAllocUnregisterExp;
  pDdiTable->pfnHostAllocRegisterExp =
      ur::level_zero::v1::urUSMHostAllocRegisterExp;
  pDdiTable->pfnImportExp = ur::level_zero::v1::urUSMImportExp;
  pDdiTable->pfnReleaseExp = ur::level_zero::v1::urUSMReleaseExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetUsmP2PExpProcAddrTable(
    ur_api_version_t version, ur_usm_p2p_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnEnablePeerAccessExp =
      ur::level_zero::v1::urUsmP2PEnablePeerAccessExp;
  pDdiTable->pfnDisablePeerAccessExp =
      ur::level_zero::v1::urUsmP2PDisablePeerAccessExp;
  pDdiTable->pfnPeerAccessGetInfoExp =
      ur::level_zero::v1::urUsmP2PPeerAccessGetInfoExp;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetVirtualMemProcAddrTable(
    ur_api_version_t version, ur_virtual_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGranularityGetInfo =
      ur::level_zero::v1::urVirtualMemGranularityGetInfo;
  pDdiTable->pfnReserve = ur::level_zero::v1::urVirtualMemReserve;
  pDdiTable->pfnFree = ur::level_zero::v1::urVirtualMemFree;
  pDdiTable->pfnMap = ur::level_zero::v1::urVirtualMemMap;
  pDdiTable->pfnUnmap = ur::level_zero::v1::urVirtualMemUnmap;
  pDdiTable->pfnSetAccess = ur::level_zero::v1::urVirtualMemSetAccess;
  pDdiTable->pfnGetInfo = ur::level_zero::v1::urVirtualMemGetInfo;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    ur_api_version_t version, ur_device_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGet = ur::level_zero::v1::urDeviceGet;
  pDdiTable->pfnGetInfo = ur::level_zero::v1::urDeviceGetInfo;
  pDdiTable->pfnRetain = ur::level_zero::v1::urDeviceRetain;
  pDdiTable->pfnRelease = ur::level_zero::v1::urDeviceRelease;
  pDdiTable->pfnPartition = ur::level_zero::v1::urDevicePartition;
  pDdiTable->pfnSelectBinary = ur::level_zero::v1::urDeviceSelectBinary;
  pDdiTable->pfnGetNativeHandle = ur::level_zero::v1::urDeviceGetNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur::level_zero::v1::urDeviceCreateWithNativeHandle;
  pDdiTable->pfnGetGlobalTimestamps =
      ur::level_zero::v1::urDeviceGetGlobalTimestamps;

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetDeviceExpProcAddrTable(
    ur_api_version_t version, ur_device_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnWaitExp = ur::level_zero::v1::urDeviceWaitExp;

  return result;
}

#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO
} // namespace ur::level_zero::v1
#else
} // extern "C"
#endif

namespace {
ur_result_t populateDdiTable(ur_dditable_t *ddi) {
  if (ddi == nullptr) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  ur_result_t result;

#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO
#define NAMESPACE_ ::ur::level_zero::v1
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

namespace ur::level_zero::v1 {
const ur_dditable_t *ddi_getter::value() {
  static std::once_flag flag;
  static ur_dditable_t table;

  std::call_once(flag, []() { populateDdiTable(&table); });
  return &table;
}

#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO
ur_result_t urAdapterGetDdiTables(ur_dditable_t *ddi) {
  return populateDdiTable(ddi);
}
#endif
} // namespace ur::level_zero::v1
