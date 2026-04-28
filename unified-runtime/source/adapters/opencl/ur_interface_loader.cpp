//===--------- ur_interface_loader.cpp - Unified Runtime  ------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include <unified-runtime/ur_api.h>
#include <unified-runtime/ur_ddi.h>

namespace {

// TODO - this is a duplicate of what is in the L0 plugin
// We should move this to somewhere common
ur_result_t validateProcInputs(ur_api_version_t Version, void *pDdiTable) {
  if (nullptr == pDdiTable) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }
  // Pre 1.0 we enforce loader and adapter must have same version.
  // Post 1.0 only major version match should be required.
  if (Version != UR_API_VERSION_CURRENT) {
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }
  return UR_RESULT_SUCCESS;
}
} // namespace

#ifdef UR_STATIC_ADAPTER_OPENCL
#define NAMESPACE_ ur::opencl
namespace ur::opencl {
#else
#define NAMESPACE_
extern "C" {
#endif

UR_DLLEXPORT ur_result_t UR_APICALL urGetAdapterProcAddrTable(
    ur_api_version_t version, ur_adapter_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnGet = NAMESPACE_::urAdapterGet;
  pDdiTable->pfnRelease = NAMESPACE_::urAdapterRelease;
  pDdiTable->pfnRetain = NAMESPACE_::urAdapterRetain;
  pDdiTable->pfnGetLastError = NAMESPACE_::urAdapterGetLastError;
  pDdiTable->pfnGetInfo = NAMESPACE_::urAdapterGetInfo;
  pDdiTable->pfnSetLoggerCallback = NAMESPACE_::urAdapterSetLoggerCallback;
  pDdiTable->pfnSetLoggerCallbackLevel = NAMESPACE_::urAdapterSetLoggerCallbackLevel;

  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    ur_api_version_t Version, ur_platform_dditable_t *pDdiTable) {
  auto Result = validateProcInputs(Version, pDdiTable);
  if (UR_RESULT_SUCCESS != Result) {
    return Result;
  }
  pDdiTable->pfnCreateWithNativeHandle = NAMESPACE_::urPlatformCreateWithNativeHandle;
  pDdiTable->pfnGet = NAMESPACE_::urPlatformGet;
  pDdiTable->pfnGetApiVersion = NAMESPACE_::urPlatformGetApiVersion;
  pDdiTable->pfnGetInfo = NAMESPACE_::urPlatformGetInfo;
  pDdiTable->pfnGetNativeHandle = NAMESPACE_::urPlatformGetNativeHandle;
  pDdiTable->pfnGetBackendOption = NAMESPACE_::urPlatformGetBackendOption;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetContextProcAddrTable(
    ur_api_version_t Version, ur_context_dditable_t *pDdiTable) {
  auto Result = validateProcInputs(Version, pDdiTable);
  if (UR_RESULT_SUCCESS != Result) {
    return Result;
  }
  pDdiTable->pfnCreate = NAMESPACE_::urContextCreate;
  pDdiTable->pfnCreateWithNativeHandle = NAMESPACE_::urContextCreateWithNativeHandle;
  pDdiTable->pfnGetInfo = NAMESPACE_::urContextGetInfo;
  pDdiTable->pfnGetNativeHandle = NAMESPACE_::urContextGetNativeHandle;
  pDdiTable->pfnRelease = NAMESPACE_::urContextRelease;
  pDdiTable->pfnRetain = NAMESPACE_::urContextRetain;
  pDdiTable->pfnSetExtendedDeleter = NAMESPACE_::urContextSetExtendedDeleter;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetEventProcAddrTable(
    ur_api_version_t Version, ur_event_dditable_t *pDdiTable) {
  auto Result = validateProcInputs(Version, pDdiTable);
  if (UR_RESULT_SUCCESS != Result) {
    return Result;
  }
  pDdiTable->pfnCreateWithNativeHandle = NAMESPACE_::urEventCreateWithNativeHandle;
  pDdiTable->pfnGetInfo = NAMESPACE_::urEventGetInfo;
  pDdiTable->pfnGetNativeHandle = NAMESPACE_::urEventGetNativeHandle;
  pDdiTable->pfnGetProfilingInfo = NAMESPACE_::urEventGetProfilingInfo;
  pDdiTable->pfnRelease = NAMESPACE_::urEventRelease;
  pDdiTable->pfnRetain = NAMESPACE_::urEventRetain;
  pDdiTable->pfnSetCallback = NAMESPACE_::urEventSetCallback;
  pDdiTable->pfnWait = NAMESPACE_::urEventWait;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetProgramProcAddrTable(
    ur_api_version_t Version, ur_program_dditable_t *pDdiTable) {
  auto Result = validateProcInputs(Version, pDdiTable);
  if (UR_RESULT_SUCCESS != Result) {
    return Result;
  }
  pDdiTable->pfnBuild = NAMESPACE_::urProgramBuild;
  pDdiTable->pfnCompile = NAMESPACE_::urProgramCompile;
  pDdiTable->pfnCreateWithBinary = NAMESPACE_::urProgramCreateWithBinary;
  pDdiTable->pfnCreateWithIL = NAMESPACE_::urProgramCreateWithIL;
  pDdiTable->pfnCreateWithNativeHandle = NAMESPACE_::urProgramCreateWithNativeHandle;
  pDdiTable->pfnGetBuildInfo = NAMESPACE_::urProgramGetBuildInfo;
  pDdiTable->pfnGetFunctionPointer = NAMESPACE_::urProgramGetFunctionPointer;
  pDdiTable->pfnGetGlobalVariablePointer = NAMESPACE_::urProgramGetGlobalVariablePointer;
  pDdiTable->pfnGetInfo = NAMESPACE_::urProgramGetInfo;
  pDdiTable->pfnGetNativeHandle = NAMESPACE_::urProgramGetNativeHandle;
  pDdiTable->pfnLink = NAMESPACE_::urProgramLink;
  pDdiTable->pfnRelease = NAMESPACE_::urProgramRelease;
  pDdiTable->pfnRetain = NAMESPACE_::urProgramRetain;
  pDdiTable->pfnSetSpecializationConstants =
      urProgramSetSpecializationConstants;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetKernelProcAddrTable(
    ur_api_version_t Version, ur_kernel_dditable_t *pDdiTable) {
  auto Result = validateProcInputs(Version, pDdiTable);
  if (UR_RESULT_SUCCESS != Result) {
    return Result;
  }
  pDdiTable->pfnCreate = NAMESPACE_::urKernelCreate;
  pDdiTable->pfnCreateWithNativeHandle = NAMESPACE_::urKernelCreateWithNativeHandle;
  pDdiTable->pfnGetGroupInfo = NAMESPACE_::urKernelGetGroupInfo;
  pDdiTable->pfnGetInfo = NAMESPACE_::urKernelGetInfo;
  pDdiTable->pfnGetNativeHandle = NAMESPACE_::urKernelGetNativeHandle;
  pDdiTable->pfnGetSubGroupInfo = NAMESPACE_::urKernelGetSubGroupInfo;
  pDdiTable->pfnRelease = NAMESPACE_::urKernelRelease;
  pDdiTable->pfnRetain = NAMESPACE_::urKernelRetain;
  pDdiTable->pfnSetExecInfo = NAMESPACE_::urKernelSetExecInfo;
  pDdiTable->pfnSetSpecializationConstants = NAMESPACE_::urKernelSetSpecializationConstants;
  pDdiTable->pfnGetSuggestedLocalWorkSize = NAMESPACE_::urKernelGetSuggestedLocalWorkSize;
  pDdiTable->pfnGetSuggestedLocalWorkSizeWithArgs =
      urKernelGetSuggestedLocalWorkSizeWithArgs;
  pDdiTable->pfnSuggestMaxCooperativeGroupCount =
      urKernelSuggestMaxCooperativeGroupCount;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    ur_api_version_t Version, ur_sampler_dditable_t *pDdiTable) {
  auto Result = validateProcInputs(Version, pDdiTable);
  if (UR_RESULT_SUCCESS != Result) {
    return Result;
  }
  pDdiTable->pfnCreate = NAMESPACE_::urSamplerCreate;
  pDdiTable->pfnCreateWithNativeHandle = NAMESPACE_::urSamplerCreateWithNativeHandle;
  pDdiTable->pfnGetInfo = NAMESPACE_::urSamplerGetInfo;
  pDdiTable->pfnGetNativeHandle = NAMESPACE_::urSamplerGetNativeHandle;
  pDdiTable->pfnRelease = NAMESPACE_::urSamplerRelease;
  pDdiTable->pfnRetain = NAMESPACE_::urSamplerRetain;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL
urGetMemProcAddrTable(ur_api_version_t Version, ur_mem_dditable_t *pDdiTable) {
  auto Result = validateProcInputs(Version, pDdiTable);
  if (UR_RESULT_SUCCESS != Result) {
    return Result;
  }
  pDdiTable->pfnBufferCreate = NAMESPACE_::urMemBufferCreate;
  pDdiTable->pfnBufferPartition = NAMESPACE_::urMemBufferPartition;
  pDdiTable->pfnBufferCreateWithNativeHandle =
      urMemBufferCreateWithNativeHandle;
  pDdiTable->pfnImageCreateWithNativeHandle = NAMESPACE_::urMemImageCreateWithNativeHandle;
  pDdiTable->pfnGetInfo = NAMESPACE_::urMemGetInfo;
  pDdiTable->pfnGetNativeHandle = NAMESPACE_::urMemGetNativeHandle;
  pDdiTable->pfnImageCreate = NAMESPACE_::urMemImageCreate;
  pDdiTable->pfnImageGetInfo = NAMESPACE_::urMemImageGetInfo;
  pDdiTable->pfnImageCreateWithNativeHandle = NAMESPACE_::urMemImageCreateWithNativeHandle;
  pDdiTable->pfnRelease = NAMESPACE_::urMemRelease;
  pDdiTable->pfnRetain = NAMESPACE_::urMemRetain;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    ur_api_version_t Version, ur_enqueue_dditable_t *pDdiTable) {
  auto Result = validateProcInputs(Version, pDdiTable);
  if (UR_RESULT_SUCCESS != Result) {
    return Result;
  }
  pDdiTable->pfnDeviceGlobalVariableRead = NAMESPACE_::urEnqueueDeviceGlobalVariableRead;
  pDdiTable->pfnDeviceGlobalVariableWrite = NAMESPACE_::urEnqueueDeviceGlobalVariableWrite;
  pDdiTable->pfnEventsWait = NAMESPACE_::urEnqueueEventsWait;
  pDdiTable->pfnEventsWaitWithBarrier = NAMESPACE_::urEnqueueEventsWaitWithBarrier;
  pDdiTable->pfnEventsWaitWithBarrierExt = NAMESPACE_::urEnqueueEventsWaitWithBarrierExt;
  pDdiTable->pfnMemBufferCopy = NAMESPACE_::urEnqueueMemBufferCopy;
  pDdiTable->pfnMemBufferCopyRect = NAMESPACE_::urEnqueueMemBufferCopyRect;
  pDdiTable->pfnMemBufferFill = NAMESPACE_::urEnqueueMemBufferFill;
  pDdiTable->pfnMemBufferMap = NAMESPACE_::urEnqueueMemBufferMap;
  pDdiTable->pfnMemBufferRead = NAMESPACE_::urEnqueueMemBufferRead;
  pDdiTable->pfnMemBufferReadRect = NAMESPACE_::urEnqueueMemBufferReadRect;
  pDdiTable->pfnMemBufferWrite = NAMESPACE_::urEnqueueMemBufferWrite;
  pDdiTable->pfnMemBufferWriteRect = NAMESPACE_::urEnqueueMemBufferWriteRect;
  pDdiTable->pfnMemImageCopy = NAMESPACE_::urEnqueueMemImageCopy;
  pDdiTable->pfnMemImageRead = NAMESPACE_::urEnqueueMemImageRead;
  pDdiTable->pfnMemImageWrite = NAMESPACE_::urEnqueueMemImageWrite;
  pDdiTable->pfnMemUnmap = NAMESPACE_::urEnqueueMemUnmap;
  pDdiTable->pfnUSMFill2D = NAMESPACE_::urEnqueueUSMFill2D;
  pDdiTable->pfnUSMFill = NAMESPACE_::urEnqueueUSMFill;
  pDdiTable->pfnUSMAdvise = NAMESPACE_::urEnqueueUSMAdvise;
  pDdiTable->pfnUSMMemcpy2D = NAMESPACE_::urEnqueueUSMMemcpy2D;
  pDdiTable->pfnUSMMemcpy = NAMESPACE_::urEnqueueUSMMemcpy;
  pDdiTable->pfnUSMPrefetch = NAMESPACE_::urEnqueueUSMPrefetch;
  pDdiTable->pfnReadHostPipe = NAMESPACE_::urEnqueueReadHostPipe;
  pDdiTable->pfnWriteHostPipe = NAMESPACE_::urEnqueueWriteHostPipe;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetQueueProcAddrTable(
    ur_api_version_t Version, ur_queue_dditable_t *pDdiTable) {
  auto Result = validateProcInputs(Version, pDdiTable);
  if (UR_RESULT_SUCCESS != Result) {
    return Result;
  }
  pDdiTable->pfnCreate = NAMESPACE_::urQueueCreate;
  pDdiTable->pfnCreateWithNativeHandle = NAMESPACE_::urQueueCreateWithNativeHandle;
  pDdiTable->pfnFinish = NAMESPACE_::urQueueFinish;
  pDdiTable->pfnFlush = NAMESPACE_::urQueueFlush;
  pDdiTable->pfnGetInfo = NAMESPACE_::urQueueGetInfo;
  pDdiTable->pfnGetNativeHandle = NAMESPACE_::urQueueGetNativeHandle;
  pDdiTable->pfnRelease = NAMESPACE_::urQueueRelease;
  pDdiTable->pfnRetain = NAMESPACE_::urQueueRetain;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL
urGetUSMProcAddrTable(ur_api_version_t Version, ur_usm_dditable_t *pDdiTable) {
  auto Result = validateProcInputs(Version, pDdiTable);
  if (UR_RESULT_SUCCESS != Result) {
    return Result;
  }
  pDdiTable->pfnDeviceAlloc = NAMESPACE_::urUSMDeviceAlloc;
  pDdiTable->pfnFree = NAMESPACE_::urUSMFree;
  pDdiTable->pfnGetMemAllocInfo = NAMESPACE_::urUSMGetMemAllocInfo;
  pDdiTable->pfnHostAlloc = NAMESPACE_::urUSMHostAlloc;
  pDdiTable->pfnPoolCreate = NAMESPACE_::urUSMPoolCreate;
  pDdiTable->pfnPoolRetain = NAMESPACE_::urUSMPoolRetain;
  pDdiTable->pfnPoolRelease = NAMESPACE_::urUSMPoolRelease;
  pDdiTable->pfnPoolGetInfo = NAMESPACE_::urUSMPoolGetInfo;
  pDdiTable->pfnSharedAlloc = NAMESPACE_::urUSMSharedAlloc;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetUSMExpProcAddrTable(
    ur_api_version_t Version, ur_usm_exp_dditable_t *pDdiTable) {
  auto Result = validateProcInputs(Version, pDdiTable);
  if (UR_RESULT_SUCCESS != Result) {
    return Result;
  }

  pDdiTable->pfnImportExp = NAMESPACE_::urUSMImportExp;
  pDdiTable->pfnReleaseExp = NAMESPACE_::urUSMReleaseExp;
  pDdiTable->pfnContextMemcpyExp = NAMESPACE_::urUSMContextMemcpyExp;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    ur_api_version_t Version, ur_device_dditable_t *pDdiTable) {
  auto Result = validateProcInputs(Version, pDdiTable);
  if (UR_RESULT_SUCCESS != Result) {
    return Result;
  }
  pDdiTable->pfnCreateWithNativeHandle = NAMESPACE_::urDeviceCreateWithNativeHandle;
  pDdiTable->pfnGet = NAMESPACE_::urDeviceGet;
  pDdiTable->pfnGetGlobalTimestamps = NAMESPACE_::urDeviceGetGlobalTimestamps;
  pDdiTable->pfnGetInfo = NAMESPACE_::urDeviceGetInfo;
  pDdiTable->pfnGetNativeHandle = NAMESPACE_::urDeviceGetNativeHandle;
  pDdiTable->pfnPartition = NAMESPACE_::urDevicePartition;
  pDdiTable->pfnRelease = NAMESPACE_::urDeviceRelease;
  pDdiTable->pfnRetain = NAMESPACE_::urDeviceRetain;
  pDdiTable->pfnSelectBinary = NAMESPACE_::urDeviceSelectBinary;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetDeviceExpProcAddrTable(
    ur_api_version_t version, ur_device_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnWaitExp = NAMESPACE_::urDeviceWaitExp;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetCommandBufferExpProcAddrTable(
    ur_api_version_t version, ur_command_buffer_exp_dditable_t *pDdiTable) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnCreateExp = NAMESPACE_::urCommandBufferCreateExp;
  pDdiTable->pfnRetainExp = NAMESPACE_::urCommandBufferRetainExp;
  pDdiTable->pfnReleaseExp = NAMESPACE_::urCommandBufferReleaseExp;
  pDdiTable->pfnFinalizeExp = NAMESPACE_::urCommandBufferFinalizeExp;
  pDdiTable->pfnAppendKernelLaunchExp = NAMESPACE_::urCommandBufferAppendKernelLaunchExp;
  pDdiTable->pfnAppendKernelLaunchWithArgsExp =
      urCommandBufferAppendKernelLaunchWithArgsExp;
  pDdiTable->pfnAppendUSMMemcpyExp = NAMESPACE_::urCommandBufferAppendUSMMemcpyExp;
  pDdiTable->pfnAppendUSMFillExp = NAMESPACE_::urCommandBufferAppendUSMFillExp;
  pDdiTable->pfnAppendMemBufferCopyExp = NAMESPACE_::urCommandBufferAppendMemBufferCopyExp;
  pDdiTable->pfnAppendMemBufferCopyRectExp =
      urCommandBufferAppendMemBufferCopyRectExp;
  pDdiTable->pfnAppendMemBufferReadExp = NAMESPACE_::urCommandBufferAppendMemBufferReadExp;
  pDdiTable->pfnAppendMemBufferReadRectExp =
      urCommandBufferAppendMemBufferReadRectExp;
  pDdiTable->pfnAppendMemBufferWriteExp =
      urCommandBufferAppendMemBufferWriteExp;
  pDdiTable->pfnAppendMemBufferWriteRectExp =
      urCommandBufferAppendMemBufferWriteRectExp;
  pDdiTable->pfnAppendUSMPrefetchExp = NAMESPACE_::urCommandBufferAppendUSMPrefetchExp;
  pDdiTable->pfnAppendUSMAdviseExp = NAMESPACE_::urCommandBufferAppendUSMAdviseExp;
  pDdiTable->pfnAppendMemBufferFillExp = NAMESPACE_::urCommandBufferAppendMemBufferFillExp;
  pDdiTable->pfnUpdateKernelLaunchExp = NAMESPACE_::urCommandBufferUpdateKernelLaunchExp;
  pDdiTable->pfnGetInfoExp = NAMESPACE_::urCommandBufferGetInfoExp;
  pDdiTable->pfnUpdateWaitEventsExp = NAMESPACE_::urCommandBufferUpdateWaitEventsExp;
  pDdiTable->pfnUpdateSignalEventExp = NAMESPACE_::urCommandBufferUpdateSignalEventExp;
  pDdiTable->pfnAppendNativeCommandExp = NAMESPACE_::urCommandBufferAppendNativeCommandExp;
  pDdiTable->pfnGetNativeHandleExp = NAMESPACE_::urCommandBufferGetNativeHandleExp;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetUsmP2PExpProcAddrTable(
    ur_api_version_t version, ur_usm_p2p_exp_dditable_t *pDdiTable) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnEnablePeerAccessExp = NAMESPACE_::urUsmP2PEnablePeerAccessExp;
  pDdiTable->pfnDisablePeerAccessExp = NAMESPACE_::urUsmP2PDisablePeerAccessExp;
  pDdiTable->pfnPeerAccessGetInfoExp = NAMESPACE_::urUsmP2PPeerAccessGetInfoExp;

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
  pDdiTable->pfnImageAllocateExp = NAMESPACE_::urBindlessImagesImageAllocateExp;
  pDdiTable->pfnImageFreeExp = NAMESPACE_::urBindlessImagesImageFreeExp;
  pDdiTable->pfnUnsampledImageCreateExp =
      urBindlessImagesUnsampledImageCreateExp;
  pDdiTable->pfnSampledImageCreateExp = NAMESPACE_::urBindlessImagesSampledImageCreateExp;
  pDdiTable->pfnImageCopyExp = NAMESPACE_::urBindlessImagesImageCopyExp;
  pDdiTable->pfnImageGetInfoExp = NAMESPACE_::urBindlessImagesImageGetInfoExp;
  pDdiTable->pfnMipmapGetLevelExp = NAMESPACE_::urBindlessImagesMipmapGetLevelExp;
  pDdiTable->pfnMipmapFreeExp = NAMESPACE_::urBindlessImagesMipmapFreeExp;
  pDdiTable->pfnImportExternalMemoryExp =
      urBindlessImagesImportExternalMemoryExp;
  pDdiTable->pfnMapExternalArrayExp = NAMESPACE_::urBindlessImagesMapExternalArrayExp;
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

UR_DLLEXPORT ur_result_t UR_APICALL urGetMemoryExportExpProcAddrTable(
    ur_api_version_t version, ur_memory_export_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnAllocExportableMemoryExp =
      urMemoryExportAllocExportableMemoryExp;
  pDdiTable->pfnFreeExportableMemoryExp = NAMESPACE_::urMemoryExportFreeExportableMemoryExp;
  pDdiTable->pfnExportMemoryHandleExp = NAMESPACE_::urMemoryExportExportMemoryHandleExp;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetVirtualMemProcAddrTable(
    ur_api_version_t version, ur_virtual_mem_dditable_t *pDdiTable) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }

  pDdiTable->pfnFree = NAMESPACE_::urVirtualMemFree;
  pDdiTable->pfnGetInfo = NAMESPACE_::urVirtualMemGetInfo;
  pDdiTable->pfnGranularityGetInfo = NAMESPACE_::urVirtualMemGranularityGetInfo;
  pDdiTable->pfnMap = NAMESPACE_::urVirtualMemMap;
  pDdiTable->pfnReserve = NAMESPACE_::urVirtualMemReserve;
  pDdiTable->pfnSetAccess = NAMESPACE_::urVirtualMemSetAccess;
  pDdiTable->pfnUnmap = NAMESPACE_::urVirtualMemUnmap;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    ur_api_version_t version, ur_physical_mem_dditable_t *pDdiTable) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }

  pDdiTable->pfnCreate = NAMESPACE_::urPhysicalMemCreate;
  pDdiTable->pfnRelease = NAMESPACE_::urPhysicalMemRelease;
  pDdiTable->pfnRetain = NAMESPACE_::urPhysicalMemRetain;
  pDdiTable->pfnGetInfo = NAMESPACE_::urPhysicalMemGetInfo;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    ur_api_version_t version, ur_enqueue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnTimestampRecordingExp = NAMESPACE_::urEnqueueTimestampRecordingExp;
  pDdiTable->pfnNativeCommandExp = NAMESPACE_::urEnqueueNativeCommandExp;
  pDdiTable->pfnCommandBufferExp = NAMESPACE_::urEnqueueCommandBufferExp;
  pDdiTable->pfnKernelLaunchWithArgsExp = NAMESPACE_::urEnqueueKernelLaunchWithArgsExp;
  pDdiTable->pfnGraphExp = NAMESPACE_::urEnqueueGraphExp;

  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetIPCExpProcAddrTable(
    ur_api_version_t version, ur_ipc_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnGetMemHandleExp = NAMESPACE_::urIPCGetMemHandleExp;
  pDdiTable->pfnPutMemHandleExp = NAMESPACE_::urIPCPutMemHandleExp;
  pDdiTable->pfnOpenMemHandleExp = NAMESPACE_::urIPCOpenMemHandleExp;
  pDdiTable->pfnCloseMemHandleExp = NAMESPACE_::urIPCCloseMemHandleExp;

  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetProgramExpProcAddrTable(
    ur_api_version_t version, ur_program_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnBuildExp = NAMESPACE_::urProgramBuildExp;
  pDdiTable->pfnCompileExp = NAMESPACE_::urProgramCompileExp;
  pDdiTable->pfnLinkExp = NAMESPACE_::urProgramLinkExp;
  pDdiTable->pfnDynamicLinkExp = NAMESPACE_::urProgramDynamicLinkExp;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetQueueExpProcAddrTable(
    ur_api_version_t version, ur_queue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnBeginGraphCaptureExp = NAMESPACE_::urQueueBeginGraphCaptureExp;
  pDdiTable->pfnBeginCaptureIntoGraphExp = NAMESPACE_::urQueueBeginCaptureIntoGraphExp;
  pDdiTable->pfnEndGraphCaptureExp = NAMESPACE_::urQueueEndGraphCaptureExp;
  pDdiTable->pfnIsGraphCaptureEnabledExp = NAMESPACE_::urQueueIsGraphCaptureEnabledExp;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetGraphExpProcAddrTable(
    ur_api_version_t version, ur_graph_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreateExp = NAMESPACE_::urGraphCreateExp;
  pDdiTable->pfnInstantiateGraphExp = NAMESPACE_::urGraphInstantiateGraphExp;
  pDdiTable->pfnDestroyExp = NAMESPACE_::urGraphDestroyExp;
  pDdiTable->pfnExecutableGraphDestroyExp = NAMESPACE_::urGraphExecutableGraphDestroyExp;
  pDdiTable->pfnIsEmptyExp = NAMESPACE_::urGraphIsEmptyExp;
  pDdiTable->pfnDumpContentsExp = NAMESPACE_::urGraphDumpContentsExp;

  return UR_RESULT_SUCCESS;
}

#ifdef UR_STATIC_ADAPTER_OPENCL
} // namespace ur::opencl
#else
} // extern "C"
#endif
#undef NAMESPACE_

namespace {
ur_result_t populateDdiTable(ur_dditable_t *ddi) {
  if (ddi == nullptr) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  ur_result_t result;
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
  result = NAMESPACE_::urGetIPCExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                &ddi->IPCExp);
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
  result = NAMESPACE_::urGetKernelProcAddrTable(UR_API_VERSION_CURRENT,
                                                &ddi->Kernel);
  if (result != UR_RESULT_SUCCESS)
    return result;
  result =
      NAMESPACE_::urGetMemProcAddrTable(UR_API_VERSION_CURRENT, &ddi->Mem);
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
  result = NAMESPACE_::urGetMemoryExportExpProcAddrTable(
      UR_API_VERSION_CURRENT, &ddi->MemoryExportExp);
  if (result != UR_RESULT_SUCCESS)
    return result;

  return UR_RESULT_SUCCESS;
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
