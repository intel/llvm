//===----------- ur_interface_loader.cpp - LLVM Offload Plugin  -----------===//
//
// Copyright (C) 2024 Intel Corporation
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
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnGet = urPlatformGet;
  pDdiTable->pfnGetApiVersion = nullptr;
  pDdiTable->pfnGetInfo = urPlatformGetInfo;
  pDdiTable->pfnGetNativeHandle = nullptr;
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
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnRelease = urContextRelease;
  pDdiTable->pfnRetain = urContextRetain;
  pDdiTable->pfnSetExtendedDeleter = nullptr;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetEventProcAddrTable(
    ur_api_version_t version, ur_event_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnGetProfilingInfo = nullptr;
  pDdiTable->pfnRelease = urEventRelease;
  pDdiTable->pfnRetain = urEventRetain;
  pDdiTable->pfnSetCallback = nullptr;
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
  pDdiTable->pfnCompile = nullptr;
  pDdiTable->pfnCreateWithBinary = urProgramCreateWithBinary;
  pDdiTable->pfnCreateWithIL = nullptr;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnGetBuildInfo = nullptr;
  pDdiTable->pfnGetFunctionPointer = nullptr;
  pDdiTable->pfnGetGlobalVariablePointer = nullptr;
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnLink = nullptr;
  pDdiTable->pfnRelease = urProgramRelease;
  pDdiTable->pfnRetain = urProgramRetain;
  pDdiTable->pfnSetSpecializationConstants = nullptr;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetKernelProcAddrTable(
    ur_api_version_t version, ur_kernel_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreate = urKernelCreate;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnGetGroupInfo = urKernelGetGroupInfo;
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnGetSubGroupInfo = nullptr;
  pDdiTable->pfnRelease = urKernelRelease;
  pDdiTable->pfnRetain = urKernelRetain;
  pDdiTable->pfnSetArgLocal = nullptr;
  pDdiTable->pfnSetArgMemObj = nullptr;
  pDdiTable->pfnSetArgPointer = urKernelSetArgPointer;
  pDdiTable->pfnSetArgSampler = nullptr;
  pDdiTable->pfnSetArgValue = urKernelSetArgValue;
  pDdiTable->pfnSetExecInfo = urKernelSetExecInfo;
  pDdiTable->pfnSetSpecializationConstants = nullptr;
  pDdiTable->pfnGetSuggestedLocalWorkSize = nullptr;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    ur_api_version_t version, ur_sampler_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreate = nullptr;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnRelease = nullptr;
  pDdiTable->pfnRetain = nullptr;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL
urGetMemProcAddrTable(ur_api_version_t version, ur_mem_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnBufferCreate = nullptr;
  pDdiTable->pfnBufferPartition = nullptr;
  pDdiTable->pfnBufferCreateWithNativeHandle = nullptr;
  pDdiTable->pfnImageCreateWithNativeHandle = nullptr;
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
  pDdiTable->pfnImageCreate = nullptr;
  pDdiTable->pfnImageGetInfo = nullptr;
  pDdiTable->pfnRelease = nullptr;
  pDdiTable->pfnRetain = nullptr;
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
  pDdiTable->pfnKernelLaunch = urEnqueueKernelLaunch;
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
  pDdiTable->pfnReadHostPipe = nullptr;
  pDdiTable->pfnWriteHostPipe = nullptr;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetGlobalProcAddrTable(
    ur_api_version_t version, ur_global_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnAdapterGet = urAdapterGet;
  pDdiTable->pfnAdapterGetInfo = urAdapterGetInfo;
  pDdiTable->pfnAdapterRelease = urAdapterRelease;
  pDdiTable->pfnAdapterRetain = urAdapterRetain;
  pDdiTable->pfnAdapterGetLastError = nullptr;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetQueueProcAddrTable(
    ur_api_version_t version, ur_queue_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreate = urQueueCreate;
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnFinish = urQueueFinish;
  pDdiTable->pfnFlush = nullptr;
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnGetNativeHandle = nullptr;
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
  pDdiTable->pfnGetMemAllocInfo = nullptr;
  pDdiTable->pfnHostAlloc = urUSMHostAlloc;
  pDdiTable->pfnPoolCreate = nullptr;
  pDdiTable->pfnPoolRetain = nullptr;
  pDdiTable->pfnPoolRelease = nullptr;
  pDdiTable->pfnPoolGetInfo = nullptr;
  pDdiTable->pfnSharedAlloc = urUSMSharedAlloc;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    ur_api_version_t version, ur_device_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnCreateWithNativeHandle = nullptr;
  pDdiTable->pfnGet = urDeviceGet;
  pDdiTable->pfnGetGlobalTimestamps = nullptr;
  pDdiTable->pfnGetInfo = urDeviceGetInfo;
  pDdiTable->pfnGetNativeHandle = nullptr;
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
  pDdiTable->pfnCreateExp = nullptr;
  pDdiTable->pfnRetainExp = nullptr;
  pDdiTable->pfnReleaseExp = nullptr;
  pDdiTable->pfnFinalizeExp = nullptr;
  pDdiTable->pfnAppendKernelLaunchExp = nullptr;
  pDdiTable->pfnAppendUSMMemcpyExp = nullptr;
  pDdiTable->pfnAppendMemBufferCopyExp = nullptr;
  pDdiTable->pfnAppendMemBufferCopyRectExp = nullptr;
  pDdiTable->pfnAppendMemBufferReadExp = nullptr;
  pDdiTable->pfnAppendMemBufferReadRectExp = nullptr;
  pDdiTable->pfnAppendMemBufferWriteExp = nullptr;
  pDdiTable->pfnAppendMemBufferWriteRectExp = nullptr;
  pDdiTable->pfnUpdateKernelLaunchExp = nullptr;
  pDdiTable->pfnGetInfoExp = nullptr;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetUsmP2PExpProcAddrTable(
    ur_api_version_t version, ur_usm_p2p_exp_dditable_t *pDdiTable) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }
  pDdiTable->pfnEnablePeerAccessExp = nullptr;
  pDdiTable->pfnDisablePeerAccessExp = nullptr;
  pDdiTable->pfnPeerAccessGetInfoExp = nullptr;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetBindlessImagesExpProcAddrTable(
    ur_api_version_t version, ur_bindless_images_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnUnsampledImageHandleDestroyExp = nullptr;
  pDdiTable->pfnSampledImageHandleDestroyExp = nullptr;
  pDdiTable->pfnImageAllocateExp = nullptr;
  pDdiTable->pfnImageFreeExp = nullptr;
  pDdiTable->pfnUnsampledImageCreateExp = nullptr;
  pDdiTable->pfnSampledImageCreateExp = nullptr;
  pDdiTable->pfnImageCopyExp = nullptr;
  pDdiTable->pfnImageGetInfoExp = nullptr;
  pDdiTable->pfnMipmapGetLevelExp = nullptr;
  pDdiTable->pfnMipmapFreeExp = nullptr;
  pDdiTable->pfnImportExternalMemoryExp = nullptr;
  pDdiTable->pfnMapExternalArrayExp = nullptr;
  pDdiTable->pfnMapExternalLinearMemoryExp = nullptr;
  pDdiTable->pfnReleaseExternalMemoryExp = nullptr;
  pDdiTable->pfnImportExternalSemaphoreExp = nullptr;
  pDdiTable->pfnReleaseExternalSemaphoreExp = nullptr;
  pDdiTable->pfnWaitExternalSemaphoreExp = nullptr;
  pDdiTable->pfnSignalExternalSemaphoreExp = nullptr;
  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    ur_api_version_t version, ur_physical_mem_dditable_t *pDdiTable) {
  auto retVal = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != retVal) {
    return retVal;
  }

  pDdiTable->pfnCreate = nullptr;
  pDdiTable->pfnRelease = nullptr;
  pDdiTable->pfnRetain = nullptr;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetUSMExpProcAddrTable(
    ur_api_version_t version, ur_usm_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }
  pDdiTable->pfnPitchedAllocExp = nullptr;
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

  pDdiTable->pfnFree = nullptr;
  pDdiTable->pfnGetInfo = nullptr;
  pDdiTable->pfnGranularityGetInfo = nullptr;
  pDdiTable->pfnMap = nullptr;
  pDdiTable->pfnReserve = nullptr;
  pDdiTable->pfnSetAccess = nullptr;
  pDdiTable->pfnUnmap = nullptr;

  return retVal;
}

UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    ur_api_version_t version, ur_enqueue_exp_dditable_t *pDdiTable) {
  auto result = validateProcInputs(version, pDdiTable);
  if (UR_RESULT_SUCCESS != result) {
    return result;
  }

  pDdiTable->pfnCooperativeKernelLaunchExp = nullptr;
  pDdiTable->pfnTimestampRecordingExp = nullptr;
  pDdiTable->pfnNativeCommandExp = nullptr;

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
  pDdiTable->pfnCompileExp = nullptr;
  pDdiTable->pfnLinkExp = nullptr;

  return UR_RESULT_SUCCESS;
}

UR_DLLEXPORT ur_result_t UR_APICALL urAllAddrTable(ur_api_version_t version,
                                                   ur_dditable_t *pDdiTable) {
  urGetGlobalProcAddrTable(version, &pDdiTable->Global);
  urGetBindlessImagesExpProcAddrTable(version, &pDdiTable->BindlessImagesExp);
  urGetCommandBufferExpProcAddrTable(version, &pDdiTable->CommandBufferExp);
  urGetContextProcAddrTable(version, &pDdiTable->Context);
  urGetEnqueueProcAddrTable(version, &pDdiTable->Enqueue);
  urGetEnqueueExpProcAddrTable(version, &pDdiTable->EnqueueExp);
  urGetEventProcAddrTable(version, &pDdiTable->Event);
  urGetKernelProcAddrTable(version, &pDdiTable->Kernel);
  urGetKernelExpProcAddrTable(version, &pDdiTable->KernelExp);
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
  urGetAdapterProcAddrTable(version, &pDdiTable->Adapter);

  return UR_RESULT_SUCCESS;
}

} // extern "C"

const ur_dditable_t *ur::offload::ddi_getter::value() {
  static std::once_flag flag;
  static ur_dditable_t table;

  std::call_once(flag,
                 []() { urAllAddrTable(UR_API_VERSION_CURRENT, &table); });
  return &table;
}
