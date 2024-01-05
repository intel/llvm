//===--------- command_buffer.cpp - OpenCL Adapter ---------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "command_buffer.hpp"
#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_exp_command_buffer_desc_t *pCommandBufferDesc,
    ur_exp_command_buffer_handle_t *phCommandBuffer) {

  ur_queue_handle_t Queue = nullptr;
  UR_RETURN_ON_FAILURE(urQueueCreate(hContext, hDevice, nullptr, &Queue));

  cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
  cl_ext::clCreateCommandBufferKHR_fn clCreateCommandBufferKHR = nullptr;
  cl_int Res =
      cl_ext::getExtFuncFromContext<decltype(clCreateCommandBufferKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clCreateCommandBufferKHRCache,
          cl_ext::CreateCommandBufferName, &clCreateCommandBufferKHR);

  if (!clCreateCommandBufferKHR || Res != CL_SUCCESS)
    return UR_RESULT_ERROR_INVALID_OPERATION;

  auto CLCommandBuffer = clCreateCommandBufferKHR(
      1, cl_adapter::cast<cl_command_queue *>(&Queue), nullptr, &Res);
  CL_RETURN_ON_FAILURE_AND_SET_NULL(Res, phCommandBuffer);

  try {
    auto URCommandBuffer = std::make_unique<ur_exp_command_buffer_handle_t_>(
        Queue, hContext, CLCommandBuffer);
    *phCommandBuffer = URCommandBuffer.release();
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  CL_RETURN_ON_FAILURE(Res);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferRetainExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  UR_RETURN_ON_FAILURE(urQueueRetain(hCommandBuffer->hInternalQueue));

  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clRetainCommandBufferKHR_fn clRetainCommandBuffer = nullptr;
  cl_int Res = cl_ext::getExtFuncFromContext<decltype(clRetainCommandBuffer)>(
      CLContext, cl_ext::ExtFuncPtrCache->clRetainCommandBufferKHRCache,
      cl_ext::RetainCommandBufferName, &clRetainCommandBuffer);

  if (!clRetainCommandBuffer || Res != CL_SUCCESS)
    return UR_RESULT_ERROR_INVALID_OPERATION;

  CL_RETURN_ON_FAILURE(clRetainCommandBuffer(hCommandBuffer->CLCommandBuffer));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferReleaseExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  UR_RETURN_ON_FAILURE(urQueueRelease(hCommandBuffer->hInternalQueue));

  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clReleaseCommandBufferKHR_fn clReleaseCommandBufferKHR = nullptr;
  cl_int Res =
      cl_ext::getExtFuncFromContext<decltype(clReleaseCommandBufferKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clReleaseCommandBufferKHRCache,
          cl_ext::ReleaseCommandBufferName, &clReleaseCommandBufferKHR);

  if (!clReleaseCommandBufferKHR || Res != CL_SUCCESS)
    return UR_RESULT_ERROR_INVALID_OPERATION;

  CL_RETURN_ON_FAILURE(
      clReleaseCommandBufferKHR(hCommandBuffer->CLCommandBuffer));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferFinalizeExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clFinalizeCommandBufferKHR_fn clFinalizeCommandBufferKHR = nullptr;
  cl_int Res =
      cl_ext::getExtFuncFromContext<decltype(clFinalizeCommandBufferKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clFinalizeCommandBufferKHRCache,
          cl_ext::FinalizeCommandBufferName, &clFinalizeCommandBufferKHR);

  if (!clFinalizeCommandBufferKHR || Res != CL_SUCCESS)
    return UR_RESULT_ERROR_INVALID_OPERATION;

  CL_RETURN_ON_FAILURE(
      clFinalizeCommandBufferKHR(hCommandBuffer->CLCommandBuffer));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_kernel_handle_t hKernel,
    uint32_t workDim, const size_t *pGlobalWorkOffset,
    const size_t *pGlobalWorkSize, const size_t *pLocalWorkSize,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clCommandNDRangeKernelKHR_fn clCommandNDRangeKernelKHR = nullptr;
  cl_int Res =
      cl_ext::getExtFuncFromContext<decltype(clCommandNDRangeKernelKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clCommandNDRangeKernelKHRCache,
          cl_ext::CommandNRRangeKernelName, &clCommandNDRangeKernelKHR);

  if (!clCommandNDRangeKernelKHR || Res != CL_SUCCESS)
    return UR_RESULT_ERROR_INVALID_OPERATION;

  CL_RETURN_ON_FAILURE(clCommandNDRangeKernelKHR(
      hCommandBuffer->CLCommandBuffer, nullptr, nullptr,
      cl_adapter::cast<cl_kernel>(hKernel), workDim, pGlobalWorkOffset,
      pGlobalWorkSize, pLocalWorkSize, numSyncPointsInWaitList,
      pSyncPointWaitList, pSyncPoint, nullptr));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMMemcpyExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] void *pDst, [[maybe_unused]] const void *pSrc,
    [[maybe_unused]] size_t size,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMFillExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] void *pMemory, [[maybe_unused]] const void *pPattern,
    [[maybe_unused]] size_t patternSize, [[maybe_unused]] size_t size,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clCommandCopyBufferKHR_fn clCommandCopyBufferKHR = nullptr;
  cl_int Res = cl_ext::getExtFuncFromContext<decltype(clCommandCopyBufferKHR)>(
      CLContext, cl_ext::ExtFuncPtrCache->clCommandCopyBufferKHRCache,
      cl_ext::CommandCopyBufferName, &clCommandCopyBufferKHR);

  if (!clCommandCopyBufferKHR || Res != CL_SUCCESS)
    return UR_RESULT_ERROR_INVALID_OPERATION;

  CL_RETURN_ON_FAILURE(clCommandCopyBufferKHR(
      hCommandBuffer->CLCommandBuffer, nullptr,
      cl_adapter::cast<cl_mem>(hSrcMem), cl_adapter::cast<cl_mem>(hDstMem),
      srcOffset, dstOffset, size, numSyncPointsInWaitList, pSyncPointWaitList,
      pSyncPoint, nullptr));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyRectExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] ur_mem_handle_t hSrcMem,
    [[maybe_unused]] ur_mem_handle_t hDstMem,
    [[maybe_unused]] ur_rect_offset_t srcOrigin,
    [[maybe_unused]] ur_rect_offset_t dstOrigin,
    [[maybe_unused]] ur_rect_region_t region,
    [[maybe_unused]] size_t srcRowPitch, [[maybe_unused]] size_t srcSlicePitch,
    [[maybe_unused]] size_t dstRowPitch, [[maybe_unused]] size_t dstSlicePitch,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  size_t OpenCLOriginRect[3]{srcOrigin.x, srcOrigin.y, srcOrigin.z};
  size_t OpenCLDstRect[3]{dstOrigin.x, dstOrigin.y, dstOrigin.z};
  size_t OpenCLRegion[3]{region.width, region.height, region.depth};

  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clCommandCopyBufferRectKHR_fn clCommandCopyBufferRectKHR = nullptr;
  cl_int Res =
      cl_ext::getExtFuncFromContext<decltype(clCommandCopyBufferRectKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clCommandCopyBufferRectKHRCache,
          cl_ext::CommandCopyBufferRectName, &clCommandCopyBufferRectKHR);

  if (!clCommandCopyBufferRectKHR || Res != CL_SUCCESS)
    return UR_RESULT_ERROR_INVALID_OPERATION;

  CL_RETURN_ON_FAILURE(clCommandCopyBufferRectKHR(
      hCommandBuffer->CLCommandBuffer, nullptr,
      cl_adapter::cast<cl_mem>(hSrcMem), cl_adapter::cast<cl_mem>(hDstMem),
      OpenCLOriginRect, OpenCLDstRect, OpenCLRegion, srcRowPitch, srcSlicePitch,
      dstRowPitch, dstSlicePitch, numSyncPointsInWaitList, pSyncPointWaitList,
      pSyncPoint, nullptr));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] ur_mem_handle_t hBuffer, [[maybe_unused]] size_t offset,
    [[maybe_unused]] size_t size, [[maybe_unused]] const void *pSrc,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] ur_mem_handle_t hBuffer, [[maybe_unused]] size_t offset,
    [[maybe_unused]] size_t size, [[maybe_unused]] void *pDst,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteRectExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] ur_mem_handle_t hBuffer,
    [[maybe_unused]] ur_rect_offset_t bufferOffset,
    [[maybe_unused]] ur_rect_offset_t hostOffset,
    [[maybe_unused]] ur_rect_region_t region,
    [[maybe_unused]] size_t bufferRowPitch,
    [[maybe_unused]] size_t bufferSlicePitch,
    [[maybe_unused]] size_t hostRowPitch,
    [[maybe_unused]] size_t hostSlicePitch, [[maybe_unused]] void *pSrc,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadRectExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] ur_mem_handle_t hBuffer,
    [[maybe_unused]] ur_rect_offset_t bufferOffset,
    [[maybe_unused]] ur_rect_offset_t hostOffset,
    [[maybe_unused]] ur_rect_region_t region,
    [[maybe_unused]] size_t bufferRowPitch,
    [[maybe_unused]] size_t bufferSlicePitch,
    [[maybe_unused]] size_t hostRowPitch,
    [[maybe_unused]] size_t hostSlicePitch, [[maybe_unused]] void *pDst,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    const void *pPattern, size_t patternSize, size_t offset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clCommandFillBufferKHR_fn clCommandFillBufferKHR = nullptr;
  cl_int Res = cl_ext::getExtFuncFromContext<decltype(clCommandFillBufferKHR)>(
      CLContext, cl_ext::ExtFuncPtrCache->clCommandFillBufferKHRCache,
      cl_ext::CommandFillBufferName, &clCommandFillBufferKHR);

  if (!clCommandFillBufferKHR || Res != CL_SUCCESS)
    return UR_RESULT_ERROR_INVALID_OPERATION;

  CL_RETURN_ON_FAILURE(clCommandFillBufferKHR(
      hCommandBuffer->CLCommandBuffer, nullptr,
      cl_adapter::cast<cl_mem>(hBuffer), pPattern, patternSize, offset, size,
      numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint, nullptr));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMPrefetchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, const void *mem, size_t size,
    ur_usm_migration_flags_t flags, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  (void)hCommandBuffer;
  (void)mem;
  (void)size;
  (void)flags;
  (void)numSyncPointsInWaitList;
  (void)pSyncPointWaitList;
  (void)pSyncPoint;

  // Not implemented
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMAdviseExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, const void *mem, size_t size,
    ur_usm_advice_flags_t advice, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  (void)hCommandBuffer;
  (void)mem;
  (void)size;
  (void)advice;
  (void)numSyncPointsInWaitList;
  (void)pSyncPointWaitList;
  (void)pSyncPoint;

  // Not implemented
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_queue_handle_t hQueue,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clEnqueueCommandBufferKHR_fn clEnqueueCommandBufferKHR = nullptr;
  cl_int Res =
      cl_ext::getExtFuncFromContext<decltype(clEnqueueCommandBufferKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clEnqueueCommandBufferKHRCache,
          cl_ext::EnqueueCommandBufferName, &clEnqueueCommandBufferKHR);

  if (!clEnqueueCommandBufferKHR || Res != CL_SUCCESS)
    return UR_RESULT_ERROR_INVALID_OPERATION;

  const uint32_t NumberOfQueues = 1;

  CL_RETURN_ON_FAILURE(clEnqueueCommandBufferKHR(
      NumberOfQueues, cl_adapter::cast<cl_command_queue *>(&hQueue),
      hCommandBuffer->CLCommandBuffer, numEventsInWaitList,
      cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

  return UR_RESULT_SUCCESS;
}
