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

/// The ur_exp_command_buffer_handle_t_ destructor calls CL release
/// command-buffer to free the underlying object.
ur_exp_command_buffer_handle_t_::~ur_exp_command_buffer_handle_t_() {
  urQueueRelease(hInternalQueue);

  cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
  cl_ext::clReleaseCommandBufferKHR_fn clReleaseCommandBufferKHR = nullptr;
  cl_int Res =
      cl_ext::getExtFuncFromContext<decltype(clReleaseCommandBufferKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clReleaseCommandBufferKHRCache,
          cl_ext::ReleaseCommandBufferName, &clReleaseCommandBufferKHR);
  assert(Res == CL_SUCCESS);
  (void)Res;

  clReleaseCommandBufferKHR(CLCommandBuffer);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_exp_command_buffer_desc_t *pCommandBufferDesc,
    ur_exp_command_buffer_handle_t *phCommandBuffer) {

  ur_queue_handle_t Queue = nullptr;
  ur_queue_properties_t QueueProperties = {UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
                                           nullptr, 0};
  const bool IsInOrder =
      pCommandBufferDesc ? pCommandBufferDesc->isInOrder : false;
  if (!IsInOrder) {
    QueueProperties.flags = UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  }
  UR_RETURN_ON_FAILURE(
      urQueueCreate(hContext, hDevice, &QueueProperties, &Queue));

  cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
  cl_ext::clCreateCommandBufferKHR_fn clCreateCommandBufferKHR = nullptr;
  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<decltype(clCreateCommandBufferKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clCreateCommandBufferKHRCache,
          cl_ext::CreateCommandBufferName, &clCreateCommandBufferKHR));

  const bool IsUpdatable = pCommandBufferDesc->isUpdatable;

  ur_device_command_buffer_update_capability_flags_t UpdateCapabilities;
  cl_device_id CLDevice = cl_adapter::cast<cl_device_id>(hDevice);
  CL_RETURN_ON_FAILURE(
      getDeviceCommandBufferUpdateCapabilities(CLDevice, UpdateCapabilities));
  bool DeviceSupportsUpdate = UpdateCapabilities > 0;

  if (IsUpdatable && !DeviceSupportsUpdate) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  cl_command_buffer_properties_khr Properties[3] = {
      CL_COMMAND_BUFFER_FLAGS_KHR,
      IsUpdatable ? CL_COMMAND_BUFFER_MUTABLE_KHR : 0u, 0};

  cl_int Res = CL_SUCCESS;
  auto CLCommandBuffer = clCreateCommandBufferKHR(
      1, cl_adapter::cast<cl_command_queue *>(&Queue), Properties, &Res);
  CL_RETURN_ON_FAILURE_AND_SET_NULL(Res, phCommandBuffer);

  try {
    auto URCommandBuffer = std::make_unique<ur_exp_command_buffer_handle_t_>(
        Queue, hContext, hDevice, CLCommandBuffer, IsUpdatable, IsInOrder);
    *phCommandBuffer = URCommandBuffer.release();
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  CL_RETURN_ON_FAILURE(Res);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferRetainExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  hCommandBuffer->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferReleaseExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  if (hCommandBuffer->decrementReferenceCount() == 0) {
    delete hCommandBuffer;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferFinalizeExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  UR_ASSERT(!hCommandBuffer->IsFinalized, UR_RESULT_ERROR_INVALID_OPERATION);
  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clFinalizeCommandBufferKHR_fn clFinalizeCommandBufferKHR = nullptr;
  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<decltype(clFinalizeCommandBufferKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clFinalizeCommandBufferKHRCache,
          cl_ext::FinalizeCommandBufferName, &clFinalizeCommandBufferKHR));

  CL_RETURN_ON_FAILURE(
      clFinalizeCommandBufferKHR(hCommandBuffer->CLCommandBuffer));
  hCommandBuffer->IsFinalized = true;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_kernel_handle_t hKernel,
    uint32_t workDim, const size_t *pGlobalWorkOffset,
    const size_t *pGlobalWorkSize, const size_t *pLocalWorkSize,
    uint32_t /*numKernelAlternatives*/,
    ur_kernel_handle_t * /*phKernelAlternatives*/,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommandHandle) {
  (void)numEventsInWaitList;
  (void)phEventWaitList;
  (void)phEvent;

  // Command handles can only be obtained from updatable command-buffers
  UR_ASSERT(!(phCommandHandle && !hCommandBuffer->IsUpdatable),
            UR_RESULT_ERROR_INVALID_OPERATION);

  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clCommandNDRangeKernelKHR_fn clCommandNDRangeKernelKHR = nullptr;
  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<decltype(clCommandNDRangeKernelKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clCommandNDRangeKernelKHRCache,
          cl_ext::CommandNRRangeKernelName, &clCommandNDRangeKernelKHR));

  cl_mutable_command_khr CommandHandle = nullptr;
  cl_mutable_command_khr *OutCommandHandle =
      hCommandBuffer->IsUpdatable ? &CommandHandle : nullptr;

  cl_command_properties_khr UpdateProperties[] = {
      CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
      CL_MUTABLE_DISPATCH_GLOBAL_OFFSET_KHR |
          CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR |
          CL_MUTABLE_DISPATCH_LOCAL_SIZE_KHR |
          CL_MUTABLE_DISPATCH_ARGUMENTS_KHR | CL_MUTABLE_DISPATCH_EXEC_INFO_KHR,
      0};

  cl_command_properties_khr *Properties =
      hCommandBuffer->IsUpdatable ? UpdateProperties : nullptr;

  const bool IsInOrder = hCommandBuffer->IsInOrder;
  cl_sync_point_khr *RetSyncPoint = IsInOrder ? nullptr : pSyncPoint;
  const cl_sync_point_khr *SyncPointWaitList =
      IsInOrder ? nullptr : pSyncPointWaitList;
  uint32_t WaitListSize = IsInOrder ? 0 : numSyncPointsInWaitList;
  CL_RETURN_ON_FAILURE(clCommandNDRangeKernelKHR(
      hCommandBuffer->CLCommandBuffer, nullptr, Properties,
      cl_adapter::cast<cl_kernel>(hKernel), workDim, pGlobalWorkOffset,
      pGlobalWorkSize, pLocalWorkSize, WaitListSize, SyncPointWaitList,
      RetSyncPoint, OutCommandHandle));

  try {
    auto Handle = std::make_unique<ur_exp_command_buffer_command_handle_t_>(
        hCommandBuffer, CommandHandle, hKernel, workDim,
        pLocalWorkSize != nullptr);
    if (phCommandHandle) {
      *phCommandHandle = Handle.get();
    }

    hCommandBuffer->CommandHandles.push_back(std::move(Handle));
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMMemcpyExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] void *pDst, [[maybe_unused]] const void *pSrc,
    [[maybe_unused]] size_t size,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint,
    [[maybe_unused]] ur_event_handle_t *phEvent,
    [[maybe_unused]] ur_exp_command_buffer_command_handle_t *phCommand) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMFillExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] void *pMemory, [[maybe_unused]] const void *pPattern,
    [[maybe_unused]] size_t patternSize, [[maybe_unused]] size_t size,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint,
    [[maybe_unused]] ur_event_handle_t *phEvent,
    [[maybe_unused]] ur_exp_command_buffer_command_handle_t *phCommand) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) {
  (void)numEventsInWaitList;
  (void)phEventWaitList;
  (void)phEvent;
  (void)phCommand;
  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clCommandCopyBufferKHR_fn clCommandCopyBufferKHR = nullptr;
  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<decltype(clCommandCopyBufferKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clCommandCopyBufferKHRCache,
          cl_ext::CommandCopyBufferName, &clCommandCopyBufferKHR));

  const bool IsInOrder = hCommandBuffer->IsInOrder;
  cl_sync_point_khr *RetSyncPoint = IsInOrder ? nullptr : pSyncPoint;
  const cl_sync_point_khr *SyncPointWaitList =
      IsInOrder ? nullptr : pSyncPointWaitList;
  uint32_t WaitListSize = IsInOrder ? 0 : numSyncPointsInWaitList;
  CL_RETURN_ON_FAILURE(clCommandCopyBufferKHR(
      hCommandBuffer->CLCommandBuffer, nullptr, nullptr,
      cl_adapter::cast<cl_mem>(hSrcMem), cl_adapter::cast<cl_mem>(hDstMem),
      srcOffset, dstOffset, size, WaitListSize, SyncPointWaitList, RetSyncPoint,
      nullptr));

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
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint,
    [[maybe_unused]] ur_event_handle_t *phEvent,
    [[maybe_unused]] ur_exp_command_buffer_command_handle_t *phCommand) {

  size_t OpenCLOriginRect[3]{srcOrigin.x, srcOrigin.y, srcOrigin.z};
  size_t OpenCLDstRect[3]{dstOrigin.x, dstOrigin.y, dstOrigin.z};
  size_t OpenCLRegion[3]{region.width, region.height, region.depth};

  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clCommandCopyBufferRectKHR_fn clCommandCopyBufferRectKHR = nullptr;
  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<decltype(clCommandCopyBufferRectKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clCommandCopyBufferRectKHRCache,
          cl_ext::CommandCopyBufferRectName, &clCommandCopyBufferRectKHR));

  const bool IsInOrder = hCommandBuffer->IsInOrder;
  cl_sync_point_khr *RetSyncPoint = IsInOrder ? nullptr : pSyncPoint;
  const cl_sync_point_khr *SyncPointWaitList =
      IsInOrder ? nullptr : pSyncPointWaitList;
  uint32_t WaitListSize = IsInOrder ? 0 : numSyncPointsInWaitList;
  CL_RETURN_ON_FAILURE(clCommandCopyBufferRectKHR(
      hCommandBuffer->CLCommandBuffer, nullptr, nullptr,
      cl_adapter::cast<cl_mem>(hSrcMem), cl_adapter::cast<cl_mem>(hDstMem),
      OpenCLOriginRect, OpenCLDstRect, OpenCLRegion, srcRowPitch, srcSlicePitch,
      dstRowPitch, dstSlicePitch, WaitListSize, SyncPointWaitList, RetSyncPoint,
      nullptr));

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
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint,
    [[maybe_unused]] ur_event_handle_t *phEvent,
    [[maybe_unused]] ur_exp_command_buffer_command_handle_t *phCommand) {
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
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint,
    [[maybe_unused]] ur_event_handle_t *phEvent,
    [[maybe_unused]] ur_exp_command_buffer_command_handle_t *phCommand) {
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
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint,
    [[maybe_unused]] ur_event_handle_t *phEvent,
    [[maybe_unused]] ur_exp_command_buffer_command_handle_t *phCommand) {
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
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint,
    [[maybe_unused]] ur_event_handle_t *phEvent,
    [[maybe_unused]] ur_exp_command_buffer_command_handle_t *phCommand) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    const void *pPattern, size_t patternSize, size_t offset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    [[maybe_unused]] ur_event_handle_t *phEvent,
    [[maybe_unused]] ur_exp_command_buffer_command_handle_t *phCommand) {

  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clCommandFillBufferKHR_fn clCommandFillBufferKHR = nullptr;
  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<decltype(clCommandFillBufferKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clCommandFillBufferKHRCache,
          cl_ext::CommandFillBufferName, &clCommandFillBufferKHR));

  const bool IsInOrder = hCommandBuffer->IsInOrder;
  cl_sync_point_khr *RetSyncPoint = IsInOrder ? nullptr : pSyncPoint;
  const cl_sync_point_khr *SyncPointWaitList =
      IsInOrder ? nullptr : pSyncPointWaitList;
  uint32_t WaitListSize = IsInOrder ? 0 : numSyncPointsInWaitList;
  CL_RETURN_ON_FAILURE(clCommandFillBufferKHR(
      hCommandBuffer->CLCommandBuffer, nullptr, nullptr,
      cl_adapter::cast<cl_mem>(hBuffer), pPattern, patternSize, offset, size,
      WaitListSize, SyncPointWaitList, RetSyncPoint, nullptr));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMPrefetchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, const void *mem, size_t size,
    ur_usm_migration_flags_t flags, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) {
  (void)hCommandBuffer;
  (void)mem;
  (void)size;
  (void)flags;
  (void)numSyncPointsInWaitList;
  (void)pSyncPointWaitList;
  (void)numEventsInWaitList;
  (void)phEventWaitList;
  (void)pSyncPoint;
  (void)phEvent;
  (void)phCommand;

  // Not implemented
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMAdviseExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, const void *mem, size_t size,
    ur_usm_advice_flags_t advice, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) {
  (void)hCommandBuffer;
  (void)mem;
  (void)size;
  (void)advice;
  (void)numSyncPointsInWaitList;
  (void)pSyncPointWaitList;
  (void)numEventsInWaitList;
  (void)phEventWaitList;
  (void)pSyncPoint;
  (void)phEvent;
  (void)phCommand;

  // Not implemented
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_queue_handle_t hQueue,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clEnqueueCommandBufferKHR_fn clEnqueueCommandBufferKHR = nullptr;
  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<decltype(clEnqueueCommandBufferKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clEnqueueCommandBufferKHRCache,
          cl_ext::EnqueueCommandBufferName, &clEnqueueCommandBufferKHR));

  const uint32_t NumberOfQueues = 1;

  CL_RETURN_ON_FAILURE(clEnqueueCommandBufferKHR(
      NumberOfQueues, cl_adapter::cast<cl_command_queue *>(&hQueue),
      hCommandBuffer->CLCommandBuffer, numEventsInWaitList,
      cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

  return UR_RESULT_SUCCESS;
}

namespace {
void updateKernelPointerArgs(
    std::vector<cl_mutable_dispatch_arg_khr> &CLUSMArgs,
    const ur_exp_command_buffer_update_kernel_launch_desc_t
        &pUpdateKernelLaunch) {

  // WARNING - This relies on USM and SVM using the same implementation,
  // which is not guaranteed.
  // See https://github.com/KhronosGroup/OpenCL-Docs/issues/843
  const uint32_t NumPointerArgs = pUpdateKernelLaunch.numNewPointerArgs;
  const ur_exp_command_buffer_update_pointer_arg_desc_t *ArgPointerList =
      pUpdateKernelLaunch.pNewPointerArgList;

  CLUSMArgs.resize(NumPointerArgs);
  for (uint32_t i = 0; i < NumPointerArgs; i++) {
    const ur_exp_command_buffer_update_pointer_arg_desc_t &URPointerArg =
        ArgPointerList[i];
    cl_mutable_dispatch_arg_khr &USMArg = CLUSMArgs[i];
    USMArg.arg_index = URPointerArg.argIndex;
    USMArg.arg_value = *(void *const *)URPointerArg.pNewPointerArg;
  }
}

void updateKernelArgs(std::vector<cl_mutable_dispatch_arg_khr> &CLArgs,
                      const ur_exp_command_buffer_update_kernel_launch_desc_t
                          &pUpdateKernelLaunch) {
  const uint32_t NumMemobjArgs = pUpdateKernelLaunch.numNewMemObjArgs;
  const ur_exp_command_buffer_update_memobj_arg_desc_t *ArgMemobjList =
      pUpdateKernelLaunch.pNewMemObjArgList;
  const uint32_t NumValueArgs = pUpdateKernelLaunch.numNewValueArgs;
  const ur_exp_command_buffer_update_value_arg_desc_t *ArgValueList =
      pUpdateKernelLaunch.pNewValueArgList;

  for (uint32_t i = 0; i < NumMemobjArgs; i++) {
    const ur_exp_command_buffer_update_memobj_arg_desc_t &URMemObjArg =
        ArgMemobjList[i];
    cl_mutable_dispatch_arg_khr CLArg{
        URMemObjArg.argIndex, // arg_index
        sizeof(cl_mem),       // arg_size
        cl_adapter::cast<const cl_mem *>(
            &URMemObjArg.hNewMemObjArg) // arg_value
    };

    CLArgs.push_back(CLArg);
  }

  for (uint32_t i = 0; i < NumValueArgs; i++) {
    const ur_exp_command_buffer_update_value_arg_desc_t &URValueArg =
        ArgValueList[i];
    cl_mutable_dispatch_arg_khr CLArg{
        URValueArg.argIndex,    // arg_index
        URValueArg.argSize,     // arg_size
        URValueArg.pNewValueArg // arg_value
    };
    CLArgs.push_back(CLArg);
  }
}

ur_result_t validateCommandDesc(
    ur_exp_command_buffer_handle_t CommandBuffer,
    const ur_exp_command_buffer_update_kernel_launch_desc_t &UpdateDesc) {
  if (!CommandBuffer->IsFinalized || !CommandBuffer->IsUpdatable) {
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  auto Command = UpdateDesc.hCommand;
  if (CommandBuffer != Command->hCommandBuffer) {
    return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP;
  }

  // Kernel handle updates are not yet supported.
  if (UpdateDesc.hNewKernel && UpdateDesc.hNewKernel != Command->Kernel) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  // Error if work-dim has changed but a new global size/offset hasn't been set
  if (UpdateDesc.newWorkDim != Command->WorkDim &&
      (!UpdateDesc.pNewGlobalWorkOffset || !UpdateDesc.pNewGlobalWorkSize)) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  // Verify that the device supports updating the aspects of the kernel that
  // the user is requesting.
  ur_device_handle_t URDevice = CommandBuffer->hDevice;
  cl_device_id CLDevice = cl_adapter::cast<cl_device_id>(URDevice);

  ur_device_command_buffer_update_capability_flags_t UpdateCapabilities = 0;
  CL_RETURN_ON_FAILURE(
      getDeviceCommandBufferUpdateCapabilities(CLDevice, UpdateCapabilities));

  size_t *NewGlobalWorkOffset = UpdateDesc.pNewGlobalWorkOffset;
  UR_ASSERT(
      !NewGlobalWorkOffset ||
          (UpdateCapabilities &
           UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_OFFSET),
      UR_RESULT_ERROR_UNSUPPORTED_FEATURE);

  size_t *NewLocalWorkSize = UpdateDesc.pNewLocalWorkSize;
  UR_ASSERT(
      !NewLocalWorkSize ||
          (UpdateCapabilities &
           UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_LOCAL_WORK_SIZE),
      UR_RESULT_ERROR_UNSUPPORTED_FEATURE);

  size_t *NewGlobalWorkSize = UpdateDesc.pNewGlobalWorkSize;
  UR_ASSERT(
      !NewGlobalWorkSize ||
          (UpdateCapabilities &
           UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_SIZE),
      UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
  UR_ASSERT(
      !(NewGlobalWorkSize && !NewLocalWorkSize) ||
          (UpdateCapabilities &
           UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_LOCAL_WORK_SIZE),
      UR_RESULT_ERROR_UNSUPPORTED_FEATURE);

  UR_ASSERT(
      (!UpdateDesc.numNewMemObjArgs && !UpdateDesc.numNewPointerArgs &&
       !UpdateDesc.numNewValueArgs) ||
          (UpdateCapabilities &
           UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_ARGUMENTS),
      UR_RESULT_ERROR_UNSUPPORTED_FEATURE);

  return UR_RESULT_SUCCESS;
}
} // end anonymous namespace

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferUpdateKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, uint32_t numKernelUpdates,
    const ur_exp_command_buffer_update_kernel_launch_desc_t
        *pUpdateKernelLaunch) {
  for (uint32_t i = 0; i < numKernelUpdates; i++) {
    UR_RETURN_ON_FAILURE(
        validateCommandDesc(hCommandBuffer, pUpdateKernelLaunch[i]));
  }

  cl_context CLContext = cl_adapter::cast<cl_context>(hCommandBuffer->hContext);
  cl_ext::clUpdateMutableCommandsKHR_fn clUpdateMutableCommandsKHR = nullptr;
  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<decltype(clUpdateMutableCommandsKHR)>(
          CLContext, cl_ext::ExtFuncPtrCache->clUpdateMutableCommandsKHRCache,
          cl_ext::UpdateMutableCommandsName, &clUpdateMutableCommandsKHR));

  std::vector<cl_mutable_dispatch_config_khr> ConfigList(numKernelUpdates);
  std::vector<std::vector<cl_mutable_dispatch_arg_khr>> CLUSMArgsList(
      numKernelUpdates);
  std::vector<std::vector<cl_mutable_dispatch_arg_khr>> CLArgsList(
      numKernelUpdates);

  std::vector<std::vector<size_t>> CLGlobalWorkOffsetList(numKernelUpdates);
  std::vector<std::vector<size_t>> CLGlobalWorkSizeList(numKernelUpdates);
  std::vector<std::vector<size_t>> CLLocalWorkSizeList(numKernelUpdates);

  // Lambda for N-Dimensional update
  auto updateNDRange = [](std::vector<size_t> &NDRange, cl_uint WorkDim,
                          size_t *UpdatePtr) {
    NDRange.resize(WorkDim, 0);
    const size_t CopySize = sizeof(size_t) * WorkDim;
    std::memcpy(NDRange.data(), UpdatePtr, CopySize);
  };

  for (uint32_t i = 0; i < numKernelUpdates; i++) {
    cl_mutable_dispatch_config_khr &Config = ConfigList[i];
    std::vector<cl_mutable_dispatch_arg_khr> &CLUSMArgs = CLUSMArgsList[i];
    std::vector<cl_mutable_dispatch_arg_khr> &CLArgs = CLArgsList[i];
    std::vector<size_t> &CLGlobalWorkOffset = CLGlobalWorkOffsetList[i];
    std::vector<size_t> &CLGlobalWorkSize = CLGlobalWorkSizeList[i];
    std::vector<size_t> &CLLocalWorkSize = CLLocalWorkSizeList[i];

    const auto &UpdateDesc = pUpdateKernelLaunch[i];
    // Find the CL USM pointer arguments to the kernel to update
    updateKernelPointerArgs(CLUSMArgs, UpdateDesc);

    // Find the memory object and scalar arguments to the kernel to update
    updateKernelArgs(CLArgs, UpdateDesc);

    // Find the updated ND-Range configuration of the kernel.
    auto Command = UpdateDesc.hCommand;
    cl_uint &CommandWorkDim = Command->WorkDim;

    if (auto GlobalWorkOffsetPtr = UpdateDesc.pNewGlobalWorkOffset) {
      updateNDRange(CLGlobalWorkOffset, CommandWorkDim, GlobalWorkOffsetPtr);
    }

    if (auto GlobalWorkSizePtr = UpdateDesc.pNewGlobalWorkSize) {
      updateNDRange(CLGlobalWorkSize, CommandWorkDim, GlobalWorkSizePtr);
    }

    if (auto LocalWorkSizePtr = UpdateDesc.pNewLocalWorkSize) {
      updateNDRange(CLLocalWorkSize, CommandWorkDim, LocalWorkSizePtr);
    }

    cl_mutable_command_khr CLCommand =
        cl_adapter::cast<cl_mutable_command_khr>(Command->CLMutableCommand);
    Config = cl_mutable_dispatch_config_khr{
        CLCommand,
        static_cast<cl_uint>(CLArgs.size()),    // num_args
        static_cast<cl_uint>(CLUSMArgs.size()), // num_svm_args
        0,                                      // num_exec_infos
        CommandWorkDim,                         // work_dim
        CLArgs.data(),                          // arg_list
        CLUSMArgs.data(),                       // arg_svm_list
        nullptr,                                // exec_info_list
        CLGlobalWorkOffset.data(),              // global_work_offset
        CLGlobalWorkSize.data(),                // global_work_size
        CLLocalWorkSize.data(),                 // local_work_size
    };
  }

  cl_uint NumConfigs = ConfigList.size();
  std::vector<cl_command_buffer_update_type_khr> ConfigTypes(
      NumConfigs, CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR);
  std::vector<const void *> ConfigPtrs(NumConfigs);
  for (cl_uint i = 0; i < NumConfigs; i++) {
    ConfigPtrs[i] = &ConfigList[i];
  }
  CL_RETURN_ON_FAILURE(clUpdateMutableCommandsKHR(
      hCommandBuffer->CLCommandBuffer, NumConfigs, ConfigTypes.data(),
      (const void **)ConfigPtrs.data()));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferUpdateSignalEventExp(
    [[maybe_unused]] ur_exp_command_buffer_command_handle_t Command,
    [[maybe_unused]] ur_event_handle_t *Event) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferUpdateWaitEventsExp(
    [[maybe_unused]] ur_exp_command_buffer_command_handle_t Command,
    [[maybe_unused]] uint32_t NumEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *EventWaitList) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferGetInfoExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,
    ur_exp_command_buffer_info_t propName, size_t propSize, void *pPropValue,
    size_t *pPropSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_EXP_COMMAND_BUFFER_INFO_REFERENCE_COUNT:
    return ReturnValue(hCommandBuffer->getReferenceCount());
  case UR_EXP_COMMAND_BUFFER_INFO_DESCRIPTOR: {
    ur_exp_command_buffer_desc_t Descriptor{};
    Descriptor.stype = UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC;
    Descriptor.pNext = nullptr;
    Descriptor.isUpdatable = hCommandBuffer->IsUpdatable;
    Descriptor.isInOrder = false;
    Descriptor.enableProfiling = false;

    return ReturnValue(Descriptor);
  }
  default:
    assert(false && "Command-buffer info request not implemented");
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}
