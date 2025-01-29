//===--------- command_buffer.cpp - Level Zero Adapter ---------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "command_buffer.hpp"
#include "../helpers/kernel_helpers.hpp"
#include "../ur_interface_loader.hpp"
#include "logger/ur_logger.hpp"

namespace {

// Checks whether zeCommandListImmediateAppendCommandListsExp can be used for a
// given context.
void checkImmediateAppendSupport(ur_context_handle_t context) {
  if (!context->getPlatform()->ZeCommandListImmediateAppendExt.Supported) {
    logger::error("Adapter v2 is used but "
                  "the current driver does not support the "
                  "zeCommandListImmediateAppendCommandListsExp entrypoint.");
    throw UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }
}

} // namespace

ur_exp_command_buffer_handle_t_::ur_exp_command_buffer_handle_t_(
    ur_context_handle_t context, ur_device_handle_t device,
    v2::raii::command_list_unique_handle &&commandList,
    const ur_exp_command_buffer_desc_t *desc)
    : commandListManager(
          context, device,
          std::forward<v2::raii::command_list_unique_handle>(commandList)),
      isUpdatable(desc ? desc->isUpdatable : false) {}

ur_result_t ur_exp_command_buffer_handle_t_::finalizeCommandBuffer() {
  // It is not allowed to append to command list from multiple threads.
  std::scoped_lock<ur_shared_mutex> guard(this->Mutex);
  UR_ASSERT(!isFinalized, UR_RESULT_ERROR_INVALID_OPERATION);
  // Close the command lists and have them ready for dispatch.
  ZE2UR_CALL(zeCommandListClose, (this->commandListManager.getZeCommandList()));
  isFinalized = true;
  return UR_RESULT_SUCCESS;
}

namespace ur::level_zero {

ur_result_t
urCommandBufferCreateExp(ur_context_handle_t context, ur_device_handle_t device,
                         const ur_exp_command_buffer_desc_t *commandBufferDesc,
                         ur_exp_command_buffer_handle_t *commandBuffer) try {
  checkImmediateAppendSupport(context);

  if (commandBufferDesc->isUpdatable &&
      !context->getPlatform()->ZeMutableCmdListExt.Supported) {
    throw UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  using queue_group_type = ur_device_handle_t_::queue_group_info_t::type;
  uint32_t queueGroupOrdinal =
      device->QueueGroup[queue_group_type::Compute].ZeOrdinal;
  v2::raii::command_list_unique_handle zeCommandList =
      context->commandListCache.getRegularCommandList(device->ZeDevice, true,
                                                      queueGroupOrdinal, true);

  *commandBuffer = new ur_exp_command_buffer_handle_t_(
      context, device, std::move(zeCommandList), commandBufferDesc);
  return UR_RESULT_SUCCESS;

} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t
urCommandBufferRetainExp(ur_exp_command_buffer_handle_t hCommandBuffer) try {
  hCommandBuffer->RefCount.increment();
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t
urCommandBufferReleaseExp(ur_exp_command_buffer_handle_t hCommandBuffer) try {
  if (!hCommandBuffer->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  delete hCommandBuffer;
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t
urCommandBufferFinalizeExp(ur_exp_command_buffer_handle_t hCommandBuffer) try {
  UR_ASSERT(hCommandBuffer, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_CALL(hCommandBuffer->finalizeCommandBuffer());
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t commandBuffer, ur_kernel_handle_t hKernel,
    uint32_t workDim, const size_t *pGlobalWorkOffset,
    const size_t *pGlobalWorkSize, const size_t *pLocalWorkSize,
    uint32_t numKernelAlternatives, ur_kernel_handle_t *kernelAlternatives,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *syncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *eventWaitList,
    ur_exp_command_buffer_sync_point_t *retSyncPoint, ur_event_handle_t *event,
    ur_exp_command_buffer_command_handle_t *command) try {
  // TODO: These parameters aren't implemented in V1 yet, and are a fair amount
  // of work. Need to know semantics: should they be checked before kernel
  // execution (difficult) or before kernel appending to list (easy fix).
  std::ignore = numEventsInWaitList;
  std::ignore = eventWaitList;
  std::ignore = event;

  // sync mechanic can be ignored, because all lists are in-order
  std::ignore = numSyncPointsInWaitList;
  std::ignore = syncPointWaitList;
  std::ignore = retSyncPoint;

  // TODO
  std::ignore = numKernelAlternatives;
  std::ignore = kernelAlternatives;
  std::ignore = command;
  UR_CALL(commandBuffer->commandListManager.appendKernelLaunch(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize, 0,
      nullptr, nullptr));
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_queue_handle_t hQueue,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) try {
  return hQueue->enqueueCommandBuffer(
      hCommandBuffer->commandListManager.getZeCommandList(), phEvent,
      numEventsInWaitList, phEventWaitList);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t
urCommandBufferGetInfoExp(ur_exp_command_buffer_handle_t hCommandBuffer,
                          ur_exp_command_buffer_info_t propName,
                          size_t propSize, void *pPropValue,
                          size_t *pPropSizeRet) try {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_EXP_COMMAND_BUFFER_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{hCommandBuffer->RefCount.load()});
  case UR_EXP_COMMAND_BUFFER_INFO_DESCRIPTOR: {
    ur_exp_command_buffer_desc_t Descriptor{};
    Descriptor.stype = UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC;
    Descriptor.pNext = nullptr;
    Descriptor.isUpdatable = hCommandBuffer->isUpdatable;
    Descriptor.isInOrder = true;
    Descriptor.enableProfiling = hCommandBuffer->isProfilingEnabled;

    return ReturnValue(Descriptor);
  }
  default:
    assert(!"Command-buffer info request not implemented");
  }
  return UR_RESULT_ERROR_INVALID_ENUMERATION;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

} // namespace ur::level_zero
