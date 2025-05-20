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
#include "../command_buffer_command.hpp"
#include "../helpers/kernel_helpers.hpp"
#include "../ur_interface_loader.hpp"
#include "logger/ur_logger.hpp"
#include "queue_handle.hpp"

namespace {

ur_result_t getZeKernelWrapped(ur_kernel_handle_t kernel,
                               ze_kernel_handle_t &zeKernel,
                               ur_device_handle_t device) {
  zeKernel = kernel->getZeHandle(device);
  return UR_RESULT_SUCCESS;
}

ur_result_t getMemPtr(ur_mem_handle_t memObj,
                      const ur_kernel_arg_mem_obj_properties_t *properties,
                      char **&zeHandlePtr, ur_device_handle_t device,
                      device_ptr_storage_t *ptrStorage) {
  char *ptr;
  if (memObj->isImage()) {
    auto imageObj = memObj->getImage();
    ptr = reinterpret_cast<char *>(imageObj->getZeImage());
  } else {
    auto memBuffer = memObj->getBuffer();
    auto urAccessMode = ur_mem_buffer_t::device_access_mode_t::read_write;
    if (properties != nullptr) {
      urAccessMode =
          ur_mem_buffer_t::getDeviceAccessMode(properties->memoryAccess);
    }
    ptr = ur_cast<char *>(
        memBuffer->getDevicePtr(device, urAccessMode, 0, memBuffer->getSize(),
                                [&](void *, void *, size_t) {}));
  }
  assert(ptrStorage != nullptr);
  ptrStorage->push_back(std::make_unique<char *>(ptr));
  zeHandlePtr = (*ptrStorage)[ptrStorage->size() - 1].get();

  return UR_RESULT_SUCCESS;
}

// Checks whether zeCommandListImmediateAppendCommandListsExp can be used for a
// given context.
void checkImmediateAppendSupport(ur_context_handle_t context) {
  if (!context->getPlatform()->ZeCommandListImmediateAppendExt.Supported) {
    UR_LOG(ERR, "Adapter v2 is used but the current driver does not support "
                "the zeCommandListImmediateAppendCommandListsExp entrypoint.");
    throw UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }
}

} // namespace

ur_exp_command_buffer_handle_t_::ur_exp_command_buffer_handle_t_(
    ur_context_handle_t context, ur_device_handle_t device,
    v2::raii::command_list_unique_handle &&commandList,
    const ur_exp_command_buffer_desc_t *desc)
    : isUpdatable(desc ? desc->isUpdatable : false),
      isInOrder(desc ? desc->isInOrder : false),
      commandListManager(
          context, device,
          std::forward<v2::raii::command_list_unique_handle>(commandList),
          isInOrder ? v2::EVENT_FLAGS_COUNTER : 0, nullptr, false),
      NextSyncPoint(0), context(context), device(device) {}

ur_exp_command_buffer_sync_point_t
ur_exp_command_buffer_handle_t_::getSyncPoint(ur_event_handle_t event) {
  auto syncPoint = NextSyncPoint++;
  syncPoints[syncPoint] = event;
  return syncPoint;
}

ur_event_handle_t *ur_exp_command_buffer_handle_t_::getWaitListFromSyncPoints(
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numSyncPointsInWaitList) {
  if (numSyncPointsInWaitList == 0) {
    return nullptr;
  }
  syncPointWaitList.resize(numSyncPointsInWaitList);
  for (uint32_t i = 0; i < numSyncPointsInWaitList; ++i) {
    auto it = syncPoints.find(pSyncPointWaitList[i]);
    if (it == syncPoints.end()) {
      UR_LOG(ERR, "Invalid sync point");
      throw UR_RESULT_ERROR_INVALID_VALUE;
    }
    syncPointWaitList[i] = it->second;
  }
  return syncPointWaitList.data();
}

ur_result_t ur_exp_command_buffer_handle_t_::createCommandHandle(
    locked<ur_command_list_manager> &commandListLocked,
    ur_kernel_handle_t hKernel, uint32_t workDim, const size_t *pGlobalWorkSize,
    uint32_t numKernelAlternatives, ur_kernel_handle_t *kernelAlternatives,
    ur_exp_command_buffer_command_handle_t *command) {

  auto platform = context->getPlatform();
  ze_command_list_handle_t zeCommandList =
      commandListLocked->getZeCommandList();
  std::unique_ptr<kernel_command_handle> newCommand;
  UR_CALL(createCommandHandleUnlocked(this, zeCommandList, hKernel, workDim,
                                      pGlobalWorkSize, numKernelAlternatives,
                                      kernelAlternatives, platform,
                                      getZeKernelWrapped, device, newCommand));
  *command = newCommand.get();

  commandHandles.push_back(std::move(newCommand));
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_exp_command_buffer_handle_t_::finalizeCommandBuffer() {
  // It is not allowed to append to command list from multiple threads.
  auto commandListLocked = commandListManager.lock();
  UR_ASSERT(!isFinalized, UR_RESULT_ERROR_INVALID_OPERATION);
  // Close the command lists and have them ready for dispatch.
  ZE2UR_CALL(zeCommandListClose, (commandListLocked->getZeCommandList()));
  isFinalized = true;
  return UR_RESULT_SUCCESS;
}
ur_event_handle_t ur_exp_command_buffer_handle_t_::getExecutionEventUnlocked() {
  return currentExecution;
}

ur_result_t ur_exp_command_buffer_handle_t_::registerExecutionEventUnlocked(
    ur_event_handle_t nextExecutionEvent) {
  if (currentExecution) {
    UR_CALL(currentExecution->release());
    currentExecution = nullptr;
  }
  if (nextExecutionEvent) {
    currentExecution = nextExecutionEvent;
    UR_CALL(nextExecutionEvent->retain());
  }
  return UR_RESULT_SUCCESS;
}

ur_exp_command_buffer_handle_t_::~ur_exp_command_buffer_handle_t_() {
  if (currentExecution) {
    currentExecution->release();
  }
}

ur_result_t ur_exp_command_buffer_handle_t_::applyUpdateCommands(
    uint32_t numUpdateCommands,
    const ur_exp_command_buffer_update_kernel_launch_desc_t *updateCommands) {
  auto commandListLocked = commandListManager.lock();
  if (!isFinalized) {
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }
  UR_CALL(validateCommandDescUnlocked(
      this, device, context->getPlatform()->ZeDriverGlobalOffsetExtensionFound,
      numUpdateCommands, updateCommands));

  if (currentExecution) {
    // TODO: Move synchronization to command buffer enqueue
    // it would require to remember the update commands and perform update
    // before appending to the queue
    ZE2UR_CALL(zeEventHostSynchronize,
               (currentExecution->getZeEvent(), UINT64_MAX));
    currentExecution->release();
    currentExecution = nullptr;
  }

  device_ptr_storage_t zeHandles;

  auto platform = context->getPlatform();
  ze_command_list_handle_t zeCommandList =
      commandListLocked->getZeCommandList();
  UR_CALL(updateCommandBufferUnlocked(
      getZeKernelWrapped, getMemPtr, zeCommandList, platform, device,
      &zeHandles, numUpdateCommands, updateCommands));
  ZE2UR_CALL(zeCommandListClose, (zeCommandList));

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
  v2::command_list_desc_t listDesc;
  listDesc.IsInOrder = commandBufferDesc->isInOrder;
  listDesc.Ordinal = queueGroupOrdinal;
  listDesc.CopyOffloadEnable = true;
  listDesc.Mutable = commandBufferDesc->isUpdatable;
  v2::raii::command_list_unique_handle zeCommandList =
      context->getCommandListCache().getRegularCommandList(device->ZeDevice,
                                                           listDesc);

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
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*eventWaitList*/,
    ur_exp_command_buffer_sync_point_t *retSyncPoint,
    ur_event_handle_t * /*event*/,
    ur_exp_command_buffer_command_handle_t *command) try {

  if (command != nullptr && !commandBuffer->isUpdatable) {
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  if (numKernelAlternatives > 0 && kernelAlternatives == nullptr) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  auto commandListLocked = commandBuffer->commandListManager.lock();
  if (command != nullptr) {
    UR_CALL(commandBuffer->createCommandHandle(
        commandListLocked, hKernel, workDim, pGlobalWorkSize,
        numKernelAlternatives, kernelAlternatives, command));
  }
  auto eventsWaitList = commandBuffer->getWaitListFromSyncPoints(
      syncPointWaitList, numSyncPointsInWaitList);
  ur_event_handle_t *event = nullptr;
  ur_event_handle_t signalEvent = nullptr;
  if (retSyncPoint != nullptr) {
    event = &signalEvent;
  }
  UR_CALL(commandListLocked->appendKernelLaunch(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
      numSyncPointsInWaitList, eventsWaitList, event));

  if (retSyncPoint != nullptr) {
    *retSyncPoint = commandBuffer->getSyncPoint(signalEvent);
  }

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendUSMMemcpyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pDst, const void *pSrc,
    size_t size, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  auto eventsWaitList = hCommandBuffer->getWaitListFromSyncPoints(
      pSyncPointWaitList, numSyncPointsInWaitList);
  ur_event_handle_t *event = nullptr;
  ur_event_handle_t signalEvent = nullptr;
  if (pSyncPoint != nullptr) {
    event = &signalEvent;
  }
  UR_CALL(commandListLocked->appendUSMMemcpy(
      false, pDst, pSrc, size, numSyncPointsInWaitList, eventsWaitList, event));

  if (pSyncPoint != nullptr) {
    *pSyncPoint = hCommandBuffer->getSyncPoint(signalEvent);
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp
  // sync mechanic can be ignored, because all lists are in-order
  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  auto eventsWaitList = hCommandBuffer->getWaitListFromSyncPoints(
      pSyncPointWaitList, numSyncPointsInWaitList);
  ur_event_handle_t *event = nullptr;
  ur_event_handle_t signalEvent = nullptr;
  if (pSyncPoint != nullptr) {
    event = &signalEvent;
  }
  UR_CALL(commandListLocked->appendMemBufferCopy(
      hSrcMem, hDstMem, srcOffset, dstOffset, size, numSyncPointsInWaitList,
      eventsWaitList, event));

  if (pSyncPoint != nullptr) {
    *pSyncPoint = hCommandBuffer->getSyncPoint(signalEvent);
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferWriteExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, const void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp
  // sync mechanic can be ignored, because all lists are in-order
  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  auto eventsWaitList = hCommandBuffer->getWaitListFromSyncPoints(
      pSyncPointWaitList, numSyncPointsInWaitList);
  ur_event_handle_t *event = nullptr;
  ur_event_handle_t signalEvent = nullptr;
  if (pSyncPoint != nullptr) {
    event = &signalEvent;
  }
  UR_CALL(commandListLocked->appendMemBufferWrite(hBuffer, false, offset, size,
                                                  pSrc, numSyncPointsInWaitList,
                                                  eventsWaitList, event));

  if (pSyncPoint != nullptr) {
    *pSyncPoint = hCommandBuffer->getSyncPoint(signalEvent);
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferReadExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, void *pDst, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp
  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  auto eventsWaitList = hCommandBuffer->getWaitListFromSyncPoints(
      pSyncPointWaitList, numSyncPointsInWaitList);
  ur_event_handle_t *event = nullptr;
  ur_event_handle_t signalEvent = nullptr;
  if (pSyncPoint != nullptr) {
    event = &signalEvent;
  }
  UR_CALL(commandListLocked->appendMemBufferRead(hBuffer, false, offset, size,
                                                 pDst, numSyncPointsInWaitList,
                                                 eventsWaitList, event));

  if (pSyncPoint != nullptr) {
    *pSyncPoint = hCommandBuffer->getSyncPoint(signalEvent);
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferCopyRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp
  // sync mechanic can be ignored, because all lists are in-order
  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  auto eventsWaitList = hCommandBuffer->getWaitListFromSyncPoints(
      pSyncPointWaitList, numSyncPointsInWaitList);
  ur_event_handle_t *event = nullptr;
  ur_event_handle_t signalEvent = nullptr;
  if (pSyncPoint != nullptr) {
    event = &signalEvent;
  }
  UR_CALL(commandListLocked->appendMemBufferCopyRect(
      hSrcMem, hDstMem, srcOrigin, dstOrigin, region, srcRowPitch,
      srcSlicePitch, dstRowPitch, dstSlicePitch, numSyncPointsInWaitList,
      eventsWaitList, event));

  if (pSyncPoint != nullptr) {
    *pSyncPoint = hCommandBuffer->getSyncPoint(signalEvent);
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferWriteRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp

  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  auto eventsWaitList = hCommandBuffer->getWaitListFromSyncPoints(
      pSyncPointWaitList, numSyncPointsInWaitList);
  ur_event_handle_t *event = nullptr;
  ur_event_handle_t signalEvent = nullptr;
  if (pSyncPoint != nullptr) {
    event = &signalEvent;
  }
  UR_CALL(commandListLocked->appendMemBufferWriteRect(
      hBuffer, false, bufferOffset, hostOffset, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc,
      numSyncPointsInWaitList, eventsWaitList, event));

  if (pSyncPoint != nullptr) {
    *pSyncPoint = hCommandBuffer->getSyncPoint(signalEvent);
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferReadRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp

  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  auto eventsWaitList = hCommandBuffer->getWaitListFromSyncPoints(
      pSyncPointWaitList, numSyncPointsInWaitList);
  ur_event_handle_t *event = nullptr;
  ur_event_handle_t signalEvent = nullptr;
  if (pSyncPoint != nullptr) {
    event = &signalEvent;
  }
  UR_CALL(commandListLocked->appendMemBufferReadRect(
      hBuffer, false, bufferOffset, hostOffset, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst,
      numSyncPointsInWaitList, eventsWaitList, event));

  if (pSyncPoint != nullptr) {
    *pSyncPoint = hCommandBuffer->getSyncPoint(signalEvent);
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendUSMFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pMemory,
    const void *pPattern, size_t patternSize, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  auto eventsWaitList = hCommandBuffer->getWaitListFromSyncPoints(
      pSyncPointWaitList, numSyncPointsInWaitList);
  ur_event_handle_t *event = nullptr;
  ur_event_handle_t signalEvent = nullptr;
  if (pSyncPoint != nullptr) {
    event = &signalEvent;
  }
  UR_CALL(commandListLocked->appendUSMFill(pMemory, patternSize, pPattern, size,
                                           numSyncPointsInWaitList,
                                           eventsWaitList, event));
  if (pSyncPoint != nullptr) {
    *pSyncPoint = hCommandBuffer->getSyncPoint(signalEvent);
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    const void *pPattern, size_t patternSize, size_t offset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  auto eventsWaitList = hCommandBuffer->getWaitListFromSyncPoints(
      pSyncPointWaitList, numSyncPointsInWaitList);
  ur_event_handle_t *event = nullptr;
  ur_event_handle_t signalEvent = nullptr;
  if (pSyncPoint != nullptr) {
    event = &signalEvent;
  }
  UR_CALL(commandListLocked->appendMemBufferFill(
      hBuffer, pPattern, patternSize, offset, size, numSyncPointsInWaitList,
      eventsWaitList, event));
  if (pSyncPoint != nullptr) {
    *pSyncPoint = hCommandBuffer->getSyncPoint(signalEvent);
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendUSMPrefetchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, const void *pMemory,
    size_t size, ur_usm_migration_flags_t flags,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp

  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  auto eventsWaitList = hCommandBuffer->getWaitListFromSyncPoints(
      pSyncPointWaitList, numSyncPointsInWaitList);
  ur_event_handle_t *event = nullptr;
  ur_event_handle_t signalEvent = nullptr;
  if (pSyncPoint != nullptr) {
    event = &signalEvent;
  }
  UR_CALL(commandListLocked->appendUSMPrefetch(
      pMemory, size, flags, numSyncPointsInWaitList, eventsWaitList, event));

  if (pSyncPoint != nullptr) {
    *pSyncPoint = hCommandBuffer->getSyncPoint(signalEvent);
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendUSMAdviseExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, const void *pMemory,
    size_t size, ur_usm_advice_flags_t advice, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {
  // the same issue as in urCommandBufferAppendKernelLaunchExp

  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  auto eventsWaitList = hCommandBuffer->getWaitListFromSyncPoints(
      pSyncPointWaitList, numSyncPointsInWaitList);
  ur_event_handle_t *event = nullptr;
  ur_event_handle_t signalEvent = nullptr;
  if (pSyncPoint != nullptr) {
    event = &signalEvent;
  }
  UR_CALL(commandListLocked->appendUSMAdvise(
      pMemory, size, advice, numSyncPointsInWaitList, eventsWaitList, event));

  if (pSyncPoint != nullptr) {
    *pSyncPoint = hCommandBuffer->getSyncPoint(signalEvent);
  }
  return UR_RESULT_SUCCESS;
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
    Descriptor.isInOrder = hCommandBuffer->isInOrder;
    Descriptor.enableProfiling = hCommandBuffer->isProfilingEnabled;

    return ReturnValue(Descriptor);
  }
  default:
    assert(false && "Command-buffer info request not implemented");
  }
  return UR_RESULT_ERROR_INVALID_ENUMERATION;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendNativeCommandExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,
    ur_exp_command_buffer_native_command_function_t pfnNativeCommand,
    void *pData, ur_exp_command_buffer_handle_t,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  // Barrier on all commands before user defined commands.

  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  auto eventsWaitList = hCommandBuffer->getWaitListFromSyncPoints(
      pSyncPointWaitList, numSyncPointsInWaitList);
  ur_event_handle_t *event = nullptr;
  ur_event_handle_t signalEvent = nullptr;
  if (pSyncPoint != nullptr) {
    event = &signalEvent;
  }
  UR_CALL(commandListLocked->appendBarrier(numSyncPointsInWaitList,
                                           eventsWaitList, nullptr));

  // Call user-defined function immediately
  pfnNativeCommand(pData);

  // Barrier on all commands after user defined commands.
  UR_CALL(commandListLocked->appendBarrier(0, nullptr, event));

  if (pSyncPoint != nullptr) {
    *pSyncPoint = hCommandBuffer->getSyncPoint(signalEvent);
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t
urCommandBufferGetNativeHandleExp(ur_exp_command_buffer_handle_t hCommandBuffer,
                                  ur_native_handle_t *phNativeCommandBuffer) {

  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  ze_command_list_handle_t ZeCommandList =
      commandListLocked->getZeCommandList();
  *phNativeCommandBuffer = reinterpret_cast<ur_native_handle_t>(ZeCommandList);
  return UR_RESULT_SUCCESS;
}

ur_result_t urCommandBufferUpdateKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, uint32_t numUpdateCommands,
    const ur_exp_command_buffer_update_kernel_launch_desc_t
        *pUpdateKernelLaunch) {
  UR_CALL(hCommandBuffer->applyUpdateCommands(numUpdateCommands,
                                              pUpdateKernelLaunch));
  return UR_RESULT_SUCCESS;
}

ur_result_t urCommandBufferUpdateSignalEventExp(
    ur_exp_command_buffer_command_handle_t hCommand,
    ur_event_handle_t *phEvent) {
  // needs to be implemented together with signal event handling
  (void)hCommand;
  (void)phEvent;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urCommandBufferUpdateWaitEventsExp(
    ur_exp_command_buffer_command_handle_t hCommand,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList) {
  // needs to be implemented together with wait event handling
  (void)hCommand;
  (void)numEventsInWaitList;
  (void)phEventWaitList;

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::level_zero
