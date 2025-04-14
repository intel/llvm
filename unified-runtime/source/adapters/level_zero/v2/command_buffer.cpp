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
#include "../helpers/mutable_helpers.hpp"
#include "../ur_interface_loader.hpp"
#include "logger/ur_logger.hpp"
#include "queue_handle.hpp"

namespace {

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

ur_result_t ur_exp_command_buffer_handle_t_::updateKernelSizes(
    const ur_exp_command_buffer_update_kernel_launch_desc_t commandDesc,
    kernel_command_handle *command, void **nextDesc,
    ze_group_count_t &zeThreadGroupDimensionsList, desc_storage_t &descs) {
  uint32_t dim = commandDesc.newWorkDim;
  // Update global offset if provided.
  if (size_t *newGlobalWorkOffset = commandDesc.pNewGlobalWorkOffset;
      newGlobalWorkOffset && dim > 0) {
    auto mutableGroupOffestDesc =
        std::make_unique<ZeStruct<ze_mutable_global_offset_exp_desc_t>>();
    UR_CALL(setMutableOffsetDesc(mutableGroupOffestDesc, dim,
                                 newGlobalWorkOffset, *nextDesc,
                                 command->commandId));
    *nextDesc = mutableGroupOffestDesc.get();
    descs.push_back(std::move(mutableGroupOffestDesc));
  }

  // Update local-size/group-size if provided.
  size_t *newLocalWorkSize = commandDesc.pNewLocalWorkSize;
  if (newLocalWorkSize && dim > 0) {
    auto mutableGroupSizeDesc =
        std::make_unique<ZeStruct<ze_mutable_group_size_exp_desc_t>>();

    uint32_t workgroupSize[3] = {1, 1, 1};
    for (size_t d = 0; d < dim; d++) {
      workgroupSize[d] = newLocalWorkSize[d];
    }

    UR_CALL(setMutableGroupSizeDesc(mutableGroupSizeDesc, dim, workgroupSize,
                                    *nextDesc, command->commandId));
    *nextDesc = mutableGroupSizeDesc.get();
    descs.push_back(std::move(mutableGroupSizeDesc));
  }

  // Update global-size/group-count if provided, and also
  // local-size/group-size if required
  if (size_t *newGlobalWorkSize = commandDesc.pNewGlobalWorkSize;
      (newGlobalWorkSize || newLocalWorkSize) && dim > 0) {

    // If a new global work size is provided update that in the command,
    // otherwise the previous work group size will be used
    if (newGlobalWorkSize) {
      command->workDim = dim;
      command->setGlobalWorkSize(newGlobalWorkSize);
    }

    // If a new global work size is provided but a new local work size is not
    // then we still need to update local work size based on the size
    // suggested
    // by the driver for the kernel.
    bool updateWGSize = newLocalWorkSize == nullptr;

    ze_kernel_handle_t zeKernel = command->kernel->getZeHandle(device);

    uint32_t workgroupSize[3];

    UR_CALL(calculateKernelWorkDimensions(
        zeKernel, device, zeThreadGroupDimensionsList, workgroupSize, dim,
        command->globalWorkSize, newLocalWorkSize));

    auto mutableGroupCountDesc =
        std::make_unique<ZeStruct<ze_mutable_group_count_exp_desc_t>>();
    UR_CALL(setMutableGroupCountDesc(mutableGroupCountDesc,
                                     &zeThreadGroupDimensionsList, *nextDesc,
                                     command->commandId));
    *nextDesc = mutableGroupCountDesc.get();
    descs.push_back(std::move(mutableGroupCountDesc));

    if (updateWGSize) {
      auto mutableGroupSizeDesc =
          std::make_unique<ZeStruct<ze_mutable_group_size_exp_desc_t>>();
      UR_CALL(setMutableGroupSizeDesc(mutableGroupSizeDesc, dim, workgroupSize,
                                      *nextDesc, command->commandId));
      *nextDesc = mutableGroupSizeDesc.get();
      descs.push_back(std::move(mutableGroupSizeDesc));
    }
  }
  return UR_RESULT_SUCCESS;
}
ur_result_t ur_exp_command_buffer_handle_t_::updateKernelArguments(
    locked<ur_command_list_manager> &commandListLocked,
    const ur_exp_command_buffer_update_kernel_launch_desc_t commandDesc,
    kernel_command_handle *command, void **nextDesc,
    device_ptr_storage_t &zeHandles, desc_storage_t &descs) {
  for (uint32_t newMemObjArgNum = commandDesc.numNewMemObjArgs;
       newMemObjArgNum-- > 0;) {
    ur_exp_command_buffer_update_memobj_arg_desc_t newMemObjArgDesc =
        commandDesc.pNewMemObjArgList[newMemObjArgNum];

    auto zeMutableArgDesc =
        std::make_unique<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>>();

    ur_mem_handle_t newMemObjArg = newMemObjArgDesc.hNewMemObjArg;

    // The newMemObjArg may be a NULL pointer in which case a NULL value is
    // used for the kernel argument declared as a pointer to global or
    // constant memory.
    char **zeHandlePtr = nullptr;
    if (newMemObjArg) {
      // TODO: add support for images
      assert(!newMemObjArg->isImage());
      auto memBuffer = newMemObjArg->getBuffer();

      const ur_kernel_arg_mem_obj_properties_t *properties =
          newMemObjArgDesc.pProperties;
      auto urAccessMode = ur_mem_buffer_t::device_access_mode_t::read_write;
      if (properties != nullptr) {
        urAccessMode =
            ur_mem_buffer_t::getDeviceAccessMode(properties->memoryAccess);
      }
      auto ptr = ur_cast<char *>(memBuffer->getDevicePtr(
          device, urAccessMode, 0, memBuffer->getSize(),
          [&](void *src, void *dst, size_t size) {
            ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                              (commandListLocked->getZeCommandList(), dst, src,
                               size, nullptr, 0, nullptr));
          }));
      zeHandles.push_back(std::make_unique<char *>(ptr));
      zeHandlePtr = zeHandles[zeHandles.size() - 1].get();
    }

    UR_CALL(setMutableMemObjArgDesc(zeMutableArgDesc, newMemObjArgDesc.argIndex,
                                    zeHandlePtr, *nextDesc,
                                    command->commandId));
    *nextDesc = zeMutableArgDesc.get();
    descs.push_back(std::move(zeMutableArgDesc));
  }

  // Update pointer arguments if provided.
  for (uint32_t newPointerArgNum = commandDesc.numNewPointerArgs;
       newPointerArgNum-- > 0;) {
    ur_exp_command_buffer_update_pointer_arg_desc_t newPointerArgDesc =
        commandDesc.pNewPointerArgList[newPointerArgNum];

    auto zeMutableArgDesc =
        std::make_unique<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>>();

    UR_CALL(setMutablePointerArgDesc(zeMutableArgDesc, newPointerArgDesc,
                                     *nextDesc, command->commandId));

    *nextDesc = zeMutableArgDesc.get();
    descs.push_back(std::move(zeMutableArgDesc));
  }

  // Update value arguments if provided.
  for (uint32_t newValueArgNum = commandDesc.numNewValueArgs;
       newValueArgNum-- > 0;) {
    ur_exp_command_buffer_update_value_arg_desc_t newValueArgDesc =
        commandDesc.pNewValueArgList[newValueArgNum];

    auto zeMutableArgDesc =
        std::make_unique<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>>();

    UR_CALL(setMutableValueArgDesc(zeMutableArgDesc, newValueArgDesc, *nextDesc,
                                   command->commandId));

    *nextDesc = zeMutableArgDesc.get();
    descs.push_back(std::move(zeMutableArgDesc));
  }
  return UR_RESULT_SUCCESS;
}
ur_exp_command_buffer_handle_t_::ur_exp_command_buffer_handle_t_(
    ur_context_handle_t context, ur_device_handle_t device,
    v2::raii::command_list_unique_handle &&commandList,
    const ur_exp_command_buffer_desc_t *desc)
    : commandListManager(
          context, device,
          std::forward<v2::raii::command_list_unique_handle>(commandList),
          v2::EVENT_FLAGS_COUNTER, nullptr),
      isUpdatable(desc ? desc->isUpdatable : false), context(context),
      device(device) {}

ur_result_t ur_exp_command_buffer_handle_t_::updateKernelHandle(
    locked<ur_command_list_manager> &commandListLocked,
    ur_kernel_handle_t newKernel, kernel_command_handle *command) {
  ze_kernel_handle_t kernelHandle = newKernel->getZeHandle(device);
  auto platform = context->getPlatform();
  auto commandId = command->commandId;
  ZE2UR_CALL(
      platform->ZeMutableCmdListExt
          .zexCommandListUpdateMutableCommandKernelsExp,
      (commandListLocked->getZeCommandList(), 1, &commandId, &kernelHandle));
  // Set current kernel to be the new kernel
  command->kernel = newKernel;
  return UR_RESULT_SUCCESS;
}
ur_result_t ur_exp_command_buffer_handle_t_::createCommandHandle(
    locked<ur_command_list_manager> &commandListLocked,
    ur_kernel_handle_t hKernel, uint32_t workDim, const size_t *pGlobalWorkSize,
    uint32_t numKernelAlternatives, ur_kernel_handle_t *kernelAlternatives,
    ur_exp_command_buffer_command_handle_t *command) {

  uint64_t commandId = 0;
  ZeStruct<ze_mutable_command_id_exp_desc_t> zeMutableCommandDesc;
  zeMutableCommandDesc.flags = ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS |
                               ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT |
                               ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE |
                               ZE_MUTABLE_COMMAND_EXP_FLAG_GLOBAL_OFFSET;

  auto platform = context->getPlatform();
  ze_command_list_handle_t zeCommandList =
      commandListLocked->getZeCommandList();

  if (numKernelAlternatives > 0) {
    zeMutableCommandDesc.flags |=
        ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_INSTRUCTION;

    std::vector<ze_kernel_handle_t> kernelHandles(numKernelAlternatives + 1,
                                                  nullptr);

    kernelHandles[0] = hKernel->getZeHandle(device);

    for (size_t i = 0; i < numKernelAlternatives; i++) {
      if (hKernel == kernelAlternatives[i]) {
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
      kernelHandles[i + 1] = kernelAlternatives[i]->getZeHandle(device);
    }

    ZE2UR_CALL(platform->ZeMutableCmdListExt
                   .zexCommandListGetNextCommandIdWithKernelsExp,
               (zeCommandList, &zeMutableCommandDesc, numKernelAlternatives + 1,
                kernelHandles.data(), &commandId));

  } else {
    ZE2UR_CALL(platform->ZeMutableCmdListExt.zexCommandListGetNextCommandIdExp,
               (zeCommandList, &zeMutableCommandDesc, &commandId));
  }

  auto newCommand = std::make_unique<kernel_command_handle>(
      this, hKernel, commandId, workDim, numKernelAlternatives,
      kernelAlternatives);

  newCommand->setGlobalWorkSize(pGlobalWorkSize);

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

ur_result_t ur_exp_command_buffer_handle_t_::checkUpdateParameters(
    uint32_t numUpdateCommands,
    const ur_exp_command_buffer_update_kernel_launch_desc_t *updateCommands) {
  auto supportedFeatures =
      device->ZeDeviceMutableCmdListsProperties->mutableCommandFlags;
  logger::debug("Mutable features supported by device {}", supportedFeatures);
  for (std::size_t i = 0; i < numUpdateCommands; i++) {
    auto commandDesc = updateCommands[i];
    auto command = static_cast<kernel_command_handle *>(commandDesc.hCommand);
    UR_ASSERT(this == command->commandBuffer,
              UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP);

    UR_ASSERT(!commandDesc.hNewKernel ||
                  (supportedFeatures &
                   ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_INSTRUCTION),
              UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
    // Check if the provided new kernel is in the list of valid alternatives.
    if (commandDesc.hNewKernel &&
        !command->validKernelHandles.count(commandDesc.hNewKernel)) {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    if (commandDesc.newWorkDim != command->workDim &&
        (!commandDesc.pNewGlobalWorkOffset ||
         !commandDesc.pNewGlobalWorkSize)) {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    // Check if new global offset is provided.
    size_t *newGlobalWorkOffset = commandDesc.pNewGlobalWorkOffset;
    UR_ASSERT(
        !newGlobalWorkOffset ||
            (supportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_GLOBAL_OFFSET),
        UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
    if (newGlobalWorkOffset) {
      if (!context->getPlatform()->ZeDriverGlobalOffsetExtensionFound) {
        logger::error("No global offset extension found on this driver");
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
    }

    // Check if new group size is provided.
    size_t *newLocalWorkSize = commandDesc.pNewLocalWorkSize;
    UR_ASSERT(!newLocalWorkSize ||
                  (supportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE),
              UR_RESULT_ERROR_UNSUPPORTED_FEATURE);

    // Check if new global size is provided and we need to update group count.
    size_t *newGlobalWorkSize = commandDesc.pNewGlobalWorkSize;
    UR_ASSERT(!newGlobalWorkSize ||
                  (supportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT),
              UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
    UR_ASSERT(!(newGlobalWorkSize && !newLocalWorkSize) ||
                  (supportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE),
              UR_RESULT_ERROR_UNSUPPORTED_FEATURE);

    UR_ASSERT(
        (!commandDesc.numNewMemObjArgs && !commandDesc.numNewPointerArgs &&
         !commandDesc.numNewValueArgs) ||
            (supportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS),
        UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_exp_command_buffer_handle_t_::applyUpdateCommands(
    uint32_t numUpdateCommands,
    const ur_exp_command_buffer_update_kernel_launch_desc_t *updateCommands) {
  auto commandListLocked = commandListManager.lock();
  if (!isFinalized) {
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }
  UR_CALL(checkUpdateParameters(numUpdateCommands, updateCommands));

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
  desc_storage_t descs;

  std::vector<ze_group_count_t> zeThreadGroupDimensionsList(
      numUpdateCommands, ze_group_count_t{1, 1, 1});
  void *nextDesc = nullptr; // Used for pointer chaining
  // Iterate over every UR update descriptor struct, which corresponds to
  // several L0 update descriptor structs.
  for (uint32_t i = 0; i < numUpdateCommands; i++) {
    const auto &commandDesc = updateCommands[i];
    auto command = static_cast<kernel_command_handle *>(commandDesc.hCommand);

    std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Guard(
        command->Mutex, command->kernel->Mutex);

    // Kernel handle must be updated first for a given CommandId if required
    ur_kernel_handle_t newKernel = commandDesc.hNewKernel;
    if (newKernel && command->kernel != newKernel) {
      updateKernelHandle(commandListLocked, newKernel, command);
    }
    updateKernelSizes(commandDesc, command, &nextDesc,
                      zeThreadGroupDimensionsList[i], descs);
    updateKernelArguments(commandListLocked, commandDesc, command, &nextDesc,
                          zeHandles, descs);
  }

  auto platform = context->getPlatform();
  ze_command_list_handle_t zeCommandList =
      commandListLocked->getZeCommandList();

  ZeStruct<ze_mutable_commands_exp_desc_t> mutableCommandDesc{};
  mutableCommandDesc.pNext = nextDesc;
  mutableCommandDesc.flags = 0;
  ZE2UR_CALL(
      platform->ZeMutableCmdListExt.zexCommandListUpdateMutableCommandsExp,
      (zeCommandList, &mutableCommandDesc));

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
  listDesc.IsInOrder = true;
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
    uint32_t /*numKernelAlternatives*/,
    ur_kernel_handle_t * /*kernelAlternatives*/,
    uint32_t /*numSyncPointsInWaitList*/,
    const ur_exp_command_buffer_sync_point_t * /*syncPointWaitList*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*eventWaitList*/,
    ur_exp_command_buffer_sync_point_t * /*retSyncPoint*/,
    ur_event_handle_t * /*event*/,
    ur_exp_command_buffer_command_handle_t * /*command*/) try {
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

  if (command != nullptr && !commandBuffer->isUpdatable) {
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  if (numKernelAlternatives > 0 && command == nullptr) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  auto commandListLocked = commandBuffer->commandListManager.lock();
  if (command != nullptr) {
    UR_CALL(commandBuffer->createCommandHandle(
        commandListLocked, hKernel, workDim, pGlobalWorkSize,
        numKernelAlternatives, kernelAlternatives, command));
  }
  UR_CALL(commandListLocked->appendKernelLaunch(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize, 0,
      nullptr, nullptr));
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendUSMMemcpyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pDst, const void *pSrc,
    size_t size, uint32_t /*numSyncPointsInWaitList*/,
    const ur_exp_command_buffer_sync_point_t * /*pSyncPointWaitList*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t * /*pSyncPoint*/,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  UR_CALL(commandListLocked->appendUSMMemcpy(false, pDst, pSrc, size, 0,
                                             nullptr, nullptr));

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t /*numSyncPointsInWaitList*/,
    const ur_exp_command_buffer_sync_point_t * /*pSyncPointWaitList*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t * /*pSyncPoint*/,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp
  // sync mechanic can be ignored, because all lists are in-order
  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferCopy(
      hSrcMem, hDstMem, srcOffset, dstOffset, size, 0, nullptr, nullptr));

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferWriteExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, const void *pSrc,
    uint32_t /*numSyncPointsInWaitList*/,
    const ur_exp_command_buffer_sync_point_t * /*pSyncPointWaitList*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t * /*pSyncPoint*/,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp
  // sync mechanic can be ignored, because all lists are in-order
  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferWrite(hBuffer, false, offset, size,
                                                  pSrc, 0, nullptr, nullptr));

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferReadExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, void *pDst,
    uint32_t /*numSyncPointsInWaitList*/,
    const ur_exp_command_buffer_sync_point_t * /*pSyncPointWaitList*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t * /*pSyncPoint*/,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp
  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferRead(hBuffer, false, offset, size,
                                                 pDst, 0, nullptr, nullptr));

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferCopyRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t /*numSyncPointsInWaitList*/,
    const ur_exp_command_buffer_sync_point_t * /*pSyncPointWaitList*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t * /*pSyncPoint*/,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp
  // sync mechanic can be ignored, because all lists are in-order
  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferCopyRect(
      hSrcMem, hDstMem, srcOrigin, dstOrigin, region, srcRowPitch,
      srcSlicePitch, dstRowPitch, dstSlicePitch, 0, nullptr, nullptr));

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferWriteRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t /*numSyncPointsInWaitList*/,
    const ur_exp_command_buffer_sync_point_t * /*pSyncPointWaitList*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t * /*pSyncPoint*/,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp

  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferWriteRect(
      hBuffer, false, bufferOffset, hostOffset, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc, 0, nullptr,
      nullptr));

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferReadRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t /*numSyncPointsInWaitList*/,
    const ur_exp_command_buffer_sync_point_t * /*pSyncPointWaitList*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t * /*pSyncPoint*/,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp

  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferReadRect(
      hBuffer, false, bufferOffset, hostOffset, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst, 0, nullptr,
      nullptr));

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendUSMFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pMemory,
    const void *pPattern, size_t patternSize, size_t size,
    uint32_t /*numSyncPointsInWaitList*/,
    const ur_exp_command_buffer_sync_point_t * /*pSyncPointWaitList*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t * /*pSyncPoint*/,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  UR_CALL(commandListLocked->appendUSMFill(pMemory, patternSize, pPattern, size,
                                           0, nullptr, nullptr));
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendMemBufferFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    const void *pPattern, size_t patternSize, size_t offset, size_t size,
    uint32_t /*numSyncPointsInWaitList*/,
    const ur_exp_command_buffer_sync_point_t * /*pSyncPointWaitList*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t * /*pSyncPoint*/,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp
  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  UR_CALL(commandListLocked->appendMemBufferFill(
      hBuffer, pPattern, patternSize, offset, size, 0, nullptr, nullptr));
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendUSMPrefetchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, const void *pMemory,
    size_t size, ur_usm_migration_flags_t flags,
    uint32_t /*numSyncPointsInWaitList*/,
    const ur_exp_command_buffer_sync_point_t * /*pSyncPointWaitList*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t * /*pSyncPoint*/,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp

  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  UR_CALL(commandListLocked->appendUSMPrefetch(pMemory, size, flags, 0, nullptr,
                                               nullptr));

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urCommandBufferAppendUSMAdviseExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, const void *pMemory,
    size_t size, ur_usm_advice_flags_t advice,
    uint32_t /*numSyncPointsInWaitList*/,
    const ur_exp_command_buffer_sync_point_t * /*pSyncPointWaitList*/,
    uint32_t /*numEventsInWaitList*/,
    const ur_event_handle_t * /*phEventWaitList*/,
    ur_exp_command_buffer_sync_point_t * /*pSyncPoint*/,
    ur_event_handle_t * /*phEvent*/,
    ur_exp_command_buffer_command_handle_t * /*phCommand*/) try {
  // the same issue as in urCommandBufferAppendKernelLaunchExp

  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  UR_CALL(commandListLocked->appendUSMAdvise(pMemory, size, advice, nullptr));

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
    Descriptor.isInOrder = true;
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
  // sync mechanic can be ignored, because all lists are in-order
  (void)numSyncPointsInWaitList;
  (void)pSyncPointWaitList;
  (void)pSyncPoint;

  // Barrier on all commands before user defined commands.

  auto commandListLocked = hCommandBuffer->commandListManager.lock();
  UR_CALL(commandListLocked->appendBarrier(0, nullptr, nullptr));

  // Call user-defined function immediately
  pfnNativeCommand(pData);

  // Barrier on all commands after user defined commands.
  UR_CALL(commandListLocked->appendBarrier(0, nullptr, nullptr));

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
