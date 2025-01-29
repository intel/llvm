//===--------- command_list_manager.cpp - Level Zero Adapter --------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "command_list_manager.hpp"
#include "../helpers/kernel_helpers.hpp"
#include "../ur_interface_loader.hpp"
#include "context.hpp"
#include "kernel.hpp"

ur_command_list_manager::ur_command_list_manager(
    ur_context_handle_t context, ur_device_handle_t device,
    v2::raii::command_list_unique_handle &&commandList, v2::event_flags_t flags,
    ur_queue_handle_t queue)
    : context(context), device(device),
      eventPool(context->eventPoolCache.borrow(device->Id.value(), flags)),
      zeCommandList(std::move(commandList)), queue(queue) {
  UR_CALL_THROWS(ur::level_zero::urContextRetain(context));
  UR_CALL_THROWS(ur::level_zero::urDeviceRetain(device));
}

ur_command_list_manager::~ur_command_list_manager() {
  ur::level_zero::urContextRelease(context);
  ur::level_zero::urDeviceRelease(device);
}

std::pair<ze_event_handle_t *, uint32_t>
ur_command_list_manager::getWaitListView(const ur_event_handle_t *phWaitEvents,
                                         uint32_t numWaitEvents) {

  waitList.resize(numWaitEvents);
  for (uint32_t i = 0; i < numWaitEvents; i++) {
    waitList[i] = phWaitEvents[i]->getZeEvent();
  }

  return {waitList.data(), static_cast<uint32_t>(numWaitEvents)};
}

ze_event_handle_t
ur_command_list_manager::getSignalEvent(ur_event_handle_t *hUserEvent,
                                        ur_command_t commandType) {
  if (hUserEvent && queue) {
    *hUserEvent = eventPool->allocate();
    (*hUserEvent)->resetQueueAndCommand(queue, commandType);
    return (*hUserEvent)->getZeEvent();
  } else {
    return nullptr;
  }
}

ur_result_t ur_command_list_manager::appendKernelLaunch(
    ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY("ur_command_list_manager::appendKernelLaunch");

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hKernel->getProgramHandle(), UR_RESULT_ERROR_INVALID_NULL_POINTER);

  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  ze_kernel_handle_t hZeKernel = hKernel->getZeHandle(device);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(this->Mutex,
                                                          hKernel->Mutex);

  ze_group_count_t zeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3]{};
  UR_CALL(calculateKernelWorkDimensions(hZeKernel, device,
                                        zeThreadGroupDimensions, WG, workDim,
                                        pGlobalWorkSize, pLocalWorkSize));

  auto zeSignalEvent = getSignalEvent(phEvent, UR_COMMAND_KERNEL_LAUNCH);

  auto waitList = getWaitListView(phEventWaitList, numEventsInWaitList);

  bool memoryMigrated = false;
  auto memoryMigrate = [&](void *src, void *dst, size_t size) {
    ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                      (zeCommandList.get(), dst, src, size, nullptr,
                       waitList.second, waitList.first));
    memoryMigrated = true;
  };

  UR_CALL(hKernel->prepareForSubmission(context, device, pGlobalWorkOffset,
                                        workDim, WG[0], WG[1], WG[2],
                                        memoryMigrate));

  if (memoryMigrated) {
    // If memory was migrated, we don't need to pass the wait list to
    // the copy command again.
    waitList.first = nullptr;
    waitList.second = 0;
  }

  TRACK_SCOPE_LATENCY(
      "ur_command_list_manager::zeCommandListAppendLaunchKernel");
  ZE2UR_CALL(zeCommandListAppendLaunchKernel,
             (zeCommandList.get(), hZeKernel, &zeThreadGroupDimensions,
              zeSignalEvent, waitList.second, waitList.first));

  return UR_RESULT_SUCCESS;
}

ze_command_list_handle_t ur_command_list_manager::getZeCommandList() {
  return zeCommandList.get();
}
