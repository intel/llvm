//===--------- command_buffer.cpp - Level Zero Adapter --------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "command_buffer.hpp"
#include "command_buffer_command.hpp"
#include "helpers/kernel_helpers.hpp"
#include "helpers/mutable_helpers.hpp"
#include "logger/ur_logger.hpp"
#include "ur_api.h"
#include "ur_interface_loader.hpp"
#include "ur_level_zero.hpp"

/* L0 Command-buffer Extension Doc see:
https://github.com/intel/llvm/blob/sycl/sycl/doc/design/CommandGraph.md#level-zero
*/

// Print the name of a variable and its value in the L0 debug log
#define DEBUG_LOG(VAR) UR_LOG(DEBUG, #VAR " {}", VAR);

namespace {

ur_result_t
getMemoryAccessType(const ur_kernel_arg_mem_obj_properties_t *Properties,
                    ur_mem_handle_t_::access_mode_t &UrAccessMode) {
  UrAccessMode = ur_mem_handle_t_::read_write;
  if (Properties) {
    switch (Properties->memoryAccess) {
    case UR_MEM_FLAG_READ_WRITE:
      UrAccessMode = ur_mem_handle_t_::read_write;
      break;
    case UR_MEM_FLAG_WRITE_ONLY:
      UrAccessMode = ur_mem_handle_t_::write_only;
      break;
    case UR_MEM_FLAG_READ_ONLY:
      UrAccessMode = ur_mem_handle_t_::read_only;
      break;
    default:
      return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t getZeKernelWrapped(ur_kernel_handle_t Kernel,
                               ze_kernel_handle_t &ZeKernel,
                               ur_device_handle_t Device) {
  ze_kernel_handle_t Tmp{};
  UR_CALL(getZeKernel(Device->ZeDevice, Kernel, &Tmp));
  ZeKernel = Tmp;
  return UR_RESULT_SUCCESS;
}

ur_result_t getMemPtr(ur_mem_handle_t MemObj,
                      const ur_kernel_arg_mem_obj_properties_t *Properties,
                      char **&ZeHandlePtr, ur_device_handle_t Device,
                      device_ptr_storage_t *) {
  ur_mem_handle_t_::access_mode_t UrAccessMode;
  UR_CALL(getMemoryAccessType(Properties, UrAccessMode));

  if (MemObj) {
    UR_CALL(
        MemObj->getZeHandlePtr(ZeHandlePtr, UrAccessMode, Device, nullptr, 0u));
  }

  return UR_RESULT_SUCCESS;
}
// Checks whether zeCommandListImmediateAppendCommandListsExp can be used for a
// given Context and Device.
bool checkImmediateAppendSupport(ur_context_handle_t Context,
                                 ur_device_handle_t Device) {
  bool DriverSupportsImmediateAppend =
      Context->getPlatform()->ZeCommandListImmediateAppendExt.Supported;

  // If this environment variable is:
  //   - Set to 1: the immediate append path will always be enabled as long the
  //   pre-requisites are met.
  //   - Set to 0: the immediate append path will always be disabled.
  //   - Not Defined: The default behaviour will be used which enables the
  //   immediate append path only for some devices when the pre-requisites are
  //   met.
  const char *AppendEnvVarName = "UR_L0_CMD_BUFFER_USE_IMMEDIATE_APPEND_PATH";
  const char *UrRet = std::getenv(AppendEnvVarName);

  if (UrRet) {
    const bool EnableAppendPath = std::atoi(UrRet) == 1;

    if (EnableAppendPath && !Device->ImmCommandListUsed) {
      UR_LOG(ERR,
             "{} is set but immediate command-lists are currently "
             "disabled. Immediate command-lists are "
             "required to use the immediate append path.",
             AppendEnvVarName);
      std::abort();
    }
    if (EnableAppendPath && !DriverSupportsImmediateAppend) {
      UR_LOG(ERR,
             "{} is set but the current driver does not support the "
             "zeCommandListImmediateAppendCommandListsExp entrypoint.",
             AppendEnvVarName);
      std::abort();
    }

    return EnableAppendPath;
  }

  // Immediate Append path is temporarily disabled until related issues are
  // fixed.
  return false;
}
// Checks whether counter based events are supported for a given Device.
bool checkCounterBasedEventsSupport(ur_device_handle_t Device) {
  static const bool useDriverCounterBasedEvents = [] {
    const char *UrRet = std::getenv("UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS");
    if (!UrRet) {
      return true;
    }
    return std::atoi(UrRet) != 0;
  }();

  return Device->ImmCommandListUsed &&
         Device->Platform->allowDriverInOrderLists(
             true /*Only Allow Driver In Order List if requested*/) &&
         useDriverCounterBasedEvents &&
         Device->Platform->ZeDriverEventPoolCountingEventsExtensionFound;
}

// Gets a C pointer from a vector. If the vector is empty returns nullptr
// instead. This is different from the behaviour of the data() member function
// of the vector class which might not return nullptr when the vector is empty.
template <typename T> T *getPointerFromVector(std::vector<T> &V) {
  return V.size() == 0 ? nullptr : V.data();
}

/**
 * Default to using compute engine for fill operation, but allow to override
 * this with an environment variable. Disable the copy engine if the pattern
 * size is larger than the maximum supported.
 * @param[in] CommandBuffer The CommandBuffer where the fill command will be
 * appended.
 * @param[in] PatternSize The pattern size for the fill command.
 * @param[out] PreferCopyEngine Whether copy engine usage should be enabled or
 * disabled for fill commands.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t
preferCopyEngineForFill(ur_exp_command_buffer_handle_t CommandBuffer,
                        size_t PatternSize, bool &PreferCopyEngine) {
  assert(PatternSize > 0);

  PreferCopyEngine = false;
  if (!CommandBuffer->useCopyEngine()) {
    return UR_RESULT_SUCCESS;
  }

  // If the copy engine is available, and it supports this pattern size, the
  // command should be enqueued in the copy command list, otherwise enqueue it
  // in the compute command list.
  PreferCopyEngine =
      PatternSize <=
      CommandBuffer->Device
          ->QueueGroup[ur_device_handle_t_::queue_group_info_t::MainCopy]
          .ZeProperties.maxMemoryFillPatternSize;

  if (!PreferCopyEngine) {
    // Pattern size must fit the compute queue capabilities.
    UR_ASSERT(
        PatternSize <=
            CommandBuffer->Device
                ->QueueGroup[ur_device_handle_t_::queue_group_info_t::Compute]
                .ZeProperties.maxMemoryFillPatternSize,
        UR_RESULT_ERROR_INVALID_VALUE);
  }

  const char *UrRet = std::getenv("UR_L0_USE_COPY_ENGINE_FOR_FILL");
  const char *PiRet =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_FILL");

  PreferCopyEngine =
      PreferCopyEngine &&
      (UrRet ? std::stoi(UrRet) : (PiRet ? std::stoi(PiRet) : 0));

  return UR_RESULT_SUCCESS;
}

/**
 * Helper function for finding the Level Zero events associated with the
 * commands in a command-buffer, each event is pointed to by a sync-point in the
 * wait list.
 * @param[in] CommandBuffer to lookup the L0 events from.
 * @param[in] NumSyncPointsInWaitList Length of \p SyncPointWaitList.
 * @param[in] SyncPointWaitList List of sync points in \p CommandBuffer to find
 * the L0 events for.
 * @param[out] ZeEventList Return parameter for the L0 events associated with
 * each sync-point in \p SyncPointWaitList.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t getEventsFromSyncPoints(
    const ur_exp_command_buffer_handle_t &CommandBuffer,
    size_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    std::vector<ze_event_handle_t> &ZeEventList) {
  if (!SyncPointWaitList || NumSyncPointsInWaitList == 0)
    return UR_RESULT_SUCCESS;

  // For each sync-point add associated L0 event to the return list.
  for (size_t i = 0; i < NumSyncPointsInWaitList; i++) {
    if (auto EventHandle = CommandBuffer->SyncPoints.find(SyncPointWaitList[i]);
        EventHandle != CommandBuffer->SyncPoints.end()) {
      ZeEventList.push_back(EventHandle->second->ZeEvent);
    } else {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  }
  return UR_RESULT_SUCCESS;
}

/**
 * If necessary, it creates a signal event and appends it to the previous
 * command list (copy or compute), to indicate when it's finished executing.
 * @param[in] CommandBuffer The CommandBuffer where the command is appended.
 * @param[in] ZeCommandList the CommandList that's currently in use.
 * @param[out] WaitEventList The list of event for the future command list to
 * wait on before execution.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t createSyncPointBetweenCopyAndCompute(
    ur_exp_command_buffer_handle_t CommandBuffer,
    ze_command_list_handle_t ZeCommandList,
    std::vector<ze_event_handle_t> &WaitEventList) {

  if (!CommandBuffer->ZeCopyCommandList) {
    return UR_RESULT_SUCCESS;
  }

  bool IsCopy{ZeCommandList == CommandBuffer->ZeCopyCommandList};

  // Skip synchronization for the first node in a graph or if the current
  // command list matches the previous one.
  if (!CommandBuffer->MWasPrevCopyCommandList.has_value()) {
    CommandBuffer->MWasPrevCopyCommandList = IsCopy;
    return UR_RESULT_SUCCESS;
  } else if (IsCopy == CommandBuffer->MWasPrevCopyCommandList) {
    return UR_RESULT_SUCCESS;
  }

  /*
   * If the current CommandList differs from the previously used one, we must
   * append a signal event to the previous CommandList to track when
   * its execution is complete.
   */
  ur_event_handle_t SignalPrevCommandEvent = nullptr;
  UR_CALL(EventCreate(CommandBuffer->Context, nullptr /*Queue*/,
                      false /*IsMultiDevice*/, false, &SignalPrevCommandEvent,
                      false /*CounterBasedEventEnabled*/,
                      !CommandBuffer->IsProfilingEnabled,
                      false /*InterruptBasedEventEnabled*/));

  // Determine which command list to signal.
  auto CommandListToSignal = (!IsCopy && CommandBuffer->MWasPrevCopyCommandList)
                                 ? CommandBuffer->ZeCopyCommandList
                                 : CommandBuffer->ZeComputeCommandList;
  CommandBuffer->MWasPrevCopyCommandList = IsCopy;

  ZE2UR_CALL(zeCommandListAppendSignalEvent,
             (CommandListToSignal, SignalPrevCommandEvent->ZeEvent));

  // Add the event to the dependencies for future command list to wait on.
  WaitEventList.push_back(SignalPrevCommandEvent->ZeEvent);

  // Get sync point and register the event with it.
  ur_exp_command_buffer_sync_point_t SyncPoint =
      CommandBuffer->getNextSyncPoint();
  CommandBuffer->registerSyncPoint(SyncPoint, SignalPrevCommandEvent);

  return UR_RESULT_SUCCESS;
}

/**
 * If needed, creates a sync point for a given command and returns the L0
 * events associated with the sync point.
 * This operations is skipped if the command-buffer is in order.
 * @param[in] CommandType The type of the command.
 * @param[in] CommandBuffer The CommandBuffer where the command is appended.
 * @param[in] NumSyncPointsInWaitList Number of sync points that are
 * dependencies for the command.
 * @param[in] SyncPointWaitList List of sync point that are dependencies for the
 * command.
 * @param[in] HostVisible Whether the event associated with the sync point
 * should be host visible.
 * @param[out][optional] RetSyncPoint The new sync point.
 * @param[out] ZeEventList A list of L0 events that are dependencies for this
 * sync point.
 * @param[out] ZeLaunchEvent The L0 event associated with this sync point.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t createSyncPointAndGetZeEvents(
    ur_command_t CommandType, ur_exp_command_buffer_handle_t CommandBuffer,
    ze_command_list_handle_t ZeCommandList, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    bool HostVisible, ur_exp_command_buffer_sync_point_t *RetSyncPoint,
    std::vector<ze_event_handle_t> &ZeEventList,
    ze_event_handle_t &ZeLaunchEvent) {

  ZeLaunchEvent = nullptr;

  if (CommandBuffer->IsInOrderCmdList) {
    UR_CALL(createSyncPointBetweenCopyAndCompute(CommandBuffer, ZeCommandList,
                                                 ZeEventList));
    if (!ZeEventList.empty()) {
      NumSyncPointsInWaitList = ZeEventList.size();
    }
    return UR_RESULT_SUCCESS;
  }

  if (CommandBuffer->InOrderRequested && !CommandBuffer->ZeEventsList.empty()) {
    // If a user requested an in-order UR command-buffer, but driver L0
    // command-lists couldn't be used, then we need to emulate the behavior by
    // giving the command an event dependency on the last command.
    ze_event_handle_t LastEvent = CommandBuffer->ZeEventsList.back();
    ZeEventList.push_back(LastEvent);
  } else {
    UR_CALL(getEventsFromSyncPoints(CommandBuffer, NumSyncPointsInWaitList,
                                    SyncPointWaitList, ZeEventList));
  }
  ur_event_handle_t LaunchEvent;
  UR_CALL(EventCreate(CommandBuffer->Context, nullptr /*Queue*/,
                      false /*IsMultiDevice*/, HostVisible, &LaunchEvent,
                      false /*CounterBasedEventEnabled*/,
                      !CommandBuffer->IsProfilingEnabled,
                      false /*InterruptBasedEventEnabled*/));
  LaunchEvent->CommandType = CommandType;
  ZeLaunchEvent = LaunchEvent->ZeEvent;

  // Get sync point and register the event with it.
  ur_exp_command_buffer_sync_point_t SyncPoint =
      CommandBuffer->getNextSyncPoint();
  CommandBuffer->registerSyncPoint(SyncPoint, LaunchEvent);

  if (RetSyncPoint) {
    *RetSyncPoint = SyncPoint;
  }

  return UR_RESULT_SUCCESS;
}

// Shared by all memory read/write/copy UR interfaces.
// Helper function for common code when enqueuing memory operations to a
// command buffer.
ur_result_t enqueueCommandBufferMemCopyHelper(
    ur_command_t CommandType, ur_exp_command_buffer_handle_t CommandBuffer,
    void *Dst, const void *Src, size_t Size, bool PreferCopyEngine,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *RetSyncPoint) {

  ze_command_list_handle_t ZeCommandList =
      CommandBuffer->chooseCommandList(PreferCopyEngine);

  std::vector<ze_event_handle_t> ZeEventList;
  ze_event_handle_t ZeLaunchEvent = nullptr;
  UR_CALL(createSyncPointAndGetZeEvents(
      CommandType, CommandBuffer, ZeCommandList, NumSyncPointsInWaitList,
      SyncPointWaitList, false, RetSyncPoint, ZeEventList, ZeLaunchEvent));

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (ZeCommandList, Dst, Src, Size, ZeLaunchEvent, ZeEventList.size(),
              getPointerFromVector(ZeEventList)));

  return UR_RESULT_SUCCESS;
}

// Helper function for common code when enqueuing rectangular memory operations
// to a command-buffer.
ur_result_t enqueueCommandBufferMemCopyRectHelper(
    ur_command_t CommandType, ur_exp_command_buffer_handle_t CommandBuffer,
    void *Dst, const void *Src, ur_rect_offset_t SrcOrigin,
    ur_rect_offset_t DstOrigin, ur_rect_region_t Region, size_t SrcRowPitch,
    size_t DstRowPitch, size_t SrcSlicePitch, size_t DstSlicePitch,
    bool PreferCopyEngine, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *RetSyncPoint) {

  uint32_t SrcOriginX = ur_cast<uint32_t>(SrcOrigin.x);
  uint32_t SrcOriginY = ur_cast<uint32_t>(SrcOrigin.y);
  uint32_t SrcOriginZ = ur_cast<uint32_t>(SrcOrigin.z);

  uint32_t SrcPitch = SrcRowPitch;
  if (SrcPitch == 0)
    SrcPitch = ur_cast<uint32_t>(Region.width);

  if (SrcSlicePitch == 0)
    SrcSlicePitch = ur_cast<uint32_t>(Region.height) * SrcPitch;

  uint32_t DstOriginX = ur_cast<uint32_t>(DstOrigin.x);
  uint32_t DstOriginY = ur_cast<uint32_t>(DstOrigin.y);
  uint32_t DstOriginZ = ur_cast<uint32_t>(DstOrigin.z);

  uint32_t DstPitch = DstRowPitch;
  if (DstPitch == 0)
    DstPitch = ur_cast<uint32_t>(Region.width);

  if (DstSlicePitch == 0)
    DstSlicePitch = ur_cast<uint32_t>(Region.height) * DstPitch;

  uint32_t Width = ur_cast<uint32_t>(Region.width);
  uint32_t Height = ur_cast<uint32_t>(Region.height);
  uint32_t Depth = ur_cast<uint32_t>(Region.depth);

  const ze_copy_region_t ZeSrcRegion = {SrcOriginX, SrcOriginY, SrcOriginZ,
                                        Width,      Height,     Depth};
  const ze_copy_region_t ZeDstRegion = {DstOriginX, DstOriginY, DstOriginZ,
                                        Width,      Height,     Depth};

  ze_command_list_handle_t ZeCommandList =
      CommandBuffer->chooseCommandList(PreferCopyEngine);

  std::vector<ze_event_handle_t> ZeEventList;
  ze_event_handle_t ZeLaunchEvent = nullptr;
  UR_CALL(createSyncPointAndGetZeEvents(
      CommandType, CommandBuffer, ZeCommandList, NumSyncPointsInWaitList,
      SyncPointWaitList, false, RetSyncPoint, ZeEventList, ZeLaunchEvent));

  ZE2UR_CALL(zeCommandListAppendMemoryCopyRegion,
             (ZeCommandList, Dst, &ZeDstRegion, DstPitch, DstSlicePitch, Src,
              &ZeSrcRegion, SrcPitch, SrcSlicePitch, ZeLaunchEvent,
              ZeEventList.size(), getPointerFromVector(ZeEventList)));

  return UR_RESULT_SUCCESS;
}

// Helper function for enqueuing memory fills.
ur_result_t enqueueCommandBufferFillHelper(
    ur_command_t CommandType, ur_exp_command_buffer_handle_t CommandBuffer,
    void *Ptr, const void *Pattern, size_t PatternSize, size_t Size,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *RetSyncPoint) {
  // Pattern size must be a power of two.
  UR_ASSERT((PatternSize > 0) && ((PatternSize & (PatternSize - 1)) == 0),
            UR_RESULT_ERROR_INVALID_VALUE);

  bool PreferCopyEngine;
  UR_CALL(
      preferCopyEngineForFill(CommandBuffer, PatternSize, PreferCopyEngine));

  ze_command_list_handle_t ZeCommandList =
      CommandBuffer->chooseCommandList(PreferCopyEngine);

  std::vector<ze_event_handle_t> ZeEventList;
  ze_event_handle_t ZeLaunchEvent = nullptr;
  UR_CALL(createSyncPointAndGetZeEvents(
      CommandType, CommandBuffer, ZeCommandList, NumSyncPointsInWaitList,
      SyncPointWaitList, true, RetSyncPoint, ZeEventList, ZeLaunchEvent));

  ZE2UR_CALL(zeCommandListAppendMemoryFill,
             (ZeCommandList, Ptr, Pattern, PatternSize, Size, ZeLaunchEvent,
              ZeEventList.size(), getPointerFromVector(ZeEventList)));

  return UR_RESULT_SUCCESS;
}
} // namespace

ur_exp_command_buffer_handle_t_::ur_exp_command_buffer_handle_t_(
    ur_context_handle_t Context, ur_device_handle_t Device,
    ze_command_list_handle_t CommandList,
    ze_command_list_handle_t CommandListTranslated,
    ze_command_list_handle_t CommandListResetEvents,
    ze_command_list_handle_t CopyCommandList,
    ur_event_handle_t ExecutionFinishedEvent, ur_event_handle_t WaitEvent,
    ur_event_handle_t AllResetEvent, ur_event_handle_t CopyFinishedEvent,
    ur_event_handle_t ComputeFinishedEvent,
    const ur_exp_command_buffer_desc_t *Desc, const bool IsInOrderCmdList,
    const bool UseImmediateAppendPath)
    : Context(Context), Device(Device), ZeComputeCommandList(CommandList),
      ZeComputeCommandListTranslated(CommandListTranslated),
      ZeCommandListResetEvents(CommandListResetEvents),
      ZeCopyCommandList(CopyCommandList),
      ExecutionFinishedEvent(ExecutionFinishedEvent), WaitEvent(WaitEvent),
      AllResetEvent(AllResetEvent), CopyFinishedEvent(CopyFinishedEvent),
      ComputeFinishedEvent(ComputeFinishedEvent), ZeFencesMap(),
      ZeActiveFence(nullptr), SyncPoints(), NextSyncPoint(0),
      IsUpdatable(Desc->isUpdatable), IsProfilingEnabled(Desc->enableProfiling),
      InOrderRequested(Desc->isInOrder), IsInOrderCmdList(IsInOrderCmdList),
      UseImmediateAppendPath(UseImmediateAppendPath) {
  ur::level_zero::urContextRetain(Context);
  ur::level_zero::urDeviceRetain(Device);
}

void ur_exp_command_buffer_handle_t_::cleanupCommandBufferResources() {
  // Release the memory allocated to the Context stored in the command_buffer
  ur::level_zero::urContextRelease(Context);

  // Release the device
  ur::level_zero::urDeviceRelease(Device);

  // Release the memory allocated to the CommandList stored in the
  // command_buffer
  if (ZeComputeCommandList && checkL0LoaderTeardown()) {
    ZE_CALL_NOCHECK(zeCommandListDestroy, (ZeComputeCommandList));
  }
  if (useCopyEngine() && ZeCopyCommandList && checkL0LoaderTeardown()) {
    ZE_CALL_NOCHECK(zeCommandListDestroy, (ZeCopyCommandList));
  }

  // Release the memory allocated to the CommandListResetEvents stored in the
  // command_buffer
  if (ZeCommandListResetEvents && checkL0LoaderTeardown()) {
    ZE_CALL_NOCHECK(zeCommandListDestroy, (ZeCommandListResetEvents));
  }

  // Release additional events used by the command_buffer.
  if (ExecutionFinishedEvent) {
    CleanupCompletedEvent(ExecutionFinishedEvent, false /*QueueLocked*/,
                          false /*SetEventCompleted*/);
    urEventReleaseInternal(ExecutionFinishedEvent);
  }
  if (WaitEvent) {
    CleanupCompletedEvent(WaitEvent, false /*QueueLocked*/,
                          false /*SetEventCompleted*/);
    urEventReleaseInternal(WaitEvent);
  }
  if (AllResetEvent) {
    CleanupCompletedEvent(AllResetEvent, false /*QueueLocked*/,
                          false /*SetEventCompleted*/);
    urEventReleaseInternal(AllResetEvent);
  }

  if (CopyFinishedEvent) {
    CleanupCompletedEvent(CopyFinishedEvent, false /*QueueLocked*/,
                          false /*SetEventCompleted*/);
    urEventReleaseInternal(CopyFinishedEvent);
  }

  if (ComputeFinishedEvent) {
    CleanupCompletedEvent(ComputeFinishedEvent, false /*QueueLocked*/,
                          false /*SetEventCompleted*/);
    urEventReleaseInternal(ComputeFinishedEvent);
  }

  if (CurrentSubmissionEvent) {
    urEventReleaseInternal(CurrentSubmissionEvent);
  }

  // Release events added to the command_buffer
  for (auto &Sync : SyncPoints) {
    auto &Event = Sync.second;
    CleanupCompletedEvent(Event, false /*QueueLocked*/,
                          false /*SetEventCompleted*/);
    urEventReleaseInternal(Event);
  }

  // Release fences allocated to command-buffer
  for (auto &ZeFencePair : ZeFencesMap) {
    auto &ZeFence = ZeFencePair.second;
    if (checkL0LoaderTeardown()) {
      ZE_CALL_NOCHECK(zeFenceDestroy, (ZeFence));
    }
  }

  auto ReleaseIndirectMem = [](ur_kernel_handle_t Kernel) {
    if (IndirectAccessTrackingEnabled) {
      // urKernelRelease is called by CleanupCompletedEvent(Event) as soon as
      // kernel execution has finished. This is the place where we need to
      // release memory allocations. If kernel is not in use (not submitted by
      // some other thread) then release referenced memory allocations. As a
      // result, memory can be deallocated and context can be removed from
      // container in the platform. That's why we need to lock a mutex here.
      ur_platform_handle_t Platform = Kernel->Program->Context->getPlatform();
      std::scoped_lock<ur_shared_mutex> ContextsLock(Platform->ContextsMutex);

      if (--Kernel->SubmissionsCount == 0) {
        // Kernel is not submitted for execution, release referenced memory
        // allocations.
        for (auto &MemAlloc : Kernel->MemAllocs) {
          // std::pair<void *const, MemAllocRecord> *, Hash
          USMFreeHelper(MemAlloc->second.Context, MemAlloc->first,
                        MemAlloc->second.OwnNativeHandle);
        }
        Kernel->MemAllocs.clear();
      }
    }
  };

  for (auto &AssociatedKernel : KernelsList) {
    ReleaseIndirectMem(AssociatedKernel);
    ur::level_zero::urKernelRelease(AssociatedKernel);
  }
}

void ur_exp_command_buffer_handle_t_::registerSyncPoint(
    ur_exp_command_buffer_sync_point_t SyncPoint, ur_event_handle_t Event) {
  SyncPoints[SyncPoint] = Event;
  NextSyncPoint++;
  ZeEventsList.push_back(Event->ZeEvent);
}

ze_command_list_handle_t
ur_exp_command_buffer_handle_t_::chooseCommandList(bool PreferCopyEngine) {
  if (PreferCopyEngine && useCopyEngine()) {
    // We indicate that ZeCopyCommandList contains commands to be submitted.
    MCopyCommandListEmpty = false;
    return ZeCopyCommandList;
  }
  return ZeComputeCommandList;
}

ur_result_t ur_exp_command_buffer_handle_t_::getFenceForQueue(
    ze_command_queue_handle_t &ZeCommandQueue, ze_fence_handle_t &ZeFence) {
  // If we already have created a fence for this queue, first reset then reuse
  // it, otherwise create a new fence.
  auto ZeWorkloadFenceForQueue = this->ZeFencesMap.find(ZeCommandQueue);
  if (ZeWorkloadFenceForQueue == this->ZeFencesMap.end()) {
    ZeStruct<ze_fence_desc_t> ZeFenceDesc;
    ZE2UR_CALL(zeFenceCreate, (ZeCommandQueue, &ZeFenceDesc, &ZeFence));
    this->ZeFencesMap.insert({{ZeCommandQueue, ZeFence}});
  } else {
    ZeFence = ZeWorkloadFenceForQueue->second;
    ZE2UR_CALL(zeFenceReset, (ZeFence));
  }
  this->ZeActiveFence = ZeFence;
  return UR_RESULT_SUCCESS;
}

namespace ur::level_zero {

/**
 * Creates a L0 command list
 * @param[in] Context The Context associated with the command-list
 * @param[in] Device  The Device associated with the command-list
 * @param[in] IsInOrder Whether the command-list should be in-order.
 * @param[in] IsUpdatable Whether the command-list should be mutable.
 * @param[in] IsCopy Whether to use copy-engine for the the new command-list.
 * @param[out] CommandList The L0 command-list created by this function.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t createMainCommandList(ur_context_handle_t Context,
                                  ur_device_handle_t Device, bool IsInOrder,
                                  bool IsUpdatable, bool IsCopy,
                                  ze_command_list_handle_t &CommandList) {

  auto Type = IsCopy ? ur_device_handle_t_::queue_group_info_t::type::MainCopy
                     : ur_device_handle_t_::queue_group_info_t::type::Compute;
  uint32_t QueueGroupOrdinal = Device->QueueGroup[Type].ZeOrdinal;

  ZeStruct<ze_command_list_desc_t> ZeCommandListDesc;
  ZeCommandListDesc.commandQueueGroupOrdinal = QueueGroupOrdinal;

  // For non-linear graph, dependencies between commands are explicitly enforced
  // by sync points when enqueuing. Consequently, relax the command ordering in
  // the command list can enable the backend to further optimize the workload
  ZeCommandListDesc.flags = IsInOrder ? ZE_COMMAND_LIST_FLAG_IN_ORDER
                                      : ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING;

  DEBUG_LOG(ZeCommandListDesc.flags);

  ZeStruct<ze_mutable_command_list_exp_desc_t> ZeMutableCommandListDesc;
  if (IsUpdatable) {
    ZeMutableCommandListDesc.flags = 0;
    ZeCommandListDesc.pNext = &ZeMutableCommandListDesc;
  }

  ZE2UR_CALL(zeCommandListCreate, (Context->ZeContext, Device->ZeDevice,
                                   &ZeCommandListDesc, &CommandList));

  return UR_RESULT_SUCCESS;
}

/**
 * Checks whether the command-buffer can be constructed using in order
 * command-lists.
 * @param[in] Context The Context associated with the command-buffer.
 * @param[in] CommandBufferDesc The description of the command-buffer.
 * @return Returns true if in order command-lists can be enabled.
 */
bool canBeInOrder(ur_context_handle_t Context,
                  const ur_exp_command_buffer_desc_t *CommandBufferDesc) {
  bool CanUseDriverInOrderLists =
      Context->getPlatform()->allowDriverInOrderLists(
          true /*Only Allow Driver In Order List if requested*/);
  return CanUseDriverInOrderLists ? CommandBufferDesc->isInOrder : false;
}

/**
 * Append the initial barriers to the Compute and Copy command-lists.
 * @param CommandBuffer The CommandBuffer
 * @return UR_RESULT_SUCCESS or an error code on failure.
 */
ur_result_t appendExecutionWaits(ur_exp_command_buffer_handle_t CommandBuffer) {

  std::vector<ze_event_handle_t> PrecondEvents;
  if (CommandBuffer->ZeCommandListResetEvents) {
    PrecondEvents.push_back(CommandBuffer->AllResetEvent->ZeEvent);
  }
  if (!CommandBuffer->UseImmediateAppendPath) {
    PrecondEvents.push_back(CommandBuffer->WaitEvent->ZeEvent);
  }

  ZE2UR_CALL(zeCommandListAppendBarrier,
             (CommandBuffer->ZeComputeCommandList, nullptr,
              PrecondEvents.size(), PrecondEvents.data()));

  if (CommandBuffer->ZeCopyCommandList) {
    ZE2UR_CALL(zeCommandListAppendBarrier,
               (CommandBuffer->ZeCopyCommandList, nullptr, PrecondEvents.size(),
                PrecondEvents.data()));
  }

  return UR_RESULT_SUCCESS;
}

/**
 * Waits for any ongoing executions of the command-buffer to finish.
 * @param CommandBuffer The command-buffer to wait for.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t
waitForOngoingExecution(ur_exp_command_buffer_handle_t CommandBuffer) {

  if (ur_event_handle_t &CurrentSubmissionEvent =
          CommandBuffer->CurrentSubmissionEvent) {
    ZE2UR_CALL(zeEventHostSynchronize,
               (CurrentSubmissionEvent->ZeEvent, UINT64_MAX));
    UR_CALL(urEventReleaseInternal(CurrentSubmissionEvent));
    CurrentSubmissionEvent = nullptr;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t
urCommandBufferCreateExp(ur_context_handle_t Context, ur_device_handle_t Device,
                         const ur_exp_command_buffer_desc_t *CommandBufferDesc,
                         ur_exp_command_buffer_handle_t *CommandBuffer) {
  bool IsInOrder = canBeInOrder(Context, CommandBufferDesc);
  bool EnableProfiling = CommandBufferDesc->enableProfiling && !IsInOrder;
  bool IsUpdatable = CommandBufferDesc->isUpdatable;
  bool ImmediateAppendPath = checkImmediateAppendSupport(Context, Device);
  const bool WaitEventPath = !ImmediateAppendPath;
  bool UseCounterBasedEvents = checkCounterBasedEventsSupport(Device) &&
                               IsInOrder && ImmediateAppendPath;

  if (IsUpdatable) {
    UR_ASSERT(Context->getPlatform()->ZeMutableCmdListExt.Supported,
              UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
  }

  ze_command_list_handle_t ZeComputeCommandList = nullptr;
  ze_command_list_handle_t ZeCopyCommandList = nullptr;
  ze_command_list_handle_t ZeCommandListResetEvents = nullptr;
  ze_command_list_handle_t ZeComputeCommandListTranslated = nullptr;

  UR_CALL(createMainCommandList(Context, Device, IsInOrder, IsUpdatable, false,
                                ZeComputeCommandList));

  // Create a list for copy commands. Note that to simplify the implementation,
  // the current implementation only uses the main copy engine and does not use
  // the link engine even if available.
  //
  // Copy engine usage disabled for DG2, see CMPLRLLVM-68064
  if (Device->hasMainCopyEngine() && !Device->isDG2()) {
    UR_CALL(createMainCommandList(Context, Device, IsInOrder, false, true,
                                  ZeCopyCommandList));
  }

  ZE2UR_CALL(zelLoaderTranslateHandle,
             (ZEL_HANDLE_COMMAND_LIST, ZeComputeCommandList,
              (void **)&ZeComputeCommandListTranslated));

  // The CopyFinishedEvent and ComputeFinishedEvent are needed only when using
  // the ImmediateAppend Path.
  ur_event_handle_t CopyFinishedEvent = nullptr;
  ur_event_handle_t ComputeFinishedEvent = nullptr;
  if (ImmediateAppendPath) {
    if (Device->hasMainCopyEngine()) {
      UR_CALL(EventCreate(Context, nullptr /*Queue*/, false, false,
                          &CopyFinishedEvent, UseCounterBasedEvents,
                          !EnableProfiling,
                          false /*InterruptBasedEventEnabled*/));
    }

    if (EnableProfiling) {
      UR_CALL(EventCreate(Context, nullptr /*Queue*/, false /*IsMultiDevice*/,
                          false /*HostVisible*/, &ComputeFinishedEvent,
                          UseCounterBasedEvents, !EnableProfiling,
                          false /*InterruptBasedEventEnabled*/));
    }
  }

  // The WaitEvent is needed only when using WaitEvent Path.
  ur_event_handle_t WaitEvent = nullptr;
  if (WaitEventPath) {
    UR_CALL(EventCreate(Context, nullptr /*Queue*/, false /*IsMultiDevice*/,
                        false /*HostVisible*/, &WaitEvent,
                        false /*CounterBasedEventEnabled*/, !EnableProfiling,
                        false /*InterruptBasedEventEnabled*/));
  }

  // Create ZeCommandListResetEvents only if counter-based events are not being
  // used. Using counter-based events means that there is no need to reset any
  // events between executions. Counter-based events can only be enabled on the
  // ImmediateAppend Path.
  ur_event_handle_t AllResetEvent = nullptr;
  ur_event_handle_t ExecutionFinishedEvent = nullptr;
  if (!UseCounterBasedEvents) {
    UR_CALL(EventCreate(Context, nullptr /*Queue*/, false /*IsMultiDevice*/,
                        false /*HostVisible*/, &AllResetEvent,
                        false /*CounterBasedEventEnabled*/, !EnableProfiling,
                        false /*InterruptBasedEventEnabled*/));

    UR_CALL(createMainCommandList(Context, Device, false, false, false,
                                  ZeCommandListResetEvents));

    // The ExecutionFinishedEvent is only waited on by ZeCommandListResetEvents.
    UR_CALL(EventCreate(Context, nullptr /*Queue*/, false /*IsMultiDevice*/,
                        false /*HostVisible*/, &ExecutionFinishedEvent,
                        false /*CounterBasedEventEnabled*/, !EnableProfiling,
                        false /*InterruptBased*/));
  }

  try {
    *CommandBuffer = new ur_exp_command_buffer_handle_t_(
        Context, Device, ZeComputeCommandList, ZeComputeCommandListTranslated,
        ZeCommandListResetEvents, ZeCopyCommandList, ExecutionFinishedEvent,
        WaitEvent, AllResetEvent, CopyFinishedEvent, ComputeFinishedEvent,
        CommandBufferDesc, IsInOrder, ImmediateAppendPath);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  UR_CALL(appendExecutionWaits(*CommandBuffer));

  return UR_RESULT_SUCCESS;
}

ur_result_t
urCommandBufferRetainExp(ur_exp_command_buffer_handle_t CommandBuffer) {
  CommandBuffer->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t
urCommandBufferReleaseExp(ur_exp_command_buffer_handle_t CommandBuffer) {
  if (!CommandBuffer->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  waitForOngoingExecution(CommandBuffer);
  CommandBuffer->cleanupCommandBufferResources();

  delete CommandBuffer;
  return UR_RESULT_SUCCESS;
}

/* Finalizes the command-buffer so that it can later be enqueued using
 * enqueueImmediateAppendPath() which uses the
 * zeCommandListImmediateAppendCommandListsExp API. */
ur_result_t
finalizeImmediateAppendPath(ur_exp_command_buffer_handle_t CommandBuffer) {

  // Wait for the Copy Queue to finish at the end of the compute command list.
  if (!CommandBuffer->MCopyCommandListEmpty) {
    ZE2UR_CALL(zeCommandListAppendBarrier,
               (CommandBuffer->ZeCopyCommandList,
                CommandBuffer->CopyFinishedEvent->ZeEvent, 0, nullptr));

    ZE2UR_CALL(zeCommandListAppendBarrier,
               (CommandBuffer->ZeComputeCommandList, nullptr, 1,
                &CommandBuffer->CopyFinishedEvent->ZeEvent));
  }

  if (CommandBuffer->ZeCommandListResetEvents) {
    ZE2UR_CALL(zeCommandListAppendBarrier,
               (CommandBuffer->ZeCommandListResetEvents, nullptr, 1,
                &CommandBuffer->ExecutionFinishedEvent->ZeEvent));

    // Reset the L0 events we use for command-buffer sync-points to the
    // non-signaled state. This is required for multiple submissions.
    for (auto &Event : CommandBuffer->ZeEventsList) {
      ZE2UR_CALL(zeCommandListAppendEventReset,
                 (CommandBuffer->ZeCommandListResetEvents, Event));
    }

    if (!CommandBuffer->MCopyCommandListEmpty) {
      ZE2UR_CALL(zeCommandListAppendEventReset,
                 (CommandBuffer->ZeCommandListResetEvents,
                  CommandBuffer->CopyFinishedEvent->ZeEvent));
    }

    // Only the profiling command-list has a wait on the ExecutionFinishedEvent
    if (CommandBuffer->IsProfilingEnabled) {
      ZE2UR_CALL(zeCommandListAppendEventReset,
                 (CommandBuffer->ZeCommandListResetEvents,
                  CommandBuffer->ComputeFinishedEvent->ZeEvent));
    }

    ZE2UR_CALL(zeCommandListAppendEventReset,
               (CommandBuffer->ZeCommandListResetEvents,
                CommandBuffer->ExecutionFinishedEvent->ZeEvent));

    ZE2UR_CALL(zeCommandListAppendBarrier,
               (CommandBuffer->ZeCommandListResetEvents,
                CommandBuffer->AllResetEvent->ZeEvent, 0, nullptr));

    // Reset the all-reset-event for the UR command-buffer that is signaled
    // when all events of the main command-list have been reset.
    ZE2UR_CALL(zeCommandListAppendEventReset,
               (CommandBuffer->ZeComputeCommandList,
                CommandBuffer->AllResetEvent->ZeEvent));

    // All the events are reset by default. So signal the all reset event for
    // the first run of the command-buffer
    ZE2UR_CALL(zeEventHostSignal, (CommandBuffer->AllResetEvent->ZeEvent));
  }

  return UR_RESULT_SUCCESS;
}

/* Finalizes the command-buffer so that it can later be enqueued using
 * enqueueWaitEventPath() which uses the zeCommandQueueExecuteCommandLists API.
 */
ur_result_t
finalizeWaitEventPath(ur_exp_command_buffer_handle_t CommandBuffer) {

  ZE2UR_CALL(zeCommandListAppendEventReset,
             (CommandBuffer->ZeCommandListResetEvents,
              CommandBuffer->ExecutionFinishedEvent->ZeEvent));

  // Reset the L0 events we use for command-buffer sync-points to the
  // non-signaled state. This is required for multiple submissions.
  for (auto &Event : CommandBuffer->ZeEventsList) {
    ZE2UR_CALL(zeCommandListAppendEventReset,
               (CommandBuffer->ZeCommandListResetEvents, Event));
  }

  if (CommandBuffer->IsInOrderCmdList) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (CommandBuffer->ZeComputeCommandList,
                CommandBuffer->ExecutionFinishedEvent->ZeEvent));
  } else {
    // Wait for all the user added commands to complete, and signal the
    // command-buffer signal-event when they are done.
    ZE2UR_CALL(zeCommandListAppendBarrier,
               (CommandBuffer->ZeComputeCommandList,
                CommandBuffer->ExecutionFinishedEvent->ZeEvent,
                CommandBuffer->ZeEventsList.size(),
                CommandBuffer->ZeEventsList.data()));
  }

  ZE2UR_CALL(zeCommandListAppendSignalEvent,
             (CommandBuffer->ZeCommandListResetEvents,
              CommandBuffer->AllResetEvent->ZeEvent));

  return UR_RESULT_SUCCESS;
}

ur_result_t
urCommandBufferFinalizeExp(ur_exp_command_buffer_handle_t CommandBuffer) {
  UR_ASSERT(CommandBuffer, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(!CommandBuffer->IsFinalized, UR_RESULT_ERROR_INVALID_OPERATION);

  // It is not allowed to append to command list from multiple threads.
  std::scoped_lock<ur_shared_mutex> Guard(CommandBuffer->Mutex);

  if (CommandBuffer->UseImmediateAppendPath) {
    UR_CALL(finalizeImmediateAppendPath(CommandBuffer));
  } else {
    UR_CALL(finalizeWaitEventPath(CommandBuffer));
  }

  // Close the command lists and have them ready for dispatch.
  ZE2UR_CALL(zeCommandListClose, (CommandBuffer->ZeComputeCommandList));

  if (CommandBuffer->ZeCommandListResetEvents) {
    ZE2UR_CALL(zeCommandListClose, (CommandBuffer->ZeCommandListResetEvents));
  }

  if (CommandBuffer->useCopyEngine()) {
    ZE2UR_CALL(zeCommandListClose, (CommandBuffer->ZeCopyCommandList));
  }

  CommandBuffer->IsFinalized = true;

  return UR_RESULT_SUCCESS;
}

/**
 * Sets the kernel arguments for a kernel command that will be appended to the
 * command-buffer.
 * @param[in] Device The Device associated with the command-buffer where the
 * kernel command will be appended.
 * @param[in,out] Arguments stored in the ur_kernel_handle_t object to be set
 * on the /p ZeKernel object.
 * @param[in] ZeKernel The handle to the Level-Zero kernel that will be
 * appended.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t setKernelPendingArguments(
    ur_device_handle_t Device,
    std::vector<ur_kernel_handle_t_::ArgumentInfo> &PendingArguments,
    ze_kernel_handle_t ZeKernel) {
  // If there are any pending arguments set them now.
  for (auto &Arg : PendingArguments) {
    // The ArgValue may be a NULL pointer in which case a NULL value is used for
    // the kernel argument declared as a pointer to global or constant memory.
    char **ZeHandlePtr = nullptr;
    if (Arg.Value) {
      UR_CALL(Arg.Value->getZeHandlePtr(ZeHandlePtr, Arg.AccessMode, Device,
                                        nullptr, 0u));
    }
    ZE2UR_CALL(zeKernelSetArgumentValue,
               (ZeKernel, Arg.Index, Arg.Size, ZeHandlePtr));
  }
  PendingArguments.clear();

  return UR_RESULT_SUCCESS;
}

ur_result_t urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_kernel_handle_t Kernel,
    uint32_t WorkDim, const size_t *GlobalWorkOffset,
    const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
    uint32_t NumKernelAlternatives, ur_kernel_handle_t *KernelAlternatives,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    uint32_t /*NumEventsInWaitList*/,
    const ur_event_handle_t * /*EventWaitList*/,
    ur_exp_command_buffer_sync_point_t *RetSyncPoint,
    ur_event_handle_t * /*Event*/,
    ur_exp_command_buffer_command_handle_t *Command) {

  UR_ASSERT(Kernel->Program, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  // Command handles can only be obtained from updatable command-buffers
  UR_ASSERT(!(Command && !CommandBuffer->IsUpdatable),
            UR_RESULT_ERROR_INVALID_OPERATION);

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex, ur_shared_mutex> Lock(
      Kernel->Mutex, Kernel->Program->Mutex, CommandBuffer->Mutex);

  auto Device = CommandBuffer->Device;
  ze_kernel_handle_t ZeKernel{};
  UR_CALL(getZeKernel(Device->ZeDevice, Kernel, &ZeKernel));

  if (GlobalWorkOffset != NULL) {
    UR_CALL(setKernelGlobalOffset(CommandBuffer->Context, ZeKernel, WorkDim,
                                  GlobalWorkOffset));
  }

  // If there are any pending arguments set them now.
  if (!Kernel->PendingArguments.empty()) {
    UR_CALL(
        setKernelPendingArguments(Device, Kernel->PendingArguments, ZeKernel));
  }

  ze_group_count_t ZeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3];
  UR_CALL(calculateKernelWorkDimensions(ZeKernel, Device,
                                        ZeThreadGroupDimensions, WG, WorkDim,
                                        GlobalWorkSize, LocalWorkSize));

  ZE2UR_CALL(zeKernelSetGroupSize, (ZeKernel, WG[0], WG[1], WG[2]));

  CommandBuffer->KernelsList.push_back(Kernel);
  for (size_t i = 0; i < NumKernelAlternatives; i++) {
    CommandBuffer->KernelsList.push_back(KernelAlternatives[i]);
  }

  ur::level_zero::urKernelRetain(Kernel);
  // Retain alternative kernels if provided
  for (size_t i = 0; i < NumKernelAlternatives; i++) {
    ur::level_zero::urKernelRetain(KernelAlternatives[i]);
  }

  if (Command) {
    assert(CommandBuffer->IsUpdatable);
    auto Platform = CommandBuffer->Context->getPlatform();
    ze_command_list_handle_t ZeCommandList =
        CommandBuffer->ZeComputeCommandListTranslated;
    if (Platform->ZeMutableCmdListExt.LoaderExtension) {
      ZeCommandList = CommandBuffer->ZeComputeCommandList;
    }

    std::unique_ptr<kernel_command_handle> NewCommand;
    UR_CALL(createCommandHandleUnlocked(
        CommandBuffer, ZeCommandList, Kernel, WorkDim, GlobalWorkSize,
        NumKernelAlternatives, KernelAlternatives, Platform, getZeKernelWrapped,
        Device, NewCommand));
    *Command = NewCommand.get();
    CommandBuffer->CommandHandles.push_back(std::move(NewCommand));
  }
  std::vector<ze_event_handle_t> ZeEventList;
  ze_event_handle_t ZeLaunchEvent = nullptr;
  UR_CALL(createSyncPointAndGetZeEvents(
      UR_COMMAND_KERNEL_LAUNCH, CommandBuffer,
      CommandBuffer->ZeComputeCommandList, NumSyncPointsInWaitList,
      SyncPointWaitList, false, RetSyncPoint, ZeEventList, ZeLaunchEvent));

  ZE2UR_CALL(zeCommandListAppendLaunchKernel,
             (CommandBuffer->ZeComputeCommandList, ZeKernel,
              &ZeThreadGroupDimensions, ZeLaunchEvent, ZeEventList.size(),
              getPointerFromVector(ZeEventList)));

  return UR_RESULT_SUCCESS;
}

ur_result_t urCommandBufferAppendUSMMemcpyExp(
    ur_exp_command_buffer_handle_t CommandBuffer, void *Dst, const void *Src,
    size_t Size, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    uint32_t /*NumEventsInWaitList*/,
    const ur_event_handle_t * /*EventWaitList*/,
    ur_exp_command_buffer_sync_point_t *SyncPoint,
    ur_event_handle_t * /*Event*/,
    ur_exp_command_buffer_command_handle_t * /*Command*/) {

  bool PreferCopyEngine = !IsDevicePointer(CommandBuffer->Context, Src) ||
                          !IsDevicePointer(CommandBuffer->Context, Dst);
  // For better performance, Copy Engines are not preferred given Shared
  // pointers on DG2.
  if (CommandBuffer->Device->isDG2() &&
      (IsSharedPointer(CommandBuffer->Context, Src) ||
       IsSharedPointer(CommandBuffer->Context, Dst))) {
    PreferCopyEngine = false;
  }
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueCommandBufferMemCopyHelper(
      UR_COMMAND_USM_MEMCPY, CommandBuffer, Dst, Src, Size, PreferCopyEngine,
      NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

ur_result_t urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t SrcMem,
    ur_mem_handle_t DstMem, size_t SrcOffset, size_t DstOffset, size_t Size,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    uint32_t /*NumEventsInWaitList*/,
    const ur_event_handle_t * /*EventWaitList*/,
    ur_exp_command_buffer_sync_point_t *SyncPoint,
    ur_event_handle_t * /*Event*/,
    ur_exp_command_buffer_command_handle_t * /*Command*/) {
  auto SrcBuffer = ur_cast<ur_buffer *>(SrcMem);
  auto DstBuffer = ur_cast<ur_buffer *>(DstMem);

  std::shared_lock<ur_shared_mutex> SrcLock(SrcBuffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex> LockAll(
      SrcLock, DstBuffer->Mutex);

  char *ZeHandleSrc;
  UR_CALL(SrcBuffer->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                                 CommandBuffer->Device, nullptr, 0u));
  char *ZeHandleDst;
  UR_CALL(DstBuffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                 CommandBuffer->Device, nullptr, 0u));

  bool PreferCopyEngine = (SrcBuffer->OnHost || DstBuffer->OnHost);

  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueCommandBufferMemCopyHelper(
      UR_COMMAND_MEM_BUFFER_COPY, CommandBuffer, ZeHandleDst + DstOffset,
      ZeHandleSrc + SrcOffset, Size, PreferCopyEngine, NumSyncPointsInWaitList,
      SyncPointWaitList, SyncPoint);
}

ur_result_t urCommandBufferAppendMemBufferCopyRectExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t SrcMem,
    ur_mem_handle_t DstMem, ur_rect_offset_t SrcOrigin,
    ur_rect_offset_t DstOrigin, ur_rect_region_t Region, size_t SrcRowPitch,
    size_t SrcSlicePitch, size_t DstRowPitch, size_t DstSlicePitch,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    uint32_t /*NumEventsInWaitList*/,
    const ur_event_handle_t * /*EventWaitList*/,
    ur_exp_command_buffer_sync_point_t *SyncPoint,
    ur_event_handle_t * /*Event*/,
    ur_exp_command_buffer_command_handle_t * /*Command*/) {
  auto SrcBuffer = ur_cast<ur_buffer *>(SrcMem);
  auto DstBuffer = ur_cast<ur_buffer *>(DstMem);

  std::shared_lock<ur_shared_mutex> SrcLock(SrcBuffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex> LockAll(
      SrcLock, DstBuffer->Mutex);

  char *ZeHandleSrc;
  UR_CALL(SrcBuffer->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                                 CommandBuffer->Device, nullptr, 0u));
  char *ZeHandleDst;
  UR_CALL(DstBuffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                 CommandBuffer->Device, nullptr, 0u));

  bool PreferCopyEngine = (SrcBuffer->OnHost || DstBuffer->OnHost);

  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueCommandBufferMemCopyRectHelper(
      UR_COMMAND_MEM_BUFFER_COPY_RECT, CommandBuffer, ZeHandleDst, ZeHandleSrc,
      SrcOrigin, DstOrigin, Region, SrcRowPitch, DstRowPitch, SrcSlicePitch,
      DstSlicePitch, PreferCopyEngine, NumSyncPointsInWaitList,
      SyncPointWaitList, SyncPoint);
}

ur_result_t urCommandBufferAppendMemBufferWriteExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t Buffer,
    size_t Offset, size_t Size, const void *Src,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    uint32_t /*NumEventsInWaitList*/,
    const ur_event_handle_t * /*EventWaitList*/,
    ur_exp_command_buffer_sync_point_t *SyncPoint,
    ur_event_handle_t * /*Event*/,
    ur_exp_command_buffer_command_handle_t * /*Command*/) {
  std::scoped_lock<ur_shared_mutex> Lock(Buffer->Mutex);

  char *ZeHandleDst = nullptr;
  UR_CALL(Buffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                              CommandBuffer->Device, nullptr, 0u));
  // Always prefer copy engine for writes
  bool PreferCopyEngine = true;

  return enqueueCommandBufferMemCopyHelper(
      UR_COMMAND_MEM_BUFFER_WRITE, CommandBuffer,
      ZeHandleDst + Offset, // dst
      Src,                  // src
      Size, PreferCopyEngine, NumSyncPointsInWaitList, SyncPointWaitList,
      SyncPoint);
}

ur_result_t urCommandBufferAppendMemBufferWriteRectExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t Buffer,
    ur_rect_offset_t BufferOffset, ur_rect_offset_t HostOffset,
    ur_rect_region_t Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, void *Src,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    uint32_t /*NumEventsInWaitList*/,
    const ur_event_handle_t * /*EventWaitList*/,
    ur_exp_command_buffer_sync_point_t *SyncPoint,
    ur_event_handle_t * /*Event*/,
    ur_exp_command_buffer_command_handle_t * /*Command*/) {
  std::scoped_lock<ur_shared_mutex> Lock(Buffer->Mutex);

  char *ZeHandleDst = nullptr;
  UR_CALL(Buffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                              CommandBuffer->Device, nullptr, 0u));

  // Always prefer copy engine for writes
  bool PreferCopyEngine = true;

  return enqueueCommandBufferMemCopyRectHelper(
      UR_COMMAND_MEM_BUFFER_WRITE_RECT, CommandBuffer, ZeHandleDst,
      const_cast<char *>(static_cast<const char *>(Src)), HostOffset,
      BufferOffset, Region, HostRowPitch, BufferRowPitch, HostSlicePitch,
      BufferSlicePitch, PreferCopyEngine, NumSyncPointsInWaitList,
      SyncPointWaitList, SyncPoint);
}

ur_result_t urCommandBufferAppendMemBufferReadExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t Buffer,
    size_t Offset, size_t Size, void *Dst, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    uint32_t /*NumEventsInWaitList*/,
    const ur_event_handle_t * /*EventWaitList*/,
    ur_exp_command_buffer_sync_point_t *SyncPoint,
    ur_event_handle_t * /*Event*/,
    ur_exp_command_buffer_command_handle_t * /*Command*/) {
  std::scoped_lock<ur_shared_mutex> SrcLock(Buffer->Mutex);

  char *ZeHandleSrc = nullptr;
  UR_CALL(Buffer->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                              CommandBuffer->Device, nullptr, 0u));

  // Always prefer copy engine for reads
  bool PreferCopyEngine = true;

  return enqueueCommandBufferMemCopyHelper(
      UR_COMMAND_MEM_BUFFER_READ, CommandBuffer, Dst, ZeHandleSrc + Offset,
      Size, PreferCopyEngine, NumSyncPointsInWaitList, SyncPointWaitList,
      SyncPoint);
}

ur_result_t urCommandBufferAppendMemBufferReadRectExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t Buffer,
    ur_rect_offset_t BufferOffset, ur_rect_offset_t HostOffset,
    ur_rect_region_t Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, void *Dst,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    uint32_t /*NumEventsInWaitList*/,
    const ur_event_handle_t * /*EventWaitList*/,
    ur_exp_command_buffer_sync_point_t *SyncPoint,
    ur_event_handle_t * /*Event*/,
    ur_exp_command_buffer_command_handle_t * /*Command*/) {
  std::scoped_lock<ur_shared_mutex> SrcLock(Buffer->Mutex);

  char *ZeHandleSrc;
  UR_CALL(Buffer->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                              CommandBuffer->Device, nullptr, 0u));

  // Always prefer copy engine for reads
  bool PreferCopyEngine = true;

  return enqueueCommandBufferMemCopyRectHelper(
      UR_COMMAND_MEM_BUFFER_READ_RECT, CommandBuffer, Dst, ZeHandleSrc,
      BufferOffset, HostOffset, Region, BufferRowPitch, HostRowPitch,
      BufferSlicePitch, HostSlicePitch, PreferCopyEngine,
      NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

ur_result_t urCommandBufferAppendUSMPrefetchExp(
    ur_exp_command_buffer_handle_t CommandBuffer, const void *Mem, size_t Size,
    ur_usm_migration_flags_t /*Flags*/, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    uint32_t /*NumEventsInWaitList*/,
    const ur_event_handle_t * /*EventWaitList*/,
    ur_exp_command_buffer_sync_point_t *RetSyncPoint,
    ur_event_handle_t * /*Event*/,
    ur_exp_command_buffer_command_handle_t * /*Command*/) {

  std::vector<ze_event_handle_t> ZeEventList;
  ze_event_handle_t ZeLaunchEvent = nullptr;
  UR_CALL(createSyncPointAndGetZeEvents(
      UR_COMMAND_USM_PREFETCH, CommandBuffer,
      CommandBuffer->ZeComputeCommandList, NumSyncPointsInWaitList,
      SyncPointWaitList, true, RetSyncPoint, ZeEventList, ZeLaunchEvent));

  if (NumSyncPointsInWaitList) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (CommandBuffer->ZeComputeCommandList, NumSyncPointsInWaitList,
                ZeEventList.data()));
  }

  // Add the prefetch command to the command-buffer.
  // Note that L0 does not handle migration flags.
  ZE2UR_CALL(zeCommandListAppendMemoryPrefetch,
             (CommandBuffer->ZeComputeCommandList, Mem, Size));

  if (!CommandBuffer->IsInOrderCmdList) {
    // Level Zero does not have a completion "event" with the prefetch API,
    // so manually add command to signal our event.
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (CommandBuffer->ZeComputeCommandList, ZeLaunchEvent));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urCommandBufferAppendUSMAdviseExp(
    ur_exp_command_buffer_handle_t CommandBuffer, const void *Mem, size_t Size,
    ur_usm_advice_flags_t Advice, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    uint32_t /*NumEventsInWaitList*/,
    const ur_event_handle_t * /*EventWaitList*/,
    ur_exp_command_buffer_sync_point_t *RetSyncPoint,
    ur_event_handle_t * /*Event*/,
    ur_exp_command_buffer_command_handle_t * /*Command*/) {
  // A memory chunk can be advised with muliple memory advices
  // We therefore prefer if statements to switch cases to combine all potential
  // flags
  uint32_t Value = 0;
  if (Advice & UR_USM_ADVICE_FLAG_SET_READ_MOSTLY)
    Value |= static_cast<int>(ZE_MEMORY_ADVICE_SET_READ_MOSTLY);
  if (Advice & UR_USM_ADVICE_FLAG_CLEAR_READ_MOSTLY)
    Value |= static_cast<int>(ZE_MEMORY_ADVICE_CLEAR_READ_MOSTLY);
  if (Advice & UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION)
    Value |= static_cast<int>(ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION);
  if (Advice & UR_USM_ADVICE_FLAG_CLEAR_PREFERRED_LOCATION)
    Value |= static_cast<int>(ZE_MEMORY_ADVICE_CLEAR_PREFERRED_LOCATION);
  if (Advice & UR_USM_ADVICE_FLAG_SET_NON_ATOMIC_MOSTLY)
    Value |= static_cast<int>(ZE_MEMORY_ADVICE_SET_NON_ATOMIC_MOSTLY);
  if (Advice & UR_USM_ADVICE_FLAG_CLEAR_NON_ATOMIC_MOSTLY)
    Value |= static_cast<int>(ZE_MEMORY_ADVICE_CLEAR_NON_ATOMIC_MOSTLY);
  if (Advice & UR_USM_ADVICE_FLAG_BIAS_CACHED)
    Value |= static_cast<int>(ZE_MEMORY_ADVICE_BIAS_CACHED);
  if (Advice & UR_USM_ADVICE_FLAG_BIAS_UNCACHED)
    Value |= static_cast<int>(ZE_MEMORY_ADVICE_BIAS_UNCACHED);
  if (Advice & UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION_HOST)
    Value |= static_cast<int>(ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION);
  if (Advice & UR_USM_ADVICE_FLAG_CLEAR_PREFERRED_LOCATION_HOST)
    Value |= static_cast<int>(ZE_MEMORY_ADVICE_CLEAR_PREFERRED_LOCATION);

  ze_memory_advice_t ZeAdvice = static_cast<ze_memory_advice_t>(Value);

  std::vector<ze_event_handle_t> ZeEventList;
  ze_event_handle_t ZeLaunchEvent = nullptr;
  UR_CALL(createSyncPointAndGetZeEvents(
      UR_COMMAND_USM_ADVISE, CommandBuffer, CommandBuffer->ZeComputeCommandList,
      NumSyncPointsInWaitList, SyncPointWaitList, true, RetSyncPoint,
      ZeEventList, ZeLaunchEvent));

  if (NumSyncPointsInWaitList) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (CommandBuffer->ZeComputeCommandList, NumSyncPointsInWaitList,
                ZeEventList.data()));
  }

  ZE2UR_CALL(zeCommandListAppendMemAdvise,
             (CommandBuffer->ZeComputeCommandList,
              CommandBuffer->Device->ZeDevice, Mem, Size, ZeAdvice));

  if (!CommandBuffer->IsInOrderCmdList) {
    // Level Zero does not have a completion "event" with the advise API,
    // so manually add command to signal our event.
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (CommandBuffer->ZeComputeCommandList, ZeLaunchEvent));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urCommandBufferAppendMemBufferFillExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t Buffer,
    const void *Pattern, size_t PatternSize, size_t Offset, size_t Size,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    uint32_t /*NumEventsInWaitList*/,
    const ur_event_handle_t * /*EventWaitList*/,
    ur_exp_command_buffer_sync_point_t *SyncPoint,
    ur_event_handle_t * /*Event*/,
    ur_exp_command_buffer_command_handle_t * /*Command*/) {

  std::scoped_lock<ur_shared_mutex> Lock(Buffer->Mutex);

  char *ZeHandleDst = nullptr;
  ur_buffer *UrBuffer = reinterpret_cast<ur_buffer *>(Buffer);
  UR_CALL(UrBuffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                CommandBuffer->Device, nullptr, 0u));

  return enqueueCommandBufferFillHelper(
      UR_COMMAND_MEM_BUFFER_FILL, CommandBuffer, ZeHandleDst + Offset,
      Pattern,     // It will be interpreted as an 8-bit value,
      PatternSize, // which is indicated with this pattern_size==1
      Size, NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

ur_result_t urCommandBufferAppendUSMFillExp(
    ur_exp_command_buffer_handle_t CommandBuffer, void *Ptr,
    const void *Pattern, size_t PatternSize, size_t Size,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    uint32_t /*NumEventsInWaitList*/,
    const ur_event_handle_t * /*EventWaitList*/,
    ur_exp_command_buffer_sync_point_t *SyncPoint,
    ur_event_handle_t * /*Event*/,
    ur_exp_command_buffer_command_handle_t * /*Command*/) {

  return enqueueCommandBufferFillHelper(
      UR_COMMAND_MEM_BUFFER_FILL, CommandBuffer, Ptr,
      Pattern,     // It will be interpreted as an 8-bit value,
      PatternSize, // which is indicated with this pattern_size==1
      Size, NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

/**
 * Gets an L0 command queue that supports the chosen engine.
 * @param[in] Queue The UR queue used to submit the command-buffer.
 * @param[in] UseCopyEngine Which engine to use. true for the copy engine and
 * false for the compute engine.
 * @param[out] ZeCommandQueue The L0 command queue.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t getZeCommandQueue(ur_queue_handle_t Queue, bool UseCopyEngine,
                              ze_command_queue_handle_t &ZeCommandQueue) {
  auto &QGroup = Queue->getQueueGroup(UseCopyEngine);
  uint32_t QueueGroupOrdinal;
  ZeCommandQueue = QGroup.getZeQueue(&QueueGroupOrdinal);
  return UR_RESULT_SUCCESS;
}

/**
 * Waits for the all the dependencies of the command-buffer
 * @param[in] CommandBuffer The command-buffer.
 * @param[in] Queue The UR queue used to submit the command-buffer.
 * @param[in] NumEventsInWaitList The number of events to wait for.
 * @param[in] EventWaitList List of events to wait for.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t waitForDependencies(ur_exp_command_buffer_handle_t CommandBuffer,
                                ur_queue_handle_t Queue,
                                uint32_t NumEventsInWaitList,
                                const ur_event_handle_t *EventWaitList) {
  std::scoped_lock<ur_shared_mutex> Guard(CommandBuffer->Mutex);
  const bool UseCopyEngine = false;
  bool MustSignalWaitEvent = true;

  // Level-zero does not allow in-order queue when immediate command-lists are
  // not used. For that reason, if the UR queue is in-order, we need to emulate,
  // its in-order properties by adding an event dependency on the last command
  // executed by the queue.
  std::vector<ur_event_handle_t> WaitList;
  if (Queue->isInOrderQueue() && Queue->LastCommandEvent) {
    WaitList.reserve(NumEventsInWaitList + 1);

    if (NumEventsInWaitList) {
      WaitList.insert(WaitList.end(), EventWaitList,
                      EventWaitList + NumEventsInWaitList);
    }
    WaitList.push_back(Queue->LastCommandEvent);

    ++NumEventsInWaitList;
    EventWaitList = WaitList.data();
  }

  if (NumEventsInWaitList) {
    ur_ze_event_list_t TmpWaitList;
    UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
        NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

    // Update the WaitList of the Wait Event
    // Events are appended to the WaitList if the WaitList is not empty
    if (CommandBuffer->WaitEvent->WaitList.isEmpty())
      CommandBuffer->WaitEvent->WaitList = TmpWaitList;
    else
      CommandBuffer->WaitEvent->WaitList.insert(TmpWaitList);

    if (!CommandBuffer->WaitEvent->WaitList.isEmpty()) {
      // Create command-list to execute before `CommandListPtr` and will signal
      // when `EventWaitList` dependencies are complete.
      ur_command_list_ptr_t WaitCommandList{};
      UR_CALL(Queue->Context->getAvailableCommandList(
          Queue, WaitCommandList, false /*UseCopyEngine*/, NumEventsInWaitList,
          EventWaitList, false /*AllowBatching*/, nullptr /*ForcedCmdQueue*/));

      ZE2UR_CALL(zeCommandListAppendBarrier,
                 (WaitCommandList->first, CommandBuffer->WaitEvent->ZeEvent,
                  CommandBuffer->WaitEvent->WaitList.Length,
                  CommandBuffer->WaitEvent->WaitList.ZeEventList));
      Queue->executeCommandList(WaitCommandList, false /*IsBlocking*/,
                                false /*OKToBatchCommand*/);
      MustSignalWaitEvent = false;
    }
  }
  // Given WaitEvent was created without specifying Counting Events, then this
  // event can be signalled on the host.
  if (MustSignalWaitEvent) {
    ZE2UR_CALL(zeEventHostSignal, (CommandBuffer->WaitEvent->ZeEvent));
  }
  return UR_RESULT_SUCCESS;
}

/**
 * Appends a QueryKernelTimestamps command that does profiling for all the
 * sync-point events.
 * @param CommandBuffer The command-buffer that is being enqueued.
 * @param CommandList The command-list to append the QueryKernelTimestamps
 * command to.
 * @param SignalEvent The event that must be signaled after the profiling is
 * finished.
 * @param WaitEvent The event that must be waited on before starting the
 * profiling.
 * @param ProfilingEvent The event that will contain the profiling data.
 * @return UR_RESULT_SUCCESS or an error code on failure.
 */
ur_result_t appendProfilingQueries(ur_exp_command_buffer_handle_t CommandBuffer,
                                   ze_command_list_handle_t CommandList,
                                   ur_event_handle_t SignalEvent,
                                   ur_event_handle_t WaitEvent,
                                   ur_event_handle_t ProfilingEvent) {
  // Multiple submissions of a command-buffer implies that we need to save
  // the event timestamps before resubmiting the command-buffer. We
  // therefore copy these timestamps in a dedicated USM memory section
  // before completing the command-buffer execution, and then attach this
  // memory to the event returned to users to allow the profiling
  // engine to recover these timestamps.
  command_buffer_profiling_t *Profiling = new command_buffer_profiling_t();

  Profiling->NumEvents = CommandBuffer->ZeEventsList.size();
  Profiling->Timestamps =
      new ze_kernel_timestamp_result_t[Profiling->NumEvents];

  uint32_t NumWaitEvents = WaitEvent ? 1 : 0;
  ze_event_handle_t *ZeWaitEventList =
      WaitEvent ? &(WaitEvent->ZeEvent) : nullptr;
  ze_event_handle_t ZeSignalEvent =
      SignalEvent ? SignalEvent->ZeEvent : nullptr;
  ZE2UR_CALL(zeCommandListAppendQueryKernelTimestamps,
             (CommandList, CommandBuffer->ZeEventsList.size(),
              CommandBuffer->ZeEventsList.data(), (void *)Profiling->Timestamps,
              0, ZeSignalEvent, NumWaitEvents, ZeWaitEventList));

  ProfilingEvent->CommandData = static_cast<void *>(Profiling);

  return UR_RESULT_SUCCESS;
}

/* Enqueues the command-buffer using the
 * zeCommandListImmediateAppendCommandListsExp API. */
ur_result_t enqueueImmediateAppendPath(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_queue_handle_t Queue,
    uint32_t NumEventsInWaitList, const ur_event_handle_t *EventWaitList,
    ur_event_handle_t *Event, ur_command_list_ptr_t CommandListHelper,
    bool DoProfiling) {

  ur_platform_handle_t Platform = CommandBuffer->Context->getPlatform();

  assert(CommandListHelper->second.IsImmediate);
  assert(Platform->ZeCommandListImmediateAppendExt.Supported);

  ur_ze_event_list_t UrZeEventList;
  if (NumEventsInWaitList) {
    UR_CALL(UrZeEventList.createAndRetainUrZeEventList(
        NumEventsInWaitList, EventWaitList, Queue, false));
  }
  (*Event)->WaitList = UrZeEventList;
  const auto &WaitList = (*Event)->WaitList;

  if (!CommandBuffer->MCopyCommandListEmpty) {
    ur_command_list_ptr_t ZeCopyEngineImmediateListHelper{};
    UR_CALL(Queue->Context->getAvailableCommandList(
        Queue, ZeCopyEngineImmediateListHelper, true /*UseCopyEngine*/,
        NumEventsInWaitList, EventWaitList, false /*AllowBatching*/,
        nullptr /*ForcedCmdQueue*/));
    assert(ZeCopyEngineImmediateListHelper->second.IsImmediate);

    ZE2UR_CALL(Platform->ZeCommandListImmediateAppendExt
                   .zeCommandListImmediateAppendCommandListsExp,
               (ZeCopyEngineImmediateListHelper->first, 1,
                &CommandBuffer->ZeCopyCommandList, nullptr,
                UrZeEventList.Length, UrZeEventList.ZeEventList));

    UR_CALL(Queue->executeCommandList(ZeCopyEngineImmediateListHelper, false,
                                      false));
  }

  ze_event_handle_t &EventToSignal =
      DoProfiling ? CommandBuffer->ComputeFinishedEvent->ZeEvent
                  : (*Event)->ZeEvent;
  ZE2UR_CALL(Platform->ZeCommandListImmediateAppendExt
                 .zeCommandListImmediateAppendCommandListsExp,
             (CommandListHelper->first, 1, &CommandBuffer->ZeComputeCommandList,
              EventToSignal, WaitList.Length, WaitList.ZeEventList));

  if (DoProfiling) {
    UR_CALL(appendProfilingQueries(CommandBuffer, CommandListHelper->first,
                                   *Event, CommandBuffer->ComputeFinishedEvent,
                                   *Event));
  }

  // When the current execution is finished, signal ExecutionFinishedEvent to
  // reset all the events and prepare for the next execution.
  if (CommandBuffer->ZeCommandListResetEvents) {
    ZE2UR_CALL(zeCommandListAppendBarrier,
               (CommandListHelper->first,
                CommandBuffer->ExecutionFinishedEvent->ZeEvent, 0, nullptr));

    ZE2UR_CALL(Platform->ZeCommandListImmediateAppendExt
                   .zeCommandListImmediateAppendCommandListsExp,
               (CommandListHelper->first, 1,
                &CommandBuffer->ZeCommandListResetEvents, nullptr, 0, nullptr));
  }

  /* The event needs to be retained since it will be used later by the
     command-buffer. If not retained, it might be released when
     ZeImmediateListHelper is reset. If there is an existing event from a
     previous submission of the command-buffer, release it since it is no longer
     needed. */
  if (CommandBuffer->CurrentSubmissionEvent) {
    UR_CALL(urEventReleaseInternal(CommandBuffer->CurrentSubmissionEvent));
  }
  (*Event)->RefCount.increment();
  CommandBuffer->CurrentSubmissionEvent = *Event;

  UR_CALL(Queue->executeCommandList(CommandListHelper, false, false));

  return UR_RESULT_SUCCESS;
}

/* Enqueue the command-buffer using zeCommandQueueExecuteCommandLists.
 * Also uses separate command-lists to wait for the dependencies and to
 * signal the execution finished event. */
ur_result_t enqueueWaitEventPath(ur_exp_command_buffer_handle_t CommandBuffer,
                                 ur_queue_handle_t Queue,
                                 uint32_t NumEventsInWaitList,
                                 const ur_event_handle_t *EventWaitList,
                                 ur_event_handle_t *Event,
                                 ur_command_list_ptr_t SignalCommandList,
                                 bool DoProfiling) {

  ze_command_queue_handle_t ZeCommandQueue;
  getZeCommandQueue(Queue, false, ZeCommandQueue);

  ze_fence_handle_t ZeFence;
  CommandBuffer->getFenceForQueue(ZeCommandQueue, ZeFence);

  UR_CALL(waitForDependencies(CommandBuffer, Queue, NumEventsInWaitList,
                              EventWaitList));

  // Submit reset events command-list. This command-list is of a batch
  // command-list type, regardless of the UR Queue type. We therefore need to
  // submit the list directly using the Level-Zero API to avoid type
  // mismatches if using UR functions.
  ZE2UR_CALL(
      zeCommandQueueExecuteCommandLists,
      (ZeCommandQueue, 1, &CommandBuffer->ZeCommandListResetEvents, nullptr));

  // Submit main command-list. This command-list is of a batch command-list
  // type, regardless of the UR Queue type. We therefore need to submit the
  // list directly using the Level-Zero API to avoid type mismatches if using
  // UR functions.
  ZE2UR_CALL(
      zeCommandQueueExecuteCommandLists,
      (ZeCommandQueue, 1, &CommandBuffer->ZeComputeCommandList, ZeFence));

  // The Copy command-list is submitted to the main copy queue if it is not
  // empty.
  if (!CommandBuffer->MCopyCommandListEmpty) {
    ze_command_queue_handle_t ZeCopyCommandQueue;
    getZeCommandQueue(Queue, true, ZeCopyCommandQueue);
    ZE2UR_CALL(
        zeCommandQueueExecuteCommandLists,
        (ZeCopyCommandQueue, 1, &CommandBuffer->ZeCopyCommandList, nullptr));
  }

  ZE2UR_CALL(zeCommandListAppendBarrier,
             (SignalCommandList->first, nullptr, 1,
              &(CommandBuffer->ExecutionFinishedEvent->ZeEvent)));

  // Reset the wait-event for the UR command-buffer that is signaled when its
  // submission dependencies have been satisfied.
  ZE2UR_CALL(zeCommandListAppendEventReset,
             (SignalCommandList->first, CommandBuffer->WaitEvent->ZeEvent));

  // Reset the all-reset-event for the UR command-buffer that is signaled when
  // all events of the main command-list have been reset.
  ZE2UR_CALL(zeCommandListAppendEventReset,
             (SignalCommandList->first, CommandBuffer->AllResetEvent->ZeEvent));

  if (DoProfiling) {
    UR_CALL(appendProfilingQueries(CommandBuffer, SignalCommandList->first,
                                   nullptr, nullptr, *Event));
  }

  ZE2UR_CALL(zeCommandListAppendBarrier,
             (SignalCommandList->first, (*Event)->ZeEvent, 0, nullptr));

  /* The event needs to be retained since it will be used later by the
     command-buffer. If there is an existing event from a
     previous submission of the command-buffer, release it since it is no longer
     needed. */
  if (CommandBuffer->CurrentSubmissionEvent) {
    UR_CALL(urEventReleaseInternal(CommandBuffer->CurrentSubmissionEvent));
  }
  (*Event)->RefCount.increment();
  CommandBuffer->CurrentSubmissionEvent = *Event;

  UR_CALL(Queue->executeCommandList(SignalCommandList, false /*IsBlocking*/,
                                    false /*OKToBatchCommand*/));

  return UR_RESULT_SUCCESS;
}

ur_result_t urEnqueueCommandBufferExp(
    ur_queue_handle_t UrQueue, ur_exp_command_buffer_handle_t CommandBuffer,
    uint32_t NumEventsInWaitList, const ur_event_handle_t *EventWaitList,
    ur_event_handle_t *Event) {

  std::scoped_lock<ur_shared_mutex> Lock(UrQueue->Mutex);

  UR_CALL(waitForOngoingExecution(CommandBuffer));

  const bool IsInternal = (Event == nullptr);
  const bool DoProfiling =
      (UrQueue->Properties & UR_QUEUE_FLAG_PROFILING_ENABLE) &&
      (!CommandBuffer->IsInOrderCmdList) &&
      (CommandBuffer->IsProfilingEnabled) && Event;
  ur_event_handle_t InternalEvent;
  ur_event_handle_t *OutEvent = Event ? Event : &InternalEvent;

  ur_command_list_ptr_t ZeCommandListHelper{};
  UR_CALL(UrQueue->Context->getAvailableCommandList(
      UrQueue, ZeCommandListHelper, false /*UseCopyEngine*/,
      NumEventsInWaitList, EventWaitList, false /*AllowBatching*/,
      nullptr /*ForcedCmdQueue*/));

  UR_CALL(createEventAndAssociateQueue(
      UrQueue, OutEvent, UR_COMMAND_ENQUEUE_COMMAND_BUFFER_EXP,
      ZeCommandListHelper, IsInternal, false, std::nullopt));

  if (CommandBuffer->UseImmediateAppendPath) {
    UR_CALL(enqueueImmediateAppendPath(
        CommandBuffer, UrQueue, NumEventsInWaitList, EventWaitList, OutEvent,
        ZeCommandListHelper, DoProfiling));
  } else {
    UR_CALL(enqueueWaitEventPath(CommandBuffer, UrQueue, NumEventsInWaitList,
                                 EventWaitList, OutEvent, ZeCommandListHelper,
                                 DoProfiling));
  }

  return UR_RESULT_SUCCESS;
}

// anonymous namespace of update helper functions
namespace {

/**
 * Update the kernel command with the new values.
 * @param[in] CommandBuffer The command-buffer which is being updated.
 * @param[in] NumKernelUpdates Length of /p CommadnDescs.
 * @param[in] CommandDescs List of update command descriptions.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t updateCommandBuffer(
    ur_exp_command_buffer_handle_t CommandBuffer, uint32_t NumKernelUpdates,
    const ur_exp_command_buffer_update_kernel_launch_desc_t *CommandDescs) {
  std::scoped_lock<ur_shared_mutex> Guard(CommandBuffer->Mutex);
  auto Platform = CommandBuffer->Context->getPlatform();
  ze_command_list_handle_t ZeCommandList =
      CommandBuffer->ZeComputeCommandListTranslated;
  if (Platform->ZeMutableCmdListExt.LoaderExtension) {
    ZeCommandList = CommandBuffer->ZeComputeCommandList;
  }

  UR_CALL(updateCommandBufferUnlocked(
      getZeKernelWrapped, getMemPtr, ZeCommandList, Platform,
      CommandBuffer->Device, nullptr, NumKernelUpdates, CommandDescs));

  ZE2UR_CALL(zeCommandListClose, (CommandBuffer->ZeComputeCommandList));

  return UR_RESULT_SUCCESS;
}

} // namespace

ur_result_t urCommandBufferUpdateKernelLaunchExp(
    ur_exp_command_buffer_handle_t CommandBuffer, uint32_t numKernelUpdates,
    const ur_exp_command_buffer_update_kernel_launch_desc_t *CommandDesc) {
  UR_ASSERT(CommandBuffer->IsUpdatable && CommandBuffer->IsFinalized,
            UR_RESULT_ERROR_INVALID_OPERATION);
  {
    std::scoped_lock<ur_shared_mutex> Guard(CommandBuffer->Mutex);
    UR_CALL(
        validateCommandDescUnlocked(CommandBuffer, CommandBuffer->Device,
                                    CommandBuffer->Context->getPlatform()
                                        ->ZeDriverGlobalOffsetExtensionFound,
                                    numKernelUpdates, CommandDesc));
  }

  UR_CALL(waitForOngoingExecution(CommandBuffer));

  UR_CALL(updateCommandBuffer(CommandBuffer, numKernelUpdates, CommandDesc));

  return UR_RESULT_SUCCESS;
}

ur_result_t urCommandBufferUpdateSignalEventExp(
    ur_exp_command_buffer_command_handle_t /*Command*/,
    ur_event_handle_t * /*Event*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urCommandBufferUpdateWaitEventsExp(
    ur_exp_command_buffer_command_handle_t /*Command*/,
    uint32_t /*NumEventsInWaitList*/,
    const ur_event_handle_t * /*EventWaitList*/) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t
urCommandBufferGetInfoExp(ur_exp_command_buffer_handle_t hCommandBuffer,
                          ur_exp_command_buffer_info_t propName,
                          size_t propSize, void *pPropValue,
                          size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_EXP_COMMAND_BUFFER_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{hCommandBuffer->RefCount.load()});
  case UR_EXP_COMMAND_BUFFER_INFO_DESCRIPTOR: {
    ur_exp_command_buffer_desc_t Descriptor{};
    Descriptor.stype = UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC;
    Descriptor.pNext = nullptr;
    Descriptor.isUpdatable = hCommandBuffer->IsUpdatable;
    Descriptor.isInOrder = hCommandBuffer->IsInOrderCmdList;
    Descriptor.enableProfiling = hCommandBuffer->IsProfilingEnabled;

    return ReturnValue(Descriptor);
  }
  default:
    assert(false && "Command-buffer info request not implemented");
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

ur_result_t urCommandBufferAppendNativeCommandExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,
    ur_exp_command_buffer_native_command_function_t pfnNativeCommand,
    void *pData, ur_exp_command_buffer_handle_t,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  // Use Compute command-list as we don't know the types of commands the
  // user will choose to append.
  ze_command_list_handle_t ZeCommandList = hCommandBuffer->ZeComputeCommandList;

  std::vector<ze_event_handle_t> ZeEventList;
  ze_event_handle_t ZeLaunchEvent = nullptr;
  UR_CALL(createSyncPointAndGetZeEvents(
      UR_COMMAND_ENQUEUE_NATIVE_EXP, hCommandBuffer, ZeCommandList,
      numSyncPointsInWaitList, pSyncPointWaitList, true, pSyncPoint,
      ZeEventList, ZeLaunchEvent));

  // Barrier on all commands before user defined commands.
  ZE2UR_CALL(zeCommandListAppendBarrier,
             (ZeCommandList, nullptr, ZeEventList.size(),
              getPointerFromVector(ZeEventList)));

  // Call user-defined function immediately
  pfnNativeCommand(pData);

  // Barrier on all commands after user defined commands.
  ZE2UR_CALL(zeCommandListAppendBarrier,
             (ZeCommandList, ZeLaunchEvent, 0, nullptr));

  return UR_RESULT_SUCCESS;
}

ur_result_t
urCommandBufferGetNativeHandleExp(ur_exp_command_buffer_handle_t hCommandBuffer,
                                  ur_native_handle_t *phNativeCommandBuffer) {
  // Return Compute command-list as it is guaranteed to always exist
  ze_command_list_handle_t ZeCommandList = hCommandBuffer->ZeComputeCommandList;
  *phNativeCommandBuffer = reinterpret_cast<ur_native_handle_t>(ZeCommandList);
  return UR_RESULT_SUCCESS;
}
} // namespace ur::level_zero
