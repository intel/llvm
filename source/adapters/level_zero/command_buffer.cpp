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
#include "ur_level_zero.hpp"

/* Command-buffer Extension

  The UR interface for submitting a UR command-buffer takes a list
  of events to wait on, and returns an event representing the completion of
  that particular submission of the command-buffer.

  However, in `zeCommandQueueExecuteCommandLists` there are no parameters to
  take a waitlist and also the only sync primitive returned is to block on
  host.

  In order to get the UR command-buffer enqueue semantics we want with L0
  this adapter adds extra commands to the L0 command-list representing a
  UR command-buffer.

  Prefix - Commands added to the start of the L0 command-list by L0 adapter.
  Suffix - Commands added to the end of the L0 command-list by L0 adapter.

  These extra commands operate on L0 event synchronisation primitives used by
  the command-list to interact with the external UR wait-list and UR return
  event required for the enqueue interface.

  The `ur_exp_command_buffer_handle_t` class for this adapter contains a
  SignalEvent which signals the completion of the command-list in the suffix,
  and is reset in the prefix. This signal is detected by a new UR return event
  created on UR command-buffer enqueue.

  There is also a WaitEvent used by the `ur_exp_command_buffer_handle_t` class
  in the prefix to wait on any dependencies passed in the enqueue wait-list.
  This WaitEvent is reset at the end of the suffix, along with reset commands
  to reset the L0 events used to implement the UR sync-points.

  ┌──────────┬────────────────────────────────────────────────┬─────────┐
  │  Prefix  │ Commands added to UR command-buffer by UR user │ Suffix  │
  └──────────┴────────────────────────────────────────────────┴─────────┘

            ┌───────────────────┬──────────────┐──────────────────────────────┐
  Prefix    │Reset signal event │ Reset events │ Barrier waiting on wait event│
            └───────────────────┴──────────────┘──────────────────────────────┘

            ┌─────────────────────────────────────────────┐──────────────┐
  Suffix    │Barrier waiting on sync-point event,         │  Query CMD   │
            │signaling the UR command-buffer signal event │  Timestamps  │
            └─────────────────────────────────────────────┘──────────────┘

  For a call to `urCommandBufferEnqueueExp` with an event_list `EL`,
  command-buffer `CB`, and return event `RE` our implementation has to create
  and submit two new command-lists for the above approach to work. One before
  the command-list with extra commands associated with `CB`, and the other
  after `CB`.

  Command-list created on `urCommandBufferEnqueueExp` to execution before `CB`:
  ┌───────────────────────────────────────────────────────────┐
  │Barrier on `EL` than signals `CB` WaitEvent when completed │
  └───────────────────────────────────────────────────────────┘

  Command-list created on `urCommandBufferEnqueueExp` to execution after `CB`:
  ┌─────────────────────────────────────────────────────────────┐
  │Barrier on `CB` SignalEvent that signals `RE` when completed │
  └─────────────────────────────────────────────────────────────┘

Drawbacks
---------

There are two drawbacks to this approach:

1. We use 3x the command-list resources, if there are many UR command-buffers
in flight, this may exhaust L0 driver resources.

2. Each command list is submitted individually with a
`ur_queue_handle_t_::executeCommandList` call which introduces serialization in
the submission pipeline that is heavier than having a barrier or a
waitForEvents on the same list. Resulting in additional latency when executing
graphs.

*/

ur_exp_command_buffer_handle_t_::ur_exp_command_buffer_handle_t_(
    ur_context_handle_t Context, ur_device_handle_t Device,
    ze_command_list_handle_t CommandList,
    ZeStruct<ze_command_list_desc_t> ZeDesc,
    const ur_exp_command_buffer_desc_t *Desc)
    : Context(Context), Device(Device), ZeCommandList(CommandList),
      ZeCommandListDesc(ZeDesc), ZeFencesList(), QueueProperties(),
      SyncPoints(), NextSyncPoint(0) {
  (void)Desc;
  urContextRetain(Context);
  urDeviceRetain(Device);
}

// The ur_exp_command_buffer_handle_t_ destructor release all the memory objects
// allocated for command_buffer managment
ur_exp_command_buffer_handle_t_::~ur_exp_command_buffer_handle_t_() {
  // Release the memory allocated to the Context stored in the command_buffer
  urContextRelease(Context);

  // Release the device
  urDeviceRelease(Device);

  // Release the memory allocated to the CommandList stored in the
  // command_buffer
  if (ZeCommandList) {
    ZE_CALL_NOCHECK(zeCommandListDestroy, (ZeCommandList));
  }

  // Release additional signal and wait events used by command_buffer
  if (SignalEvent) {
    CleanupCompletedEvent(SignalEvent, false);
    urEventReleaseInternal(SignalEvent);
  }
  if (WaitEvent) {
    CleanupCompletedEvent(WaitEvent, false);
    urEventReleaseInternal(WaitEvent);
  }

  // Release events added to the command_buffer
  for (auto &Sync : SyncPoints) {
    auto &Event = Sync.second;
    CleanupCompletedEvent(Event, false);
    urEventReleaseInternal(Event);
  }

  // Release Fences allocated to command_buffer
  for (auto &ZeFence : ZeFencesList) {
    ZE_CALL_NOCHECK(zeFenceDestroy, (ZeFence));
  }
}

/// Helper function for calculating work dimensions for kernels
ur_result_t calculateKernelWorkDimensions(
    ur_kernel_handle_t Kernel, ur_device_handle_t Device,
    ze_group_count_t &ZeThreadGroupDimensions, uint32_t (&WG)[3],
    uint32_t WorkDim, const size_t *GlobalWorkSize,
    const size_t *LocalWorkSize) {
  // global_work_size of unused dimensions must be set to 1
  UR_ASSERT(WorkDim == 3 || GlobalWorkSize[2] == 1,
            UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT(WorkDim >= 2 || GlobalWorkSize[1] == 1,
            UR_RESULT_ERROR_INVALID_VALUE);

  if (LocalWorkSize) {
    WG[0] = ur_cast<uint32_t>(LocalWorkSize[0]);
    WG[1] = ur_cast<uint32_t>(LocalWorkSize[1]);
    WG[2] = ur_cast<uint32_t>(LocalWorkSize[2]);
  } else {
    // We can't call to zeKernelSuggestGroupSize if 64-bit GlobalWorkSize
    // values do not fit to 32-bit that the API only supports currently.
    bool SuggestGroupSize = true;
    for (int I : {0, 1, 2}) {
      if (GlobalWorkSize[I] > UINT32_MAX) {
        SuggestGroupSize = false;
      }
    }
    if (SuggestGroupSize) {
      ZE2UR_CALL(zeKernelSuggestGroupSize,
                 (Kernel->ZeKernel, GlobalWorkSize[0], GlobalWorkSize[1],
                  GlobalWorkSize[2], &WG[0], &WG[1], &WG[2]));
    } else {
      for (int I : {0, 1, 2}) {
        // Try to find a I-dimension WG size that the GlobalWorkSize[I] is
        // fully divisable with. Start with the max possible size in
        // each dimension.
        uint32_t GroupSize[] = {
            Device->ZeDeviceComputeProperties->maxGroupSizeX,
            Device->ZeDeviceComputeProperties->maxGroupSizeY,
            Device->ZeDeviceComputeProperties->maxGroupSizeZ};
        GroupSize[I] = (std::min)(size_t(GroupSize[I]), GlobalWorkSize[I]);
        while (GlobalWorkSize[I] % GroupSize[I]) {
          --GroupSize[I];
        }
        if (GlobalWorkSize[I] / GroupSize[I] > UINT32_MAX) {
          urPrint("urCommandBufferAppendKernelLaunchExp: can't find a WG size "
                  "suitable for global work size > UINT32_MAX\n");
          return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
        }
        WG[I] = GroupSize[I];
      }
      urPrint(
          "urCommandBufferAppendKernelLaunchExp: using computed WG size = {%d, "
          "%d, %d}\n",
          WG[0], WG[1], WG[2]);
    }
  }

  // TODO: assert if sizes do not fit into 32-bit?
  switch (WorkDim) {
  case 3:
    ZeThreadGroupDimensions.groupCountX =
        ur_cast<uint32_t>(GlobalWorkSize[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        ur_cast<uint32_t>(GlobalWorkSize[1] / WG[1]);
    ZeThreadGroupDimensions.groupCountZ =
        ur_cast<uint32_t>(GlobalWorkSize[2] / WG[2]);
    break;
  case 2:
    ZeThreadGroupDimensions.groupCountX =
        ur_cast<uint32_t>(GlobalWorkSize[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        ur_cast<uint32_t>(GlobalWorkSize[1] / WG[1]);
    WG[2] = 1;
    break;
  case 1:
    ZeThreadGroupDimensions.groupCountX =
        ur_cast<uint32_t>(GlobalWorkSize[0] / WG[0]);
    WG[1] = WG[2] = 1;
    break;

  default:
    urPrint("urCommandBufferAppendKernelLaunchExp: unsupported work_dim\n");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  // Error handling for non-uniform group size case
  if (GlobalWorkSize[0] !=
      size_t(ZeThreadGroupDimensions.groupCountX) * WG[0]) {
    urPrint("urCommandBufferAppendKernelLaunchExp: invalid work_dim. The range "
            "is not a "
            "multiple of the group size in the 1st dimension\n");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize[1] !=
      size_t(ZeThreadGroupDimensions.groupCountY) * WG[1]) {
    urPrint("urCommandBufferAppendKernelLaunchExp: invalid work_dim. The range "
            "is not a "
            "multiple of the group size in the 2nd dimension\n");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize[2] !=
      size_t(ZeThreadGroupDimensions.groupCountZ) * WG[2]) {
    urPrint("urCommandBufferAppendKernelLaunchExp: invalid work_dim. The range "
            "is not a "
            "multiple of the group size in the 3rd dimension\n");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }

  return UR_RESULT_SUCCESS;
}

/// Helper function for finding the Level Zero events associated with the
/// commands in a command-buffer, each event is pointed to by a sync-point in
/// the wait list.
///
/// @param[in] CommandBuffer to lookup the L0 events from.
/// @param[in] NumSyncPointsInWaitList Length of \p SyncPointWaitList.
/// @param[in] SyncPointWaitList List of sync points in \p CommandBuffer
/// to find the L0 events for.
/// @param[out] ZeEventList Return parameter for the L0 events associated with
/// each sync-point in \p SyncPointWaitList.
///
/// @return UR_RESULT_SUCCESS or an error code on failure
static ur_result_t getEventsFromSyncPoints(
    const ur_exp_command_buffer_handle_t &CommandBuffer,
    size_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    std::vector<ze_event_handle_t> &ZeEventList) {
  // Map of ur_exp_command_buffer_sync_point_t to ur_event_handle_t defining
  // the event associated with each sync-point
  auto SyncPoints = CommandBuffer->SyncPoints;

  // For each sync-point add associated L0 event to the return list.
  for (size_t i = 0; i < NumSyncPointsInWaitList; i++) {
    if (auto EventHandle = SyncPoints.find(SyncPointWaitList[i]);
        EventHandle != SyncPoints.end()) {
      ZeEventList.push_back(EventHandle->second->ZeEvent);
    } else {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  }
  return UR_RESULT_SUCCESS;
}

// Shared by all memory read/write/copy PI interfaces.
// Helper function for common code when enqueuing memory operations to a command
// buffer.
static ur_result_t enqueueCommandBufferMemCopyHelper(
    ur_command_t CommandType, ur_exp_command_buffer_handle_t CommandBuffer,
    void *Dst, const void *Src, size_t Size, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {
  std::vector<ze_event_handle_t> ZeEventList;
  UR_CALL(getEventsFromSyncPoints(CommandBuffer, NumSyncPointsInWaitList,
                                  SyncPointWaitList, ZeEventList));

  ur_event_handle_t LaunchEvent;
  UR_CALL(
      EventCreate(CommandBuffer->Context, nullptr, false, false, &LaunchEvent));
  LaunchEvent->CommandType = CommandType;

  // Get sync point and register the event with it.
  *SyncPoint = CommandBuffer->GetNextSyncPoint();
  CommandBuffer->RegisterSyncPoint(*SyncPoint, LaunchEvent);

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (CommandBuffer->ZeCommandList, Dst, Src, Size,
              LaunchEvent->ZeEvent, ZeEventList.size(), ZeEventList.data()));

  urPrint("calling zeCommandListAppendMemoryCopy() with"
          "  ZeEvent %#" PRIxPTR "\n",
          ur_cast<std::uintptr_t>(LaunchEvent->ZeEvent));

  return UR_RESULT_SUCCESS;
}

// Helper function for common code when enqueuing rectangular memory operations
// to a command buffer.
static ur_result_t enqueueCommandBufferMemCopyRectHelper(
    ur_command_t CommandType, ur_exp_command_buffer_handle_t CommandBuffer,
    void *Dst, const void *Src, ur_rect_offset_t SrcOrigin,
    ur_rect_offset_t DstOrigin, ur_rect_region_t Region, size_t SrcRowPitch,
    size_t DstRowPitch, size_t SrcSlicePitch, size_t DstSlicePitch,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {

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

  std::vector<ze_event_handle_t> ZeEventList;
  UR_CALL(getEventsFromSyncPoints(CommandBuffer, NumSyncPointsInWaitList,
                                  SyncPointWaitList, ZeEventList));

  ur_event_handle_t LaunchEvent;
  UR_CALL(
      EventCreate(CommandBuffer->Context, nullptr, false, false, &LaunchEvent));
  LaunchEvent->CommandType = CommandType;

  // Get sync point and register the event with it.
  *SyncPoint = CommandBuffer->GetNextSyncPoint();
  CommandBuffer->RegisterSyncPoint(*SyncPoint, LaunchEvent);

  ZE2UR_CALL(zeCommandListAppendMemoryCopyRegion,
             (CommandBuffer->ZeCommandList, Dst, &ZeDstRegion, DstPitch,
              DstSlicePitch, Src, &ZeSrcRegion, SrcPitch, SrcSlicePitch,
              LaunchEvent->ZeEvent, ZeEventList.size(), ZeEventList.data()));

  urPrint("calling zeCommandListAppendMemoryCopyRegion() with"
          "  ZeEvent %#" PRIxPTR "\n",
          ur_cast<std::uintptr_t>(LaunchEvent->ZeEvent));

  return UR_RESULT_SUCCESS;
}

// Helper function for enqueuing memory fills
static ur_result_t enqueueCommandBufferFillHelper(
    ur_command_t CommandType, ur_exp_command_buffer_handle_t CommandBuffer,
    void *Ptr, const void *Pattern, size_t PatternSize, size_t Size,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {
  // Pattern size must be a power of two.
  UR_ASSERT((PatternSize > 0) && ((PatternSize & (PatternSize - 1)) == 0),
            UR_RESULT_ERROR_INVALID_VALUE);

  // Pattern size must fit the compute queue capabilities.
  UR_ASSERT(
      PatternSize <=
          CommandBuffer->Device
              ->QueueGroup[ur_device_handle_t_::queue_group_info_t::Compute]
              .ZeProperties.maxMemoryFillPatternSize,
      UR_RESULT_ERROR_INVALID_VALUE);

  std::vector<ze_event_handle_t> ZeEventList;
  UR_CALL(getEventsFromSyncPoints(CommandBuffer, NumSyncPointsInWaitList,
                                  SyncPointWaitList, ZeEventList));

  ur_event_handle_t LaunchEvent;
  UR_CALL(
      EventCreate(CommandBuffer->Context, nullptr, false, true, &LaunchEvent));
  LaunchEvent->CommandType = CommandType;

  // Get sync point and register the event with it.
  *SyncPoint = CommandBuffer->GetNextSyncPoint();
  CommandBuffer->RegisterSyncPoint(*SyncPoint, LaunchEvent);

  ZE2UR_CALL(zeCommandListAppendMemoryFill,
             (CommandBuffer->ZeCommandList, Ptr, Pattern, PatternSize, Size,
              LaunchEvent->ZeEvent, ZeEventList.size(), ZeEventList.data()));

  urPrint("calling zeCommandListAppendMemoryFill() with"
          "  ZeEvent %#lx\n",
          ur_cast<std::uintptr_t>(LaunchEvent->ZeEvent));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferCreateExp(ur_context_handle_t Context, ur_device_handle_t Device,
                         const ur_exp_command_buffer_desc_t *CommandBufferDesc,
                         ur_exp_command_buffer_handle_t *CommandBuffer) {
  // Force compute queue type for now. Copy engine types may be better suited
  // for host to device copies.
  uint32_t QueueGroupOrdinal =
      Device->QueueGroup[ur_device_handle_t_::queue_group_info_t::type::Compute]
          .ZeOrdinal;

  ZeStruct<ze_command_list_desc_t> ZeCommandListDesc;
  ZeCommandListDesc.commandQueueGroupOrdinal = QueueGroupOrdinal;
  // Dependencies between commands are explicitly enforced by sync points when
  // enqueuing. Consequently, relax the command ordering in the command list
  // can enable the backend to further optimize the workload
  ZeCommandListDesc.flags = ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING;

  ze_command_list_handle_t ZeCommandList;
  // TODO We could optimize this by pooling both Level Zero command-lists and UR
  // command-buffers, then reusing them.
  ZE2UR_CALL(zeCommandListCreate, (Context->ZeContext, Device->ZeDevice,
                                   &ZeCommandListDesc, &ZeCommandList));
  try {
    *CommandBuffer = new ur_exp_command_buffer_handle_t_(
        Context, Device, ZeCommandList, ZeCommandListDesc, CommandBufferDesc);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  // Create signal & wait events to be used in the command-list for sync
  // on command-buffer enqueue.
  auto RetCommandBuffer = *CommandBuffer;
  UR_CALL(EventCreate(Context, nullptr, false, false,
                      &RetCommandBuffer->SignalEvent));
  UR_CALL(EventCreate(Context, nullptr, false, false,
                      &RetCommandBuffer->WaitEvent));

  // Add prefix commands
  ZE2UR_CALL(zeCommandListAppendEventReset,
             (ZeCommandList, RetCommandBuffer->SignalEvent->ZeEvent));
  ZE2UR_CALL(
      zeCommandListAppendBarrier,
      (ZeCommandList, nullptr, 1, &RetCommandBuffer->WaitEvent->ZeEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferRetainExp(ur_exp_command_buffer_handle_t CommandBuffer) {
  CommandBuffer->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferReleaseExp(ur_exp_command_buffer_handle_t CommandBuffer) {
  if (!CommandBuffer->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  delete CommandBuffer;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferFinalizeExp(ur_exp_command_buffer_handle_t CommandBuffer) {
  // Create a list of events for our signal event to wait on
  const size_t NumEvents = CommandBuffer->SyncPoints.size();
  std::vector<ze_event_handle_t> WaitEventList{NumEvents};
  for (size_t i = 0; i < NumEvents; i++) {
    WaitEventList[i] = CommandBuffer->SyncPoints[i]->ZeEvent;
  }

  // Wait for all the user added commands to complete, and signal the
  // command-buffer signal-event when they are done.
  ZE2UR_CALL(zeCommandListAppendBarrier,
             (CommandBuffer->ZeCommandList, CommandBuffer->SignalEvent->ZeEvent,
              NumEvents, WaitEventList.data()));

  // Close the command list and have it ready for dispatch.
  ZE2UR_CALL(zeCommandListClose, (CommandBuffer->ZeCommandList));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_kernel_handle_t Kernel,
    uint32_t WorkDim, const size_t *GlobalWorkOffset,
    const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint,
    ur_exp_command_buffer_command_handle_t *) {
  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(
      Kernel->Mutex, Kernel->Program->Mutex);

  if (GlobalWorkOffset != NULL) {
    if (!CommandBuffer->Context->getPlatform()
             ->ZeDriverGlobalOffsetExtensionFound) {
      urPrint("No global offset extension found on this driver\n");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    ZE2UR_CALL(zeKernelSetGlobalOffsetExp,
               (Kernel->ZeKernel, GlobalWorkOffset[0], GlobalWorkOffset[1],
                GlobalWorkOffset[2]));
  }

  // If there are any pending arguments set them now.
  for (auto &Arg : Kernel->PendingArguments) {
    // The ArgValue may be a NULL pointer in which case a NULL value is used for
    // the kernel argument declared as a pointer to global or constant memory.
    char **ZeHandlePtr = nullptr;
    if (Arg.Value) {
      UR_CALL(Arg.Value->getZeHandlePtr(ZeHandlePtr, Arg.AccessMode,
                                        CommandBuffer->Device));
    }
    ZE2UR_CALL(zeKernelSetArgumentValue,
               (Kernel->ZeKernel, Arg.Index, Arg.Size, ZeHandlePtr));
  }
  Kernel->PendingArguments.clear();

  ze_group_count_t ZeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3];

  UR_CALL(calculateKernelWorkDimensions(Kernel, CommandBuffer->Device,
                                        ZeThreadGroupDimensions, WG, WorkDim,
                                        GlobalWorkSize, LocalWorkSize));

  ZE2UR_CALL(zeKernelSetGroupSize, (Kernel->ZeKernel, WG[0], WG[1], WG[2]));

  std::vector<ze_event_handle_t> ZeEventList;
  UR_CALL(getEventsFromSyncPoints(CommandBuffer, NumSyncPointsInWaitList,
                                  SyncPointWaitList, ZeEventList));
  ur_event_handle_t LaunchEvent;
  UR_CALL(
      EventCreate(CommandBuffer->Context, nullptr, false, false, &LaunchEvent));
  LaunchEvent->CommandType = UR_COMMAND_KERNEL_LAUNCH;

  // Get sync point and register the event with it.
  *SyncPoint = CommandBuffer->GetNextSyncPoint();
  CommandBuffer->RegisterSyncPoint(*SyncPoint, LaunchEvent);

  LaunchEvent->CommandData = (void *)Kernel;
  // Increment the reference count of the Kernel and indicate that the Kernel
  // is in use. Once the event has been signaled, the code in
  // CleanupCompletedEvent(Event) will do a urKernelRelease to update the
  // reference count on the kernel, using the kernel saved in CommandData.
  UR_CALL(urKernelRetain(Kernel));

  ZE2UR_CALL(zeCommandListAppendLaunchKernel,
             (CommandBuffer->ZeCommandList, Kernel->ZeKernel,
              &ZeThreadGroupDimensions, LaunchEvent->ZeEvent,
              ZeEventList.size(), ZeEventList.data()));

  urPrint("calling zeCommandListAppendLaunchKernel() with"
          "  ZeEvent %#" PRIxPTR "\n",
          ur_cast<std::uintptr_t>(LaunchEvent->ZeEvent));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMMemcpyExp(
    ur_exp_command_buffer_handle_t CommandBuffer, void *Dst, const void *Src,
    size_t Size, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {
  return enqueueCommandBufferMemCopyHelper(
      UR_COMMAND_USM_MEMCPY, CommandBuffer, Dst, Src, Size,
      NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t SrcMem,
    ur_mem_handle_t DstMem, size_t SrcOffset, size_t DstOffset, size_t Size,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {
  auto SrcBuffer = ur_cast<ur_mem_handle_t>(SrcMem);
  auto DstBuffer = ur_cast<ur_mem_handle_t>(DstMem);

  std::shared_lock<ur_shared_mutex> SrcLock(SrcBuffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex> LockAll(
      SrcLock, DstBuffer->Mutex);

  char *ZeHandleSrc;
  UR_CALL(SrcBuffer->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                                 CommandBuffer->Device));
  char *ZeHandleDst;
  UR_CALL(DstBuffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                 CommandBuffer->Device));

  return enqueueCommandBufferMemCopyHelper(
      UR_COMMAND_MEM_BUFFER_COPY, CommandBuffer, ZeHandleDst + DstOffset,
      ZeHandleSrc + SrcOffset, Size, NumSyncPointsInWaitList, SyncPointWaitList,
      SyncPoint);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyRectExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t SrcMem,
    ur_mem_handle_t DstMem, ur_rect_offset_t SrcOrigin,
    ur_rect_offset_t DstOrigin, ur_rect_region_t Region, size_t SrcRowPitch,
    size_t SrcSlicePitch, size_t DstRowPitch, size_t DstSlicePitch,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {
  auto SrcBuffer = ur_cast<ur_mem_handle_t>(SrcMem);
  auto DstBuffer = ur_cast<ur_mem_handle_t>(DstMem);

  std::shared_lock<ur_shared_mutex> SrcLock(SrcBuffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex> LockAll(
      SrcLock, DstBuffer->Mutex);

  char *ZeHandleSrc;
  UR_CALL(SrcBuffer->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                                 CommandBuffer->Device));
  char *ZeHandleDst;
  UR_CALL(DstBuffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                 CommandBuffer->Device));

  return enqueueCommandBufferMemCopyRectHelper(
      UR_COMMAND_MEM_BUFFER_COPY_RECT, CommandBuffer, ZeHandleDst, ZeHandleSrc,
      SrcOrigin, DstOrigin, Region, SrcRowPitch, DstRowPitch, SrcSlicePitch,
      DstSlicePitch, NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t Buffer,
    size_t Offset, size_t Size, const void *Src,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {
  std::scoped_lock<ur_shared_mutex> Lock(Buffer->Mutex);

  char *ZeHandleDst = nullptr;
  UR_CALL(Buffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                              CommandBuffer->Device));

  return enqueueCommandBufferMemCopyHelper(
      UR_COMMAND_MEM_BUFFER_WRITE, CommandBuffer,
      ZeHandleDst + Offset, // dst
      Src,                  // src
      Size, NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteRectExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t Buffer,
    ur_rect_offset_t BufferOffset, ur_rect_offset_t HostOffset,
    ur_rect_region_t Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, void *Src,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {
  std::scoped_lock<ur_shared_mutex> Lock(Buffer->Mutex);

  char *ZeHandleDst = nullptr;
  UR_CALL(Buffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                              CommandBuffer->Device));
  return enqueueCommandBufferMemCopyRectHelper(
      UR_COMMAND_MEM_BUFFER_WRITE_RECT, CommandBuffer, ZeHandleDst,
      const_cast<char *>(static_cast<const char *>(Src)), HostOffset,
      BufferOffset, Region, HostRowPitch, BufferRowPitch, HostSlicePitch,
      BufferSlicePitch, NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t Buffer,
    size_t Offset, size_t Size, void *Dst, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {
  std::scoped_lock<ur_shared_mutex> SrcLock(Buffer->Mutex);

  char *ZeHandleSrc = nullptr;
  UR_CALL(Buffer->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                              CommandBuffer->Device));
  return enqueueCommandBufferMemCopyHelper(
      UR_COMMAND_MEM_BUFFER_READ, CommandBuffer, Dst, ZeHandleSrc + Offset,
      Size, NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadRectExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t Buffer,
    ur_rect_offset_t BufferOffset, ur_rect_offset_t HostOffset,
    ur_rect_region_t Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, void *Dst,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {
  std::scoped_lock<ur_shared_mutex> SrcLock(Buffer->Mutex);

  char *ZeHandleSrc;
  UR_CALL(Buffer->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                              CommandBuffer->Device));
  return enqueueCommandBufferMemCopyRectHelper(
      UR_COMMAND_MEM_BUFFER_READ_RECT, CommandBuffer, Dst, ZeHandleSrc,
      BufferOffset, HostOffset, Region, BufferRowPitch, HostRowPitch,
      BufferSlicePitch, HostSlicePitch, NumSyncPointsInWaitList,
      SyncPointWaitList, SyncPoint);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMPrefetchExp(
    ur_exp_command_buffer_handle_t CommandBuffer, const void *Mem, size_t Size,
    ur_usm_migration_flags_t Flags, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {
  std::ignore = Flags;

  std::vector<ze_event_handle_t> ZeEventList;
  UR_CALL(getEventsFromSyncPoints(CommandBuffer, NumSyncPointsInWaitList,
                                  SyncPointWaitList, ZeEventList));

  if (NumSyncPointsInWaitList) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (CommandBuffer->ZeCommandList, NumSyncPointsInWaitList,
                ZeEventList.data()));
  }

  ur_event_handle_t LaunchEvent;
  UR_CALL(
      EventCreate(CommandBuffer->Context, nullptr, false, true, &LaunchEvent));
  LaunchEvent->CommandType = UR_COMMAND_USM_PREFETCH;

  // Get sync point and register the event with it.
  *SyncPoint = CommandBuffer->GetNextSyncPoint();
  CommandBuffer->RegisterSyncPoint(*SyncPoint, LaunchEvent);

  // Add the prefetch command to the command buffer.
  // Note that L0 does not handle migration flags.
  ZE2UR_CALL(zeCommandListAppendMemoryPrefetch,
             (CommandBuffer->ZeCommandList, Mem, Size));

  // Level Zero does not have a completion "event" with the prefetch API,
  // so manually add command to signal our event.
  ZE2UR_CALL(zeCommandListAppendSignalEvent,
             (CommandBuffer->ZeCommandList, LaunchEvent->ZeEvent));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMAdviseExp(
    ur_exp_command_buffer_handle_t CommandBuffer, const void *Mem, size_t Size,
    ur_usm_advice_flags_t Advice, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {
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
  UR_CALL(getEventsFromSyncPoints(CommandBuffer, NumSyncPointsInWaitList,
                                  SyncPointWaitList, ZeEventList));

  if (NumSyncPointsInWaitList) {
    ZE2UR_CALL(zeCommandListAppendWaitOnEvents,
               (CommandBuffer->ZeCommandList, NumSyncPointsInWaitList,
                ZeEventList.data()));
  }

  ur_event_handle_t LaunchEvent;
  UR_CALL(
      EventCreate(CommandBuffer->Context, nullptr, false, true, &LaunchEvent));
  LaunchEvent->CommandType = UR_COMMAND_USM_ADVISE;

  // Get sync point and register the event with it.
  *SyncPoint = CommandBuffer->GetNextSyncPoint();
  CommandBuffer->RegisterSyncPoint(*SyncPoint, LaunchEvent);

  ZE2UR_CALL(zeCommandListAppendMemAdvise,
             (CommandBuffer->ZeCommandList, CommandBuffer->Device->ZeDevice,
              Mem, Size, ZeAdvice));

  // Level Zero does not have a completion "event" with the advise API,
  // so manually add command to signal our event.
  ZE2UR_CALL(zeCommandListAppendSignalEvent,
             (CommandBuffer->ZeCommandList, LaunchEvent->ZeEvent));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferFillExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_mem_handle_t Buffer,
    const void *Pattern, size_t PatternSize, size_t Offset, size_t Size,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {

  std::scoped_lock<ur_shared_mutex> Lock(Buffer->Mutex);

  char *ZeHandleDst = nullptr;
  _ur_buffer *UrBuffer = reinterpret_cast<_ur_buffer *>(Buffer);
  UR_CALL(UrBuffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                CommandBuffer->Device));

  return enqueueCommandBufferFillHelper(
      UR_COMMAND_MEM_BUFFER_FILL, CommandBuffer, ZeHandleDst + Offset,
      Pattern,     // It will be interpreted as an 8-bit value,
      PatternSize, // which is indicated with this pattern_size==1
      Size, NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMFillExp(
    ur_exp_command_buffer_handle_t CommandBuffer, void *Ptr,
    const void *Pattern, size_t PatternSize, size_t Size,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {

  return enqueueCommandBufferFillHelper(
      UR_COMMAND_MEM_BUFFER_FILL, CommandBuffer, Ptr,
      Pattern,     // It will be interpreted as an 8-bit value,
      PatternSize, // which is indicated with this pattern_size==1
      Size, NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_queue_handle_t Queue,
    uint32_t NumEventsInWaitList, const ur_event_handle_t *EventWaitList,
    ur_event_handle_t *Event) {
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);
  // Use compute engine rather than copy engine
  const auto UseCopyEngine = false;
  auto &QGroup = Queue->getQueueGroup(UseCopyEngine);
  uint32_t QueueGroupOrdinal;
  auto &ZeCommandQueue = QGroup.getZeQueue(&QueueGroupOrdinal);

  ze_fence_handle_t ZeFence;
  ZeStruct<ze_fence_desc_t> ZeFenceDesc;

  ZE2UR_CALL(zeFenceCreate, (ZeCommandQueue, &ZeFenceDesc, &ZeFence));
  CommandBuffer->ZeFencesList.push_back(ZeFence);

  // Create command-list to execute before `CommandListPtr` and will signal
  // when `EventWaitList` dependencies are complete.
  ur_command_list_ptr_t WaitCommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(Queue, WaitCommandList, false,
                                                  false));

  // Create a list of events of all the events that compose the command buffer
  // workload.
  // This loop also resets the L0 events we use for command-buffer internal
  // sync-points to the non-signaled state.
  // This is required for multiple submissions.
  const size_t NumEvents = CommandBuffer->SyncPoints.size();
  std::vector<ze_event_handle_t> WaitEventList{NumEvents};
  for (size_t i = 0; i < NumEvents; i++) {
    auto ZeEvent = CommandBuffer->SyncPoints[i]->ZeEvent;
    WaitEventList[i] = ZeEvent;
    ZE2UR_CALL(zeCommandListAppendEventReset,
               (WaitCommandList->first, ZeEvent));
  }

  bool MustSignalWaitEvent = true;
  if (NumEventsInWaitList) {
    _ur_ze_event_list_t TmpWaitList;
    UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
        NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

    // Update the WaitList of the Wait Event
    // Events are appended to the WaitList if the WaitList is not empty
    if (CommandBuffer->WaitEvent->WaitList.isEmpty())
      CommandBuffer->WaitEvent->WaitList = TmpWaitList;
    else
      CommandBuffer->WaitEvent->WaitList.insert(TmpWaitList);

    if (!CommandBuffer->WaitEvent->WaitList.isEmpty()) {
      ZE2UR_CALL(zeCommandListAppendBarrier,
                 (WaitCommandList->first, CommandBuffer->WaitEvent->ZeEvent,
                  CommandBuffer->WaitEvent->WaitList.Length,
                  CommandBuffer->WaitEvent->WaitList.ZeEventList));
      MustSignalWaitEvent = false;
    }
  }
  if (MustSignalWaitEvent) {
    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (WaitCommandList->first, CommandBuffer->WaitEvent->ZeEvent));
  }
  Queue->executeCommandList(WaitCommandList, false, false);

  // Submit main command-list. This command-list is of a batch command-list
  // type, regardless of the UR Queue type. We therefore need to submit the list
  // directly using the Level-Zero API to avoid type mismatches if using UR
  // functions.
  ZE2UR_CALL(zeCommandQueueExecuteCommandLists,
             (ZeCommandQueue, 1, &CommandBuffer->ZeCommandList, ZeFence));

  // Execution event for this enqueue of the UR command-buffer
  ur_event_handle_t RetEvent{};

  // Create a command-list to signal RetEvent on completion
  ur_command_list_ptr_t SignalCommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(Queue, SignalCommandList,
                                                  false, false));
  // Reset the wait-event for the UR command-buffer that is signaled when its
  // submission dependencies have been satisfied.
  ZE2UR_CALL(zeCommandListAppendEventReset,
             (SignalCommandList->first, CommandBuffer->WaitEvent->ZeEvent));

  if (Event) {
    UR_CALL(createEventAndAssociateQueue(
        Queue, &RetEvent, UR_COMMAND_COMMAND_BUFFER_ENQUEUE_EXP,
        SignalCommandList, false, false, true));

    if ((Queue->Properties & UR_QUEUE_FLAG_PROFILING_ENABLE)) {
      // Multiple submissions of a command buffer implies that we need to save
      // the event timestamps before resubmiting the command buffer. We
      // therefore copy the these timestamps in a dedicated USM memory section
      // before completing the command buffer execution, and then attach this
      // memory to the event returned to users to allow to allow the profiling
      // engine to recover these timestamps.
      command_buffer_profiling_t *Profiling = new command_buffer_profiling_t();

      Profiling->NumEvents = WaitEventList.size();
      Profiling->Timestamps =
          new ze_kernel_timestamp_result_t[Profiling->NumEvents];

      ZE2UR_CALL(zeCommandListAppendQueryKernelTimestamps,
                 (SignalCommandList->first, WaitEventList.size(),
                  WaitEventList.data(), (void *)Profiling->Timestamps, 0,
                  RetEvent->ZeEvent, 1,
                  &(CommandBuffer->SignalEvent->ZeEvent)));

      RetEvent->CommandData = static_cast<void *>(Profiling);
    } else {
      ZE2UR_CALL(zeCommandListAppendBarrier,
                 (SignalCommandList->first, RetEvent->ZeEvent, 1,
                  &(CommandBuffer->SignalEvent->ZeEvent)));
    }
  }

  Queue->executeCommandList(SignalCommandList, false, false);

  if (Event) {
    *Event = RetEvent;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferRetainCommandExp(ur_exp_command_buffer_command_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferReleaseCommandExp(ur_exp_command_buffer_command_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferUpdateKernelLaunchExp(
    ur_exp_command_buffer_command_handle_t,
    const ur_exp_command_buffer_update_kernel_launch_desc_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferGetInfoExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,
    ur_exp_command_buffer_info_t propName, size_t propSize, void *pPropValue,
    size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_EXP_COMMAND_BUFFER_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{hCommandBuffer->RefCount.load()});
  default:
    assert(!"Command-buffer info request not implemented");
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferCommandGetInfoExp(
    ur_exp_command_buffer_command_handle_t,
    ur_exp_command_buffer_command_info_t, size_t, void *, size_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
