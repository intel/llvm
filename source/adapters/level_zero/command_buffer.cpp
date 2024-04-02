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

/* L0 Command-buffer Extension Doc see:
https://github.com/intel/llvm/blob/sycl/sycl/doc/design/CommandGraph.md#level-zero
*/

ur_exp_command_buffer_handle_t_::ur_exp_command_buffer_handle_t_(
    ur_context_handle_t Context, ur_device_handle_t Device,
    ze_command_list_handle_t CommandList,
    ze_command_list_handle_t CommandListResetEvents,
    ZeStruct<ze_command_list_desc_t> ZeDesc,
    const ur_exp_command_buffer_desc_t *Desc)
    : Context(Context), Device(Device), ZeCommandList(CommandList),
      ZeCommandListResetEvents(CommandListResetEvents),
      ZeCommandListDesc(ZeDesc), ZeFencesList(), QueueProperties(),
      SyncPoints(), NextSyncPoint(0),
      IsUpdatable(Desc ? Desc->isUpdatable : false) {
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

  // Release the memory allocated to the CommandListResetEvents stored in the
  // command_buffer
  if (ZeCommandListResetEvents) {
    ZE_CALL_NOCHECK(zeCommandListDestroy, (ZeCommandListResetEvents));
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
  if (AllResetEvent) {
    CleanupCompletedEvent(AllResetEvent, false);
    urEventReleaseInternal(AllResetEvent);
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

ur_exp_command_buffer_command_handle_t_::
    ur_exp_command_buffer_command_handle_t_(
        ur_exp_command_buffer_handle_t CommandBuffer, uint64_t CommandId,
        ur_kernel_handle_t Kernel = nullptr)
    : CommandBuffer(CommandBuffer), CommandId(CommandId), Kernel(Kernel) {
  urCommandBufferRetainExp(CommandBuffer);
  if (Kernel)
    urKernelRetain(Kernel);
}

ur_exp_command_buffer_command_handle_t_::
    ~ur_exp_command_buffer_command_handle_t_() {
  urCommandBufferReleaseExp(CommandBuffer);
  if (Kernel)
    urKernelRelease(Kernel);
}

/// Helper function for calculating work dimensions for kernels
ur_result_t calculateKernelWorkDimensions(
    ur_kernel_handle_t Kernel, ur_device_handle_t Device,
    ze_group_count_t &ZeThreadGroupDimensions, uint32_t (&WG)[3],
    uint32_t WorkDim, const size_t *GlobalWorkSize,
    const size_t *LocalWorkSize) {

  UR_ASSERT(GlobalWorkSize, UR_RESULT_ERROR_INVALID_VALUE);
  // If LocalWorkSize is not provided then Kernel must be provided to query
  // suggested group size.
  UR_ASSERT(LocalWorkSize || Kernel, UR_RESULT_ERROR_INVALID_VALUE);

  // New variable needed because GlobalWorkSize parameter might not be of size 3
  size_t GlobalWorkSize3D[3]{1, 1, 1};
  std::copy(GlobalWorkSize, GlobalWorkSize + WorkDim, GlobalWorkSize3D);

  if (LocalWorkSize) {
    WG[0] = ur_cast<uint32_t>(LocalWorkSize[0]);
    WG[1] = WorkDim >= 2 ? ur_cast<uint32_t>(LocalWorkSize[1]) : 1;
    WG[2] = WorkDim == 3 ? ur_cast<uint32_t>(LocalWorkSize[2]) : 1;
  } else {
    // We can't call to zeKernelSuggestGroupSize if 64-bit GlobalWorkSize3D
    // values do not fit to 32-bit that the API only supports currently.
    bool SuggestGroupSize = true;
    for (int I : {0, 1, 2}) {
      if (GlobalWorkSize3D[I] > UINT32_MAX) {
        SuggestGroupSize = false;
      }
    }
    if (SuggestGroupSize) {
      ZE2UR_CALL(zeKernelSuggestGroupSize,
                 (Kernel->ZeKernel, GlobalWorkSize3D[0], GlobalWorkSize3D[1],
                  GlobalWorkSize3D[2], &WG[0], &WG[1], &WG[2]));
    } else {
      for (int I : {0, 1, 2}) {
        // Try to find a I-dimension WG size that the GlobalWorkSize3D[I] is
        // fully divisable with. Start with the max possible size in
        // each dimension.
        uint32_t GroupSize[] = {
            Device->ZeDeviceComputeProperties->maxGroupSizeX,
            Device->ZeDeviceComputeProperties->maxGroupSizeY,
            Device->ZeDeviceComputeProperties->maxGroupSizeZ};
        GroupSize[I] = (std::min)(size_t(GroupSize[I]), GlobalWorkSize3D[I]);
        while (GlobalWorkSize3D[I] % GroupSize[I]) {
          --GroupSize[I];
        }
        if (GlobalWorkSize3D[I] / GroupSize[I] > UINT32_MAX) {
          urPrint("calculateKernelWorkDimensions: can't find a WG size "
                  "suitable for global work size > UINT32_MAX\n");
          return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
        }
        WG[I] = GroupSize[I];
      }
      urPrint("calculateKernelWorkDimensions: using computed WG size = {%d, "
              "%d, %d}\n",
              WG[0], WG[1], WG[2]);
    }
  }

  // TODO: assert if sizes do not fit into 32-bit?
  switch (WorkDim) {
  case 3:
    ZeThreadGroupDimensions.groupCountX =
        ur_cast<uint32_t>(GlobalWorkSize3D[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        ur_cast<uint32_t>(GlobalWorkSize3D[1] / WG[1]);
    ZeThreadGroupDimensions.groupCountZ =
        ur_cast<uint32_t>(GlobalWorkSize3D[2] / WG[2]);
    break;
  case 2:
    ZeThreadGroupDimensions.groupCountX =
        ur_cast<uint32_t>(GlobalWorkSize3D[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        ur_cast<uint32_t>(GlobalWorkSize3D[1] / WG[1]);
    WG[2] = 1;
    break;
  case 1:
    ZeThreadGroupDimensions.groupCountX =
        ur_cast<uint32_t>(GlobalWorkSize3D[0] / WG[0]);
    WG[1] = WG[2] = 1;
    break;

  default:
    urPrint("calculateKernelWorkDimensions: unsupported work_dim\n");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  // Error handling for non-uniform group size case
  if (GlobalWorkSize3D[0] !=
      size_t(ZeThreadGroupDimensions.groupCountX) * WG[0]) {
    urPrint("calculateKernelWorkDimensions: invalid work_dim. The range "
            "is not a "
            "multiple of the group size in the 1st dimension\n");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize3D[1] !=
      size_t(ZeThreadGroupDimensions.groupCountY) * WG[1]) {
    urPrint("calculateKernelWorkDimensions: invalid work_dim. The range "
            "is not a "
            "multiple of the group size in the 2nd dimension\n");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize3D[2] !=
      size_t(ZeThreadGroupDimensions.groupCountZ) * WG[2]) {
    urPrint("calculateKernelWorkDimensions: invalid work_dim. The range "
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
  if (!SyncPointWaitList || NumSyncPointsInWaitList == 0)
    return UR_RESULT_SUCCESS;

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

  ze_command_list_handle_t ZeCommandListResetEvents;
  // Create a command-list for reseting the events associated to enqueued cmd.
  ZE2UR_CALL(zeCommandListCreate,
             (Context->ZeContext, Device->ZeDevice, &ZeCommandListDesc,
              &ZeCommandListResetEvents));

  // Dependencies between commands are explicitly enforced by sync points when
  // enqueuing. Consequently, relax the command ordering in the command list
  // can enable the backend to further optimize the workload
  ZeCommandListDesc.flags = ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING;

  ZeStruct<ze_mutable_command_list_exp_desc_t> ZeMutableCommandListDesc;
  if (CommandBufferDesc && CommandBufferDesc->isUpdatable) {
    ZeMutableCommandListDesc.flags = 0;
    ZeCommandListDesc.pNext = &ZeMutableCommandListDesc;
  }

  ze_command_list_handle_t ZeCommandList;
  // TODO We could optimize this by pooling both Level Zero command-lists and UR
  // command-buffers, then reusing them.
  ZE2UR_CALL(zeCommandListCreate, (Context->ZeContext, Device->ZeDevice,
                                   &ZeCommandListDesc, &ZeCommandList));
  try {
    *CommandBuffer = new ur_exp_command_buffer_handle_t_(
        Context, Device, ZeCommandList, ZeCommandListResetEvents,
        ZeCommandListDesc, CommandBufferDesc);
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
  UR_CALL(EventCreate(Context, nullptr, false, false,
                      &RetCommandBuffer->AllResetEvent));

  // Add prefix commands
  ZE2UR_CALL(
      zeCommandListAppendEventReset,
      (ZeCommandListResetEvents, RetCommandBuffer->SignalEvent->ZeEvent));
  std::vector<ze_event_handle_t> PrecondEvents = {
      RetCommandBuffer->WaitEvent->ZeEvent,
      RetCommandBuffer->AllResetEvent->ZeEvent};
  ZE2UR_CALL(
      zeCommandListAppendBarrier,
      (ZeCommandList, nullptr, PrecondEvents.size(), PrecondEvents.data()));
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
  UR_ASSERT(CommandBuffer, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  // It is not allowed to append to command list from multiple threads.
  std::scoped_lock<ur_shared_mutex> Guard(CommandBuffer->Mutex);

  // Create a list of events for our signal event to wait on
  // This loop also resets the L0 events we use for command-buffer internal
  // sync-points to the non-signaled state.
  // This is required for multiple submissions.
  const size_t NumEvents = CommandBuffer->SyncPoints.size();
  for (size_t i = 0; i < NumEvents; i++) {
    auto ZeEvent = CommandBuffer->SyncPoints[i]->ZeEvent;
    CommandBuffer->ZeEventsList.push_back(ZeEvent);
    ZE2UR_CALL(zeCommandListAppendEventReset,
               (CommandBuffer->ZeCommandListResetEvents, ZeEvent));
  }
  ZE2UR_CALL(zeCommandListAppendSignalEvent,
             (CommandBuffer->ZeCommandListResetEvents,
              CommandBuffer->AllResetEvent->ZeEvent));

  // Wait for all the user added commands to complete, and signal the
  // command-buffer signal-event when they are done.
  ZE2UR_CALL(zeCommandListAppendBarrier,
             (CommandBuffer->ZeCommandList, CommandBuffer->SignalEvent->ZeEvent,
              NumEvents, CommandBuffer->ZeEventsList.data()));

  // Close the command lists and have them ready for dispatch.
  ZE2UR_CALL(zeCommandListClose, (CommandBuffer->ZeCommandList));
  ZE2UR_CALL(zeCommandListClose, (CommandBuffer->ZeCommandListResetEvents));
  CommandBuffer->IsFinalized = true;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_kernel_handle_t Kernel,
    uint32_t WorkDim, const size_t *GlobalWorkOffset,
    const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
    uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint,
    ur_exp_command_buffer_command_handle_t *Command) {
  UR_ASSERT(CommandBuffer && Kernel && Kernel->Program,
            UR_RESULT_ERROR_INVALID_NULL_POINTER);
  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex, ur_shared_mutex> Lock(
      Kernel->Mutex, Kernel->Program->Mutex, CommandBuffer->Mutex);

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

  if (SyncPoint) {
    // Get sync point and register the event with it.
    *SyncPoint = CommandBuffer->GetNextSyncPoint();
    CommandBuffer->RegisterSyncPoint(*SyncPoint, LaunchEvent);
  }

  LaunchEvent->CommandData = (void *)Kernel;
  // Increment the reference count of the Kernel and indicate that the Kernel
  // is in use. Once the event has been signaled, the code in
  // CleanupCompletedEvent(Event) will do a urKernelRelease to update the
  // reference count on the kernel, using the kernel saved in CommandData.
  UR_CALL(urKernelRetain(Kernel));

  // If command-buffer is updatable then get command id which is going to be
  // used if command is updated in the future. This
  // zeCommandListGetNextCommandIdExp can be called only if command is
  // updatable.
  uint64_t CommandId = 0;
  if (CommandBuffer->IsUpdatable) {
    ZeStruct<ze_mutable_command_id_exp_desc_t> ZeMutableCommandDesc;
    ZeMutableCommandDesc.flags = ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS |
                                 ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT |
                                 ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE |
                                 ZE_MUTABLE_COMMAND_EXP_FLAG_GLOBAL_OFFSET;

    auto Plt = CommandBuffer->Context->getPlatform();
    UR_ASSERT(Plt->ZeMutableCmdListExt.Supported,
              UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
    ZE2UR_CALL(
        Plt->ZeMutableCmdListExt.zexCommandListGetNextCommandIdExp,
        (CommandBuffer->ZeCommandList, &ZeMutableCommandDesc, &CommandId));
  }
  try {
    if (Command)
      *Command = new ur_exp_command_buffer_command_handle_t_(CommandBuffer,
                                                             CommandId, Kernel);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

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
      // Create command-list to execute before `CommandListPtr` and will signal
      // when `EventWaitList` dependencies are complete.
      ur_command_list_ptr_t WaitCommandList{};
      UR_CALL(Queue->Context->getAvailableCommandList(Queue, WaitCommandList,
                                                      false, false));

      ZE2UR_CALL(zeCommandListAppendBarrier,
                 (WaitCommandList->first, CommandBuffer->WaitEvent->ZeEvent,
                  CommandBuffer->WaitEvent->WaitList.Length,
                  CommandBuffer->WaitEvent->WaitList.ZeEventList));
      Queue->executeCommandList(WaitCommandList, false, false);
      MustSignalWaitEvent = false;
    }
  }
  if (MustSignalWaitEvent) {
    ZE2UR_CALL(zeEventHostSignal, (CommandBuffer->WaitEvent->ZeEvent));
  }

  // Submit reset events command-list. This command-list is of a batch
  // command-list type, regardless of the UR Queue type. We therefore need to
  // submit the list directly using the Level-Zero API to avoid type mismatches
  // if using UR functions.
  ZE2UR_CALL(
      zeCommandQueueExecuteCommandLists,
      (ZeCommandQueue, 1, &CommandBuffer->ZeCommandListResetEvents, nullptr));

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
  // Reset the all-reset-event for the UR command-buffer that is signaled when
  // all events of the main command-list have been reset.
  ZE2UR_CALL(zeCommandListAppendEventReset,
             (SignalCommandList->first, CommandBuffer->AllResetEvent->ZeEvent));

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

      Profiling->NumEvents = CommandBuffer->ZeEventsList.size();
      Profiling->Timestamps =
          new ze_kernel_timestamp_result_t[Profiling->NumEvents];

      ZE2UR_CALL(zeCommandListAppendQueryKernelTimestamps,
                 (SignalCommandList->first, CommandBuffer->ZeEventsList.size(),
                  CommandBuffer->ZeEventsList.data(),
                  (void *)Profiling->Timestamps, 0, RetEvent->ZeEvent, 1,
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

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferRetainCommandExp(
    ur_exp_command_buffer_command_handle_t Command) {
  Command->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferReleaseCommandExp(
    ur_exp_command_buffer_command_handle_t Command) {
  if (!Command->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  delete Command;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferUpdateKernelLaunchExp(
    ur_exp_command_buffer_command_handle_t Command,
    const ur_exp_command_buffer_update_kernel_launch_desc_t *CommandDesc) {
  UR_ASSERT(Command, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(Command->Kernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(CommandDesc, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(CommandDesc->newWorkDim <= 3,
            UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  // Lock command, kernel and command buffer for update.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex, ur_shared_mutex> Guard(
      Command->Mutex, Command->CommandBuffer->Mutex, Command->Kernel->Mutex);
  UR_ASSERT(Command->CommandBuffer->IsUpdatable,
            UR_RESULT_ERROR_INVALID_OPERATION);
  UR_ASSERT(Command->CommandBuffer->IsFinalized,
            UR_RESULT_ERROR_INVALID_OPERATION);

  auto CommandBuffer = Command->CommandBuffer;
  uint32_t Dim = CommandDesc->newWorkDim;
  const void *NextDesc = nullptr;
  auto SupportedFeatures =
      Command->CommandBuffer->Device->ZeDeviceMutableCmdListsProperties
          ->mutableCommandFlags;

  // We need the created descriptors to live till the point when
  // zexCommandListUpdateMutableCommandsExp is called at the end of the
  // function.
  std::vector<std::unique_ptr<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>>>
      ArgDescs;
  std::vector<std::unique_ptr<ZeStruct<ze_mutable_global_offset_exp_desc_t>>>
      OffsetDescs;
  std::vector<std::unique_ptr<ZeStruct<ze_mutable_group_size_exp_desc_t>>>
      GroupSizeDescs;
  std::vector<std::unique_ptr<ZeStruct<ze_mutable_group_count_exp_desc_t>>>
      GroupCountDescs;

  // Check if new global offset is provided.
  size_t *NewGlobalWorkOffset = CommandDesc->pNewGlobalWorkOffset;
  UR_ASSERT(!NewGlobalWorkOffset ||
                (SupportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_GLOBAL_OFFSET),
            UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
  if (NewGlobalWorkOffset && Dim > 0) {
    if (!CommandBuffer->Context->getPlatform()
             ->ZeDriverGlobalOffsetExtensionFound) {
      urPrint("No global offset extension found on this driver\n");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    auto MutableGroupOffestDesc =
        std::make_unique<ZeStruct<ze_mutable_global_offset_exp_desc_t>>();
    MutableGroupOffestDesc->commandId = Command->CommandId;
    MutableGroupOffestDesc->pNext = NextDesc;
    MutableGroupOffestDesc->offsetX = NewGlobalWorkOffset[0];
    MutableGroupOffestDesc->offsetY = Dim >= 2 ? NewGlobalWorkOffset[1] : 0;
    MutableGroupOffestDesc->offsetZ = Dim == 3 ? NewGlobalWorkOffset[2] : 0;
    NextDesc = MutableGroupOffestDesc.get();
    OffsetDescs.push_back(std::move(MutableGroupOffestDesc));
  }

  // Check if new group size is provided.
  size_t *NewLocalWorkSize = CommandDesc->pNewLocalWorkSize;
  UR_ASSERT(!NewLocalWorkSize ||
                (SupportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE),
            UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
  if (NewLocalWorkSize && Dim > 0) {
    auto MutableGroupSizeDesc =
        std::make_unique<ZeStruct<ze_mutable_group_size_exp_desc_t>>();
    MutableGroupSizeDesc->commandId = Command->CommandId;
    MutableGroupSizeDesc->pNext = NextDesc;
    MutableGroupSizeDesc->groupSizeX = NewLocalWorkSize[0];
    MutableGroupSizeDesc->groupSizeY = Dim >= 2 ? NewLocalWorkSize[1] : 1;
    MutableGroupSizeDesc->groupSizeZ = Dim == 3 ? NewLocalWorkSize[2] : 1;
    NextDesc = MutableGroupSizeDesc.get();
    GroupSizeDescs.push_back(std::move(MutableGroupSizeDesc));
  }

  // Check if new global size is provided and we need to update group count.
  size_t *NewGlobalWorkSize = CommandDesc->pNewGlobalWorkSize;
  UR_ASSERT(!NewGlobalWorkSize ||
                (SupportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT),
            UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
  UR_ASSERT(!(NewGlobalWorkSize && !NewLocalWorkSize) ||
                (SupportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE),
            UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
  if (NewGlobalWorkSize && Dim > 0) {
    ze_group_count_t ZeThreadGroupDimensions{1, 1, 1};
    uint32_t WG[3];
    // If new global work size is provided but new local work size is not
    // provided then we still need to update local work size based on size
    // suggested by the driver for the kernel.
    bool UpdateWGSize = NewLocalWorkSize == nullptr;
    UR_CALL(calculateKernelWorkDimensions(
        Command->Kernel, CommandBuffer->Device, ZeThreadGroupDimensions, WG,
        Dim, NewGlobalWorkSize, NewLocalWorkSize));
    auto MutableGroupCountDesc =
        std::make_unique<ZeStruct<ze_mutable_group_count_exp_desc_t>>();
    MutableGroupCountDesc->pNext = NextDesc;
    MutableGroupCountDesc->commandId = Command->CommandId;
    MutableGroupCountDesc->pGroupCount = &ZeThreadGroupDimensions;
    NextDesc = MutableGroupCountDesc.get();
    GroupCountDescs.push_back(std::move(MutableGroupCountDesc));

    if (UpdateWGSize) {
      auto MutableGroupSizeDesc =
          std::make_unique<ZeStruct<ze_mutable_group_size_exp_desc_t>>();
      MutableGroupSizeDesc->commandId = Command->CommandId;
      MutableGroupSizeDesc->pNext = NextDesc;
      MutableGroupSizeDesc->groupSizeX = WG[0];
      MutableGroupSizeDesc->groupSizeY = WG[1];
      MutableGroupSizeDesc->groupSizeZ = WG[2];
      NextDesc = MutableGroupSizeDesc.get();
      GroupSizeDescs.push_back(std::move(MutableGroupSizeDesc));
    }
  }

  UR_ASSERT(
      (!CommandDesc->numNewMemObjArgs && !CommandDesc->numNewPointerArgs &&
       !CommandDesc->numNewValueArgs) ||
          (SupportedFeatures & ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS),
      UR_RESULT_ERROR_UNSUPPORTED_FEATURE);

  // Check if new memory object arguments are provided.
  for (uint32_t NewMemObjArgNum = CommandDesc->numNewMemObjArgs;
       NewMemObjArgNum-- > 0;) {
    ur_exp_command_buffer_update_memobj_arg_desc_t NewMemObjArgDesc =
        CommandDesc->pNewMemObjArgList[NewMemObjArgNum];
    const ur_kernel_arg_mem_obj_properties_t *Properties =
        NewMemObjArgDesc.pProperties;
    ur_mem_handle_t_::access_mode_t UrAccessMode = ur_mem_handle_t_::read_write;
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
    ur_mem_handle_t NewMemObjArg = NewMemObjArgDesc.hNewMemObjArg;
    // The NewMemObjArg may be a NULL pointer in which case a NULL value is used
    // for the kernel argument declared as a pointer to global or constant
    // memory.
    char **ZeHandlePtr = nullptr;
    if (NewMemObjArg) {
      UR_CALL(NewMemObjArg->getZeHandlePtr(ZeHandlePtr, UrAccessMode,
                                           CommandBuffer->Device));
    }
    auto ZeMutableArgDesc =
        std::make_unique<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>>();
    ZeMutableArgDesc->commandId = Command->CommandId;
    ZeMutableArgDesc->pNext = NextDesc;
    ZeMutableArgDesc->argIndex = NewMemObjArgDesc.argIndex;
    ZeMutableArgDesc->argSize = sizeof(void *);
    ZeMutableArgDesc->pArgValue = ZeHandlePtr;

    NextDesc = ZeMutableArgDesc.get();
    ArgDescs.push_back(std::move(ZeMutableArgDesc));
  }

  // Check if there are new pointer arguments.
  for (uint32_t NewPointerArgNum = CommandDesc->numNewPointerArgs;
       NewPointerArgNum-- > 0;) {
    ur_exp_command_buffer_update_pointer_arg_desc_t NewPointerArgDesc =
        CommandDesc->pNewPointerArgList[NewPointerArgNum];
    auto ZeMutableArgDesc =
        std::make_unique<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>>();
    ZeMutableArgDesc->commandId = Command->CommandId;
    ZeMutableArgDesc->pNext = NextDesc;
    ZeMutableArgDesc->argIndex = NewPointerArgDesc.argIndex;
    ZeMutableArgDesc->argSize = sizeof(void *);
    ZeMutableArgDesc->pArgValue = NewPointerArgDesc.pNewPointerArg;

    NextDesc = ZeMutableArgDesc.get();
    ArgDescs.push_back(std::move(ZeMutableArgDesc));
  }

  // Check if there are new value arguments.
  for (uint32_t NewValueArgNum = CommandDesc->numNewValueArgs;
       NewValueArgNum-- > 0;) {
    ur_exp_command_buffer_update_value_arg_desc_t NewValueArgDesc =
        CommandDesc->pNewValueArgList[NewValueArgNum];
    auto ZeMutableArgDesc =
        std::make_unique<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>>();
    ZeMutableArgDesc->commandId = Command->CommandId;
    ZeMutableArgDesc->pNext = NextDesc;
    ZeMutableArgDesc->argIndex = NewValueArgDesc.argIndex;
    ZeMutableArgDesc->argSize = NewValueArgDesc.argSize;
    // OpenCL: "the arg_value pointer can be NULL or point to a NULL value
    // in which case a NULL value will be used as the value for the argument
    // declared as a pointer to global or constant memory in the kernel"
    //
    // We don't know the type of the argument but it seems that the only time
    // SYCL RT would send a pointer to NULL in 'arg_value' is when the argument
    // is a NULL pointer. Treat a pointer to NULL in 'arg_value' as a NULL.
    const void *ArgValuePtr = NewValueArgDesc.pNewValueArg;
    if (NewValueArgDesc.argSize == sizeof(void *) && ArgValuePtr &&
        *(void **)(const_cast<void *>(ArgValuePtr)) == nullptr) {
      ArgValuePtr = nullptr;
    }
    ZeMutableArgDesc->pArgValue = ArgValuePtr;
    NextDesc = ZeMutableArgDesc.get();
    ArgDescs.push_back(std::move(ZeMutableArgDesc));
  }

  // Check if there are new exec info flags provided.
  for (uint32_t NewExecInfoNum = CommandDesc->numNewExecInfos;
       NewExecInfoNum-- > 0;) {
    ur_exp_command_buffer_update_exec_info_desc_t NewExecInfoDesc =
        CommandDesc->pNewExecInfoList[NewExecInfoNum];
    ur_kernel_exec_info_t PropName = NewExecInfoDesc.propName;
    const void *PropValue = NewExecInfoDesc.pNewExecInfo;
    if (PropName == UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS) {
      // The whole point for users really was to not need to know anything
      // about the types of allocations kernel uses. So in DPC++ we always
      // just set all 3 modes for each kernel.
      if (*(static_cast<const ur_bool_t *>(PropValue)) == true) {
        ze_kernel_indirect_access_flags_t IndirectFlags =
            ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST |
            ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE |
            ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
        ZE2UR_CALL(zeKernelSetIndirectAccess,
                   (Command->Kernel->ZeKernel, IndirectFlags));
      }
    } else if (PropName == UR_KERNEL_EXEC_INFO_CACHE_CONFIG) {
      ze_cache_config_flag_t ZeCacheConfig{};
      auto CacheConfig =
          *(static_cast<const ur_kernel_cache_config_t *>(PropValue));
      switch (CacheConfig) {
      case UR_KERNEL_CACHE_CONFIG_LARGE_SLM:
        ZeCacheConfig = ZE_CACHE_CONFIG_FLAG_LARGE_SLM;
        break;
      case UR_KERNEL_CACHE_CONFIG_LARGE_DATA:
        ZeCacheConfig = ZE_CACHE_CONFIG_FLAG_LARGE_DATA;
        break;
      case UR_KERNEL_CACHE_CONFIG_DEFAULT:
        ZeCacheConfig = static_cast<ze_cache_config_flag_t>(0);
        break;
      default:
        // Unexpected cache configuration value.
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
      ZE2UR_CALL(zeKernelSetCacheConfig,
                 (Command->Kernel->ZeKernel, ZeCacheConfig););
    } else if (PropName == UR_KERNEL_EXEC_INFO_USM_PTRS) {
      // Ignore this property as such kernel property is not supported by Level
      // Zero.
      continue;
    } else {
      urPrint("urCommandBufferUpdateKernelLaunchExp: unsupported name of "
              "execution attribute.\n");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  }

  ZeStruct<ze_mutable_commands_exp_desc_t> MutableCommandDesc;
  MutableCommandDesc.pNext = NextDesc;
  MutableCommandDesc.flags = 0;

  // We must synchronize mutable command list execution before mutating.
  ZE2UR_CALL(zeEventHostSynchronize,
             (CommandBuffer->SignalEvent->ZeEvent, UINT64_MAX));

  auto Plt = Command->CommandBuffer->Context->getPlatform();
  UR_ASSERT(Plt->ZeMutableCmdListExt.Supported,
            UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
  ZE2UR_CALL(Plt->ZeMutableCmdListExt.zexCommandListUpdateMutableCommandsExp,
             (CommandBuffer->ZeCommandList, &MutableCommandDesc));
  ZE2UR_CALL(zeCommandListClose, (CommandBuffer->ZeCommandList));

  return UR_RESULT_SUCCESS;
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
    ur_exp_command_buffer_command_handle_t Command,
    ur_exp_command_buffer_command_info_t PropName, size_t PropSize,
    void *PropValue, size_t *PropSizeRet) {
  UrReturnHelper ReturnValue(PropSize, PropValue, PropSizeRet);

  switch (PropName) {
  case UR_EXP_COMMAND_BUFFER_COMMAND_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{Command->RefCount.load()});
  default:
    assert(!"Command-buffer command info request not implemented");
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}
