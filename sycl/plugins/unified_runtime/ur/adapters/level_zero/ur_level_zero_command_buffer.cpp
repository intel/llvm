//===--------- ur_level_zero_command_buffer.cpp - Level Zero Adapter
//---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur_level_zero_command_buffer.hpp"
#include "ur_level_zero.hpp"

/* Command-buffer Extension

   The UR interface for submitting a UR command-buffer takes a list
   of events to wait on, and an event representing the completion of
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

  ┌──────────┬────────────────────────────────────────────────┬─────────┐
  │  Prefix  │ Commands added to UR command-buffer by UR user │ Suffix  │
  └──────────┴────────────────────────────────────────────────┴─────────┘

            ┌───────────────────┬──────────────────────────────┐
  Prefix    │Reset signal event │ Barrier waiting on wait event│
            └───────────────────┴──────────────────────────────┘

            ┌─────────────────────────────────────────┐
  Suffix    │Signal the UR command-buffer signal event│
            └─────────────────────────────────────────┘


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
*/

ur_exp_command_buffer_handle_t_::ur_exp_command_buffer_handle_t_(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ze_command_list_handle_t CommandList,
    ZeStruct<ze_command_list_desc_t> ZeDesc,
    const ur_exp_command_buffer_desc_t *Desc)
    : Context(hContext), Device(hDevice), ZeCommandList(CommandList),
      ZeCommandListDesc(ZeDesc), QueueProperties(), SyncPoints(),
      NextSyncPoint(0), CommandListMap() {
  urContextRetain(hContext);
  urDeviceRetain(hDevice);
  // TODO: Do we actually need the queue properties? Removed from the UR feature
  // for now.
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
  for (auto it = CommandListMap.begin(); it != CommandListMap.end(); ++it) {
    if (it->second.ZeFence != nullptr) {
      ZE_CALL_NOCHECK(zeFenceDestroy, (it->second.ZeFence));
    }
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
        GroupSize[I] = std::min(size_t(GroupSize[I]), GlobalWorkSize[I]);
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

/// Helper function to take a list of ur_exp_command_buffer_sync_point_ts and
/// fill the provided vector with the associated ZeEvents
static ur_result_t getEventsFromSyncPoints(
    const std::unordered_map<ur_exp_command_buffer_sync_point_t,
                             ur_event_handle_t> &SyncPoints,
    size_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    std::vector<ze_event_handle_t> &ZeEventList) {
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

// Helper function for common code when enqueuing memory operations to a command
// buffer.
static ur_result_t enqueueCommandBufferMemCopyHelper(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *Dst, const void *Src,
    size_t Size, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {
  std::vector<ze_event_handle_t> ZeEventList;
  ur_result_t Res = getEventsFromSyncPoints(hCommandBuffer->SyncPoints,
                                            NumSyncPointsInWaitList,
                                            SyncPointWaitList, ZeEventList);

  ur_event_handle_t LaunchEvent;
  Res = EventCreate(hCommandBuffer->Context, nullptr, true, &LaunchEvent);
  LaunchEvent->CommandType = UR_COMMAND_MEM_BUFFER_COPY;
  if (Res)
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (hCommandBuffer->ZeCommandList, Dst, Src, Size,
              LaunchEvent->ZeEvent, ZeEventList.size(), ZeEventList.data()));

  urPrint("calling zeCommandListAppendMemoryCopy() with"
          "  ZeEvent %#lx\n",
          ur_cast<std::uintptr_t>(LaunchEvent->ZeEvent));

  // Get sync point and register the event with it.
  *SyncPoint = hCommandBuffer->GetNextSyncPoint();
  hCommandBuffer->RegisterSyncPoint(*SyncPoint, LaunchEvent);
  return UR_RESULT_SUCCESS;
}

// Helper function for common code when enqueuing rectangular memory operations
// to a command buffer.
static ur_result_t enqueueCommandBufferMemCopyRectHelper(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *Dst, const void *Src,
    ur_rect_offset_t SrcOrigin, ur_rect_offset_t DstOrigin,
    ur_rect_region_t Region, size_t SrcRowPitch, size_t DstRowPitch,
    size_t SrcSlicePitch, size_t DstSlicePitch,
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
  ur_result_t Res = getEventsFromSyncPoints(hCommandBuffer->SyncPoints,
                                            NumSyncPointsInWaitList,
                                            SyncPointWaitList, ZeEventList);

  ur_event_handle_t LaunchEvent;
  Res = EventCreate(hCommandBuffer->Context, nullptr, true, &LaunchEvent);
  LaunchEvent->CommandType = UR_COMMAND_MEM_BUFFER_COPY_RECT;

  if (Res)
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;

  ZE2UR_CALL(zeCommandListAppendMemoryCopyRegion,
             (hCommandBuffer->ZeCommandList, Dst, &ZeDstRegion, DstPitch,
              DstSlicePitch, Src, &ZeSrcRegion, SrcPitch, SrcSlicePitch,
              LaunchEvent->ZeEvent, ZeEventList.size(), ZeEventList.data()));

  urPrint("calling zeCommandListAppendMemoryCopyRegion() with"
          "  ZeEvent %#lx\n",
          ur_cast<std::uintptr_t>(LaunchEvent->ZeEvent));

  // Get sync point and register the event with it.
  *SyncPoint = hCommandBuffer->GetNextSyncPoint();
  hCommandBuffer->RegisterSyncPoint(*SyncPoint, LaunchEvent);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_exp_command_buffer_desc_t *pCommandBufferDesc,
    ur_exp_command_buffer_handle_t *phCommandBuffer) {
  // Force compute queue type for now. Copy engine types may be better suited
  // for host to device copies.
  uint32_t QueueGroupOrdinal =
      hDevice
          ->QueueGroup[ur_device_handle_t_::queue_group_info_t::type::Compute]
          .ZeOrdinal;

  ZeStruct<ze_command_list_desc_t> ZeCommandListDesc;
  ZeCommandListDesc.commandQueueGroupOrdinal = QueueGroupOrdinal;

  ze_command_list_handle_t ZeCommandList;
  ZE2UR_CALL(zeCommandListCreate, (hContext->ZeContext, hDevice->ZeDevice,
                                   &ZeCommandListDesc, &ZeCommandList));
  try {
    *phCommandBuffer = new ur_exp_command_buffer_handle_t_(
        hContext, hDevice, ZeCommandList, ZeCommandListDesc,
        pCommandBufferDesc);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  // Create signal & wait events to be used in the command-list for sync
  // on command-buffer enqueue.
  auto hCommandBuffer = *phCommandBuffer;
  UR_CALL(EventCreate(hContext, nullptr, true, &hCommandBuffer->SignalEvent));
  UR_CALL(EventCreate(hContext, nullptr, false, &hCommandBuffer->WaitEvent));

  // Add prefix commands
  ZE2UR_CALL(zeCommandListAppendEventReset,
             (ZeCommandList, hCommandBuffer->SignalEvent->ZeEvent));
  ZE2UR_CALL(zeCommandListAppendBarrier,
             (ZeCommandList, nullptr, 1, &hCommandBuffer->WaitEvent->ZeEvent));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferRetainExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  UR_ASSERT(hCommandBuffer, UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP);
  hCommandBuffer->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferReleaseExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  UR_ASSERT(hCommandBuffer, UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP);
  if (!hCommandBuffer->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  delete hCommandBuffer;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferFinalizeExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  UR_ASSERT(hCommandBuffer, UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP);
  // We need to append signal that will indicate that command-buffer has
  // finished executing.
  ZE2UR_CALL(
      zeCommandListAppendSignalEvent,
      (hCommandBuffer->ZeCommandList, hCommandBuffer->SignalEvent->ZeEvent));
  // Close the command list and have it ready for dispatch.
  ZE2UR_CALL(zeCommandListClose, (hCommandBuffer->ZeCommandList));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_kernel_handle_t hKernel,
    uint32_t workDim, const size_t *pGlobalWorkOffset,
    const size_t *pGlobalWorkSize, const size_t *pLocalWorkSize,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  UR_ASSERT(hCommandBuffer, UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP);
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_KERNEL);
  UR_ASSERT((workDim > 0) && (workDim < 4),
            UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(
      hKernel->Mutex, hKernel->Program->Mutex);
  if (pGlobalWorkOffset != NULL) {
    if (!hCommandBuffer->Context->getPlatform()
             ->ZeDriverGlobalOffsetExtensionFound) {
      urPrint("No global offset extension found on this driver\n");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    ZE2UR_CALL(zeKernelSetGlobalOffsetExp,
               (hKernel->ZeKernel, pGlobalWorkOffset[0], pGlobalWorkOffset[1],
                pGlobalWorkOffset[2]));
  }

  // If there are any pending arguments set them now.
  for (auto &Arg : hKernel->PendingArguments) {
    // The ArgValue may be a NULL pointer in which case a NULL value is used for
    // the kernel argument declared as a pointer to global or constant memory.
    char **ZeHandlePtr = nullptr;
    if (Arg.Value) {
      // TODO: Not sure of the implication of not passing a device pointer here
      UR_CALL(Arg.Value->getZeHandlePtr(ZeHandlePtr, Arg.AccessMode));
    }
    ZE2UR_CALL(zeKernelSetArgumentValue,
               (hKernel->ZeKernel, Arg.Index, Arg.Size, ZeHandlePtr));
  }
  hKernel->PendingArguments.clear();

  ze_group_count_t ZeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3];

  auto result = calculateKernelWorkDimensions(
      hKernel, hCommandBuffer->Device, ZeThreadGroupDimensions, WG, workDim,
      pGlobalWorkSize, pLocalWorkSize);
  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  ZE2UR_CALL(zeKernelSetGroupSize, (hKernel->ZeKernel, WG[0], WG[1], WG[2]));

  std::vector<ze_event_handle_t> ZeEventList;
  ur_result_t Res = getEventsFromSyncPoints(hCommandBuffer->SyncPoints,
                                            numSyncPointsInWaitList,
                                            pSyncPointWaitList, ZeEventList);
  if (Res) {
    return Res;
  }
  ur_event_handle_t LaunchEvent;
  Res = EventCreate(hCommandBuffer->Context, nullptr, true, &LaunchEvent);
  LaunchEvent->CommandType = UR_COMMAND_KERNEL_LAUNCH;
  if (Res)
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;

  LaunchEvent->CommandData = (void *)hKernel;
  // Increment the reference count of the hKernel and indicate that the hKernel
  // is in use. Once the event has been signalled, the code in
  // CleanupCompletedEvent(Event) will do a piReleaseKernel to update the
  // reference count on the kernel, using the kernel saved in CommandData.
  UR_CALL(urKernelRetain(hKernel));

  ZE2UR_CALL(zeCommandListAppendLaunchKernel,
             (hCommandBuffer->ZeCommandList, hKernel->ZeKernel,
              &ZeThreadGroupDimensions, LaunchEvent->ZeEvent,
              ZeEventList.size(), ZeEventList.data()));

  urPrint("calling zeCommandListAppendLaunchKernel() with"
          "  ZeEvent %#lx\n",
          ur_cast<std::uintptr_t>(LaunchEvent->ZeEvent));

  // Get sync point and register the event with it.
  *pSyncPoint = hCommandBuffer->GetNextSyncPoint();
  hCommandBuffer->RegisterSyncPoint(*pSyncPoint, LaunchEvent);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemcpyUSMExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pDst, const void *pSrc,
    size_t size, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  UR_ASSERT(hCommandBuffer, UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP);
  if (!pDst) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return enqueueCommandBufferMemCopyHelper(hCommandBuffer, pDst, pSrc, size,
                                           numSyncPointsInWaitList,
                                           pSyncPointWaitList, pSyncPoint);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMembufferCopyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  UR_ASSERT(hCommandBuffer, UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP);
  UR_ASSERT(hSrcMem && hDstMem, UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  auto SrcBuffer = ur_cast<ur_mem_handle_t>(hSrcMem);
  auto DstBuffer = ur_cast<ur_mem_handle_t>(hDstMem);

  std::shared_lock<ur_shared_mutex> SrcLock(SrcBuffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex> LockAll(
      SrcLock, DstBuffer->Mutex);

  char *ZeHandleSrc;
  UR_CALL(SrcBuffer->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                                 hCommandBuffer->Device));
  char *ZeHandleDst;
  UR_CALL(DstBuffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                 hCommandBuffer->Device));

  return enqueueCommandBufferMemCopyHelper(
      hCommandBuffer, ZeHandleDst, ZeHandleSrc, size, numSyncPointsInWaitList,
      pSyncPointWaitList, pSyncPoint);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMembufferCopyRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  UR_ASSERT(hCommandBuffer, UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP);
  UR_ASSERT(hSrcMem && hDstMem, UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  auto SrcBuffer = ur_cast<ur_mem_handle_t>(hSrcMem);
  auto DstBuffer = ur_cast<ur_mem_handle_t>(hDstMem);

  std::shared_lock<ur_shared_mutex> SrcLock(SrcBuffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex> LockAll(
      SrcLock, DstBuffer->Mutex);

  char *ZeHandleSrc;
  UR_CALL(SrcBuffer->getZeHandle(ZeHandleSrc, ur_mem_handle_t_::read_only,
                                 hCommandBuffer->Device));
  char *ZeHandleDst;
  UR_CALL(DstBuffer->getZeHandle(ZeHandleDst, ur_mem_handle_t_::write_only,
                                 hCommandBuffer->Device));

  return enqueueCommandBufferMemCopyRectHelper(
      hCommandBuffer, ZeHandleDst, ZeHandleSrc, srcOrigin, dstOrigin, region,
      srcRowPitch, dstRowPitch, srcSlicePitch, dstSlicePitch,
      numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_queue_handle_t hQueue,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  UR_ASSERT(hCommandBuffer, UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP);

  std::scoped_lock<ur_shared_mutex> lock(hQueue->Mutex);
  // Use compute engine rather than copy engine
  const auto UseCopyEngine = false;
  auto &QGroup = hQueue->getQueueGroup(UseCopyEngine);
  uint32_t QueueGroupOrdinal;
  auto &ZeCommandQueue = QGroup.getZeQueue(&QueueGroupOrdinal);

  ze_fence_handle_t ZeFence;
  ZeStruct<ze_fence_desc_t> ZeFenceDesc;
  ur_command_list_ptr_t CommandListPtr;

  ZE2UR_CALL(zeFenceCreate, (ZeCommandQueue, &ZeFenceDesc, &ZeFence));
  // TODO: Refactor so requiring a map iterator is not required here, currently
  // required for executeCommandList though.
  ZeStruct<ze_command_queue_desc_t> ZeQueueDesc;
  ZeQueueDesc.ordinal = QueueGroupOrdinal;
  CommandListPtr = hCommandBuffer->CommandListMap.insert(
      std::pair<ze_command_list_handle_t, pi_command_list_info_t>(
          hCommandBuffer->ZeCommandList,
          {ZeFence, false, false, ZeCommandQueue, ZeQueueDesc}));

  // Previous execution will have closed the command list, we need to reopen
  // it otherwise calling `executeCommandList` will return early.
  CommandListPtr->second.IsClosed = false;
  CommandListPtr->second.ZeFenceInUse = true;

  // Create command-list to execute before `CommandListPtr` and will signal
  // when `EventWaitList` dependencies are complete.
  ur_command_list_ptr_t WaitCommandList{};
  if (numEventsInWaitList) {
    _ur_ze_event_list_t TmpWaitList;
    if (auto Res = TmpWaitList.createAndRetainUrZeEventList(
            numEventsInWaitList, phEventWaitList, hQueue, UseCopyEngine))
      return Res;

    if (auto Res = hQueue->Context->getAvailableCommandList(
            hQueue, WaitCommandList, false, false))
      return Res;

    // Update the WaitList of the Wait Event
    // Events are appended to the WaitList if the WaitList is not empty
    if (hCommandBuffer->WaitEvent->WaitList.isEmpty())
      hCommandBuffer->WaitEvent->WaitList = TmpWaitList;
    else
      hCommandBuffer->WaitEvent->WaitList.insert(TmpWaitList);

    ZE2UR_CALL(zeCommandListAppendBarrier,
               (WaitCommandList->first, hCommandBuffer->WaitEvent->ZeEvent,
                hCommandBuffer->WaitEvent->WaitList.Length,
                hCommandBuffer->WaitEvent->WaitList.ZeEventList));
  } else {
    if (auto Res = hQueue->Context->getAvailableCommandList(
            hQueue, WaitCommandList, false, false))
      return Res;

    ZE2UR_CALL(zeCommandListAppendSignalEvent,
               (WaitCommandList->first, hCommandBuffer->WaitEvent->ZeEvent));
  }

  // Execution event for this enqueue of the PI command-buffer
  ur_event_handle_t RetEvent{};
  // Create a command-list to signal RetEvent on completion
  ur_command_list_ptr_t SignalCommandList{};
  if (phEvent) {
    if (auto Res = hQueue->Context->getAvailableCommandList(
            hQueue, SignalCommandList, false, false))
      return Res;

    if (auto Res = createEventAndAssociateQueue(
            hQueue, &RetEvent, UR_COMMAND_COMMAND_BUFFER_ENQUEUE_EXP,
            SignalCommandList, false))
      return Res;

    ZE2UR_CALL(zeCommandListAppendBarrier,
               (SignalCommandList->first, RetEvent->ZeEvent, 1,
                &(hCommandBuffer->SignalEvent->ZeEvent)));
  }

  // Execution our command-lists asynchronously
  if (auto Res = hQueue->executeCommandList(WaitCommandList, false, false))
    return Res;

  if (auto Res = hQueue->executeCommandList(CommandListPtr, false, false))
    return Res;

  if (auto Res = hQueue->executeCommandList(SignalCommandList, false, false))
    return Res;

  if (phEvent) {
    *phEvent = RetEvent;
  }

  return UR_RESULT_SUCCESS;
}
