//===--------- kernel.cpp - Level Zero Adapter ----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel.hpp"
#include "logger/ur_logger.hpp"
#include "ur_api.h"
#include "ur_interface_loader.hpp"

#include "helpers/kernel_helpers.hpp"

ur_result_t getZeKernel(ze_device_handle_t hDevice, ur_kernel_handle_t hKernel,
                        ze_kernel_handle_t *phZeKernel) {
  if (hKernel->ZeKernelMap.empty()) {
    *phZeKernel = hKernel->ZeKernel;
  } else {
    auto It = hKernel->ZeKernelMap.find(hDevice);
    if (It == hKernel->ZeKernelMap.end()) {
      /* kernel and queue don't match */
      return UR_RESULT_ERROR_INVALID_QUEUE;
    }
    *phZeKernel = It->second;
  }

  return UR_RESULT_SUCCESS;
}

namespace ur::level_zero {

ur_result_t urKernelGetSuggestedLocalWorkSize(
    ur_kernel_handle_t hKernel, ur_queue_handle_t hQueue, uint32_t workDim,
    [[maybe_unused]] const size_t *pGlobalWorkOffset,
    const size_t *pGlobalWorkSize, size_t *pSuggestedLocalWorkSize) {
  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(pSuggestedLocalWorkSize != nullptr,
            UR_RESULT_ERROR_INVALID_NULL_POINTER);

  uint32_t LocalWorkSize[3];
  size_t GlobalWorkSize3D[3]{1, 1, 1};
  std::copy(pGlobalWorkSize, pGlobalWorkSize + workDim, GlobalWorkSize3D);

  ze_kernel_handle_t ZeKernel{};
  UR_CALL(getZeKernel(hQueue->Device->ZeDevice, hKernel, &ZeKernel));

  UR_CALL(getSuggestedLocalWorkSize(hQueue->Device, ZeKernel, GlobalWorkSize3D,
                                    LocalWorkSize));

  std::copy(LocalWorkSize, LocalWorkSize + workDim, pSuggestedLocalWorkSize);
  return UR_RESULT_SUCCESS;
}

ur_result_t urEnqueueKernelLaunch(
    ur_queue_handle_t Queue,   ///< [in] handle of the queue object
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t WorkDim, ///< [in] number of dimensions, from 1 to 3, to specify
                      ///< the global and work-group work-items
    const size_t
        *GlobalWorkOffset, ///< [in] pointer to an array of workDim unsigned
                           ///< values that specify the offset used to
                           ///< calculate the global ID of a work-item
    const size_t *GlobalWorkSize, ///< [in] pointer to an array of workDim
                                  ///< unsigned values that specify the number
                                  ///< of global work-items in workDim that
                                  ///< will execute the kernel function
    const size_t
        *LocalWorkSize, ///< [in][optional] pointer to an array of workDim
                        ///< unsigned values that specify the number of local
                        ///< work-items forming a work-group that will execute
                        ///< the kernel function. If nullptr, the runtime
                        ///< implementation will choose the work-group size.
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating that no
                        ///< wait event.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular kernel execution instance.
) {
  UR_ASSERT(WorkDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(WorkDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  ze_kernel_handle_t ZeKernel{};
  UR_CALL(getZeKernel(Queue->Device->ZeDevice, Kernel, &ZeKernel));

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex, ur_shared_mutex> Lock(
      Queue->Mutex, Kernel->Mutex, Kernel->Program->Mutex);
  if (GlobalWorkOffset != NULL) {
    UR_CALL(setKernelGlobalOffset(Queue->Context, ZeKernel, WorkDim,
                                  GlobalWorkOffset));
  }

  // If there are any pending arguments set them now.
  for (auto &Arg : Kernel->PendingArguments) {
    // The ArgValue may be a NULL pointer in which case a NULL value is used for
    // the kernel argument declared as a pointer to global or constant memory.
    char **ZeHandlePtr = nullptr;
    if (Arg.Value) {
      UR_CALL(Arg.Value->getZeHandlePtr(ZeHandlePtr, Arg.AccessMode,
                                        Queue->Device, EventWaitList,
                                        NumEventsInWaitList));
    }
    ZE2UR_CALL(zeKernelSetArgumentValue,
               (ZeKernel, Arg.Index, Arg.Size, ZeHandlePtr));
  }
  Kernel->PendingArguments.clear();

  ze_group_count_t ZeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3]{};

  UR_CALL(calculateKernelWorkDimensions(Kernel->ZeKernel, Queue->Device,
                                        ZeThreadGroupDimensions, WG, WorkDim,
                                        GlobalWorkSize, LocalWorkSize));

  ZE2UR_CALL(zeKernelSetGroupSize, (ZeKernel, WG[0], WG[1], WG[2]));

  bool UseCopyEngine = false;
  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, NumEventsInWaitList, EventWaitList,
      true /* AllowBatching */, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent{};
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;

  UR_CALL(createEventAndAssociateQueue(Queue, Event, UR_COMMAND_KERNEL_LAUNCH,
                                       CommandList, IsInternal, false));
  UR_CALL(setSignalEvent(Queue, UseCopyEngine, &ZeEvent, Event,
                         NumEventsInWaitList, EventWaitList,
                         CommandList->second.ZeQueue));
  (*Event)->WaitList = TmpWaitList;

  // Save the kernel in the event, so that when the event is signalled
  // the code can do a urKernelRelease on this kernel.
  (*Event)->CommandData = (void *)Kernel;

  // Increment the reference count of the Kernel and indicate that the Kernel
  // is in use. Once the event has been signalled, the code in
  // CleanupCompletedEvent(Event) will do a urKernelRelease to update the
  // reference count on the kernel, using the kernel saved in CommandData.
  UR_CALL(ur::level_zero::urKernelRetain(Kernel));

  // Add to list of kernels to be submitted
  if (IndirectAccessTrackingEnabled)
    Queue->KernelsToBeSubmitted.push_back(Kernel);

  if (Queue->UsingImmCmdLists && IndirectAccessTrackingEnabled) {
    // If using immediate commandlists then gathering of indirect
    // references and appending to the queue (which means submission)
    // must be done together.
    std::unique_lock<ur_shared_mutex> ContextsLock(
        Queue->Device->Platform->ContextsMutex, std::defer_lock);
    // We are going to submit kernels for execution. If indirect access flag is
    // set for a kernel then we need to make a snapshot of existing memory
    // allocations in all contexts in the platform. We need to lock the mutex
    // guarding the list of contexts in the platform to prevent creation of new
    // memory alocations in any context before we submit the kernel for
    // execution.
    ContextsLock.lock();
    Queue->CaptureIndirectAccesses();
    // Add the command to the command list, which implies submission.
    ZE2UR_CALL(zeCommandListAppendLaunchKernel,
               (CommandList->first, ZeKernel, &ZeThreadGroupDimensions, ZeEvent,
                (*Event)->WaitList.Length, (*Event)->WaitList.ZeEventList));
  } else {
    // Add the command to the command list for later submission.
    // No lock is needed here, unlike the immediate commandlist case above,
    // because the kernels are not actually submitted yet. Kernels will be
    // submitted only when the comamndlist is closed. Then, a lock is held.
    ZE2UR_CALL(zeCommandListAppendLaunchKernel,
               (CommandList->first, ZeKernel, &ZeThreadGroupDimensions, ZeEvent,
                (*Event)->WaitList.Length, (*Event)->WaitList.ZeEventList));
  }

  logger::debug("calling zeCommandListAppendLaunchKernel() with"
                "  ZeEvent {}",
                ur_cast<std::uintptr_t>(ZeEvent));
  printZeEventList((*Event)->WaitList);

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  UR_CALL(Queue->executeCommandList(CommandList, false /*IsBlocking*/,
                                    true /*OKToBatchCommand*/));

  return UR_RESULT_SUCCESS;
}

ur_result_t urEnqueueCooperativeKernelLaunchExp(
    ur_queue_handle_t Queue,   ///< [in] handle of the queue object
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t WorkDim, ///< [in] number of dimensions, from 1 to 3, to specify
                      ///< the global and work-group work-items
    const size_t
        *GlobalWorkOffset, ///< [in] pointer to an array of workDim unsigned
                           ///< values that specify the offset used to
                           ///< calculate the global ID of a work-item
    const size_t *GlobalWorkSize, ///< [in] pointer to an array of workDim
                                  ///< unsigned values that specify the number
                                  ///< of global work-items in workDim that
                                  ///< will execute the kernel function
    const size_t
        *LocalWorkSize, ///< [in][optional] pointer to an array of workDim
                        ///< unsigned values that specify the number of local
                        ///< work-items forming a work-group that will execute
                        ///< the kernel function. If nullptr, the runtime
                        ///< implementation will choose the work-group size.
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating that no
                        ///< wait event.
    ur_event_handle_t
        *OutEvent ///< [in,out][optional] return an event object that identifies
                  ///< this particular kernel execution instance.
) {
  UR_ASSERT(WorkDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(WorkDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  auto ZeDevice = Queue->Device->ZeDevice;

  ze_kernel_handle_t ZeKernel{};
  if (Kernel->ZeKernelMap.empty()) {
    ZeKernel = Kernel->ZeKernel;
  } else {
    auto It = Kernel->ZeKernelMap.find(ZeDevice);
    if (It == Kernel->ZeKernelMap.end()) {
      /* kernel and queue don't match */
      return UR_RESULT_ERROR_INVALID_QUEUE;
    }
    ZeKernel = It->second;
  }
  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex, ur_shared_mutex> Lock(
      Queue->Mutex, Kernel->Mutex, Kernel->Program->Mutex);
  if (GlobalWorkOffset != NULL) {
    UR_CALL(setKernelGlobalOffset(Queue->Context, ZeKernel, WorkDim,
                                  GlobalWorkOffset));
  }

  // If there are any pending arguments set them now.
  for (auto &Arg : Kernel->PendingArguments) {
    // The ArgValue may be a NULL pointer in which case a NULL value is used for
    // the kernel argument declared as a pointer to global or constant memory.
    char **ZeHandlePtr = nullptr;
    if (Arg.Value) {
      UR_CALL(Arg.Value->getZeHandlePtr(ZeHandlePtr, Arg.AccessMode,
                                        Queue->Device, EventWaitList,
                                        NumEventsInWaitList));
    }
    ZE2UR_CALL(zeKernelSetArgumentValue,
               (ZeKernel, Arg.Index, Arg.Size, ZeHandlePtr));
  }
  Kernel->PendingArguments.clear();

  ze_group_count_t ZeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3]{};

  // New variable needed because GlobalWorkSize parameter might not be of size 3
  size_t GlobalWorkSize3D[3]{1, 1, 1};
  std::copy(GlobalWorkSize, GlobalWorkSize + WorkDim, GlobalWorkSize3D);

  if (LocalWorkSize) {
    // L0
    UR_ASSERT(LocalWorkSize[0] < (std::numeric_limits<uint32_t>::max)(),
              UR_RESULT_ERROR_INVALID_VALUE);
    UR_ASSERT(LocalWorkSize[1] < (std::numeric_limits<uint32_t>::max)(),
              UR_RESULT_ERROR_INVALID_VALUE);
    UR_ASSERT(LocalWorkSize[2] < (std::numeric_limits<uint32_t>::max)(),
              UR_RESULT_ERROR_INVALID_VALUE);
    WG[0] = static_cast<uint32_t>(LocalWorkSize[0]);
    WG[1] = static_cast<uint32_t>(LocalWorkSize[1]);
    WG[2] = static_cast<uint32_t>(LocalWorkSize[2]);
  } else {
    // We can't call to zeKernelSuggestGroupSize if 64-bit GlobalWorkSize
    // values do not fit to 32-bit that the API only supports currently.
    bool SuggestGroupSize = true;
    for (int I : {0, 1, 2}) {
      if (GlobalWorkSize3D[I] > UINT32_MAX) {
        SuggestGroupSize = false;
      }
    }
    if (SuggestGroupSize) {
      ZE2UR_CALL(zeKernelSuggestGroupSize,
                 (ZeKernel, GlobalWorkSize3D[0], GlobalWorkSize3D[1],
                  GlobalWorkSize3D[2], &WG[0], &WG[1], &WG[2]));
    } else {
      for (int I : {0, 1, 2}) {
        // Try to find a I-dimension WG size that the GlobalWorkSize[I] is
        // fully divisable with. Start with the max possible size in
        // each dimension.
        uint32_t GroupSize[] = {
            Queue->Device->ZeDeviceComputeProperties->maxGroupSizeX,
            Queue->Device->ZeDeviceComputeProperties->maxGroupSizeY,
            Queue->Device->ZeDeviceComputeProperties->maxGroupSizeZ};
        GroupSize[I] = (std::min)(size_t(GroupSize[I]), GlobalWorkSize3D[I]);
        while (GlobalWorkSize3D[I] % GroupSize[I]) {
          --GroupSize[I];
        }

        if (GlobalWorkSize3D[I] / GroupSize[I] > UINT32_MAX) {
          logger::error(
              "urEnqueueCooperativeKernelLaunchExp: can't find a WG size "
              "suitable for global work size > UINT32_MAX");
          return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
        }
        WG[I] = GroupSize[I];
      }
      logger::debug("urEnqueueCooperativeKernelLaunchExp: using computed WG "
                    "size = {{{}, {}, {}}}",
                    WG[0], WG[1], WG[2]);
    }
  }

  // TODO: assert if sizes do not fit into 32-bit?

  switch (WorkDim) {
  case 3:
    ZeThreadGroupDimensions.groupCountX =
        static_cast<uint32_t>(GlobalWorkSize3D[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        static_cast<uint32_t>(GlobalWorkSize3D[1] / WG[1]);
    ZeThreadGroupDimensions.groupCountZ =
        static_cast<uint32_t>(GlobalWorkSize3D[2] / WG[2]);
    break;
  case 2:
    ZeThreadGroupDimensions.groupCountX =
        static_cast<uint32_t>(GlobalWorkSize3D[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        static_cast<uint32_t>(GlobalWorkSize3D[1] / WG[1]);
    WG[2] = 1;
    break;
  case 1:
    ZeThreadGroupDimensions.groupCountX =
        static_cast<uint32_t>(GlobalWorkSize3D[0] / WG[0]);
    WG[1] = WG[2] = 1;
    break;

  default:
    logger::error("urEnqueueCooperativeKernelLaunchExp: unsupported work_dim");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  // Error handling for non-uniform group size case
  if (GlobalWorkSize3D[0] !=
      size_t(ZeThreadGroupDimensions.groupCountX) * WG[0]) {
    logger::error("urEnqueueCooperativeKernelLaunchExp: invalid work_dim. The "
                  "range is not a "
                  "multiple of the group size in the 1st dimension");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize3D[1] !=
      size_t(ZeThreadGroupDimensions.groupCountY) * WG[1]) {
    logger::error("urEnqueueCooperativeKernelLaunchExp: invalid work_dim. The "
                  "range is not a "
                  "multiple of the group size in the 2nd dimension");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize3D[2] !=
      size_t(ZeThreadGroupDimensions.groupCountZ) * WG[2]) {
    logger::debug("urEnqueueCooperativeKernelLaunchExp: invalid work_dim. The "
                  "range is not a "
                  "multiple of the group size in the 3rd dimension");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }

  ZE2UR_CALL(zeKernelSetGroupSize, (ZeKernel, WG[0], WG[1], WG[2]));

  bool UseCopyEngine = false;
  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, NumEventsInWaitList, EventWaitList,
      true /* AllowBatching */, nullptr /*ForcedCmdQueue*/));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent{};
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;

  UR_CALL(createEventAndAssociateQueue(Queue, Event, UR_COMMAND_KERNEL_LAUNCH,
                                       CommandList, IsInternal, false));
  UR_CALL(setSignalEvent(Queue, UseCopyEngine, &ZeEvent, Event,
                         NumEventsInWaitList, EventWaitList,
                         CommandList->second.ZeQueue));
  (*Event)->WaitList = TmpWaitList;

  // Save the kernel in the event, so that when the event is signalled
  // the code can do a urKernelRelease on this kernel.
  (*Event)->CommandData = (void *)Kernel;

  // Increment the reference count of the Kernel and indicate that the Kernel
  // is in use. Once the event has been signalled, the code in
  // CleanupCompletedEvent(Event) will do a urKernelRelease to update the
  // reference count on the kernel, using the kernel saved in CommandData.
  UR_CALL(ur::level_zero::urKernelRetain(Kernel));

  // Add to list of kernels to be submitted
  if (IndirectAccessTrackingEnabled)
    Queue->KernelsToBeSubmitted.push_back(Kernel);

  if (Queue->UsingImmCmdLists && IndirectAccessTrackingEnabled) {
    // If using immediate commandlists then gathering of indirect
    // references and appending to the queue (which means submission)
    // must be done together.
    std::unique_lock<ur_shared_mutex> ContextsLock(
        Queue->Device->Platform->ContextsMutex, std::defer_lock);
    // We are going to submit kernels for execution. If indirect access flag is
    // set for a kernel then we need to make a snapshot of existing memory
    // allocations in all contexts in the platform. We need to lock the mutex
    // guarding the list of contexts in the platform to prevent creation of new
    // memory alocations in any context before we submit the kernel for
    // execution.
    ContextsLock.lock();
    Queue->CaptureIndirectAccesses();
    // Add the command to the command list, which implies submission.
    ZE2UR_CALL(zeCommandListAppendLaunchCooperativeKernel,
               (CommandList->first, ZeKernel, &ZeThreadGroupDimensions, ZeEvent,
                (*Event)->WaitList.Length, (*Event)->WaitList.ZeEventList));
  } else {
    // Add the command to the command list for later submission.
    // No lock is needed here, unlike the immediate commandlist case above,
    // because the kernels are not actually submitted yet. Kernels will be
    // submitted only when the comamndlist is closed. Then, a lock is held.
    ZE2UR_CALL(zeCommandListAppendLaunchCooperativeKernel,
               (CommandList->first, ZeKernel, &ZeThreadGroupDimensions, ZeEvent,
                (*Event)->WaitList.Length, (*Event)->WaitList.ZeEventList));
  }

  logger::debug("calling zeCommandListAppendLaunchCooperativeKernel() with"
                "  ZeEvent {}",
                ur_cast<std::uintptr_t>(ZeEvent));
  printZeEventList((*Event)->WaitList);

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  UR_CALL(Queue->executeCommandList(CommandList, false /*IsBlocking*/,
                                    true /*OKToBatchCommand*/));

  return UR_RESULT_SUCCESS;
}

ur_result_t urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t Queue,     ///< [in] handle of the queue to submit to.
    ur_program_handle_t Program, ///< [in] handle of the program containing the
                                 ///< device global variable.
    const char
        *Name, ///< [in] the unique identifier for the device global variable.
    bool BlockingWrite, ///< [in] indicates if this operation should block.
    size_t Count,       ///< [in] the number of bytes to copy.
    size_t Offset, ///< [in] the byte offset into the device global variable to
                   ///< start copying.
    const void *Src, ///< [in] pointer to where the data must be copied from.
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list.
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating that no
                        ///< wait event.
    ur_event_handle_t
        *Event ///< [in,out][optional] return an event object that identifies
               ///< this particular kernel execution instance.
) {
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);
  // Find global variable pointer
  size_t GlobalVarSize = 0;
  void *GlobalVarPtr = nullptr;
  ze_module_handle_t ZeModule =
      Program->getZeModuleHandle(Queue->Device->ZeDevice);
  ZE2UR_CALL(zeModuleGetGlobalPointer,
             (ZeModule, Name, &GlobalVarSize, &GlobalVarPtr));
  if (GlobalVarSize < Offset + Count) {
    setErrorMessage("Write device global variable is out of range.",
                    UR_RESULT_ERROR_INVALID_VALUE,
                    static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_ARGUMENT));
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  }

  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = !IsDevicePointer(Queue->Context, Src);
  // For better performance, Copy Engines are not preferred given Shared
  // pointers on DG2.
  if (Queue->Device->isDG2() && IsSharedPointer(Queue->Context, Src)) {
    PreferCopyEngine = false;
  }

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyHelper(UR_COMMAND_DEVICE_GLOBAL_VARIABLE_WRITE, Queue,
                              ur_cast<char *>(GlobalVarPtr) + Offset,
                              BlockingWrite, Count, Src, NumEventsInWaitList,
                              EventWaitList, Event, PreferCopyEngine);
}

ur_result_t urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t Queue,     ///< [in] handle of the queue to submit to.
    ur_program_handle_t Program, ///< [in] handle of the program containing the
                                 ///< device global variable.
    const char
        *Name, ///< [in] the unique identifier for the device global variable.
    bool BlockingRead, ///< [in] indicates if this operation should block.
    size_t Count,      ///< [in] the number of bytes to copy.
    size_t Offset, ///< [in] the byte offset into the device global variable to
                   ///< start copying.
    void *Dst,     ///< [in] pointer to where the data must be copied to.
    uint32_t NumEventsInWaitList, ///< [in] size of the event wait list.
    const ur_event_handle_t
        *EventWaitList, ///< [in][optional][range(0, numEventsInWaitList)]
                        ///< pointer to a list of events that must be complete
                        ///< before the kernel execution. If nullptr, the
                        ///< numEventsInWaitList must be 0, indicating that no
                        ///< wait event.
    ur_event_handle_t
        *Event ///< [in,out][optional] return an event object that identifies
               ///< this particular kernel execution instance.
) {
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);
  ze_module_handle_t ZeModule =
      Program->getZeModuleHandle(Queue->Device->ZeDevice);
  // Find global variable pointer
  size_t GlobalVarSize = 0;
  void *GlobalVarPtr = nullptr;
  ZE2UR_CALL(zeModuleGetGlobalPointer,
             (ZeModule, Name, &GlobalVarSize, &GlobalVarPtr));
  if (GlobalVarSize < Offset + Count) {
    setErrorMessage("Read from device global variable is out of range.",
                    UR_RESULT_ERROR_INVALID_VALUE,
                    static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_ARGUMENT));
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  }

  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = !IsDevicePointer(Queue->Context, Dst);
  // For better performance, Copy Engines are not preferred given Shared
  // pointers on DG2.
  if (Queue->Device->isDG2() && IsSharedPointer(Queue->Context, Dst)) {
    PreferCopyEngine = false;
  }

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyHelper(
      UR_COMMAND_DEVICE_GLOBAL_VARIABLE_READ, Queue, Dst, BlockingRead, Count,
      ur_cast<char *>(GlobalVarPtr) + Offset, NumEventsInWaitList,
      EventWaitList, Event, PreferCopyEngine);
}

ur_result_t urKernelCreate(
    ur_program_handle_t Program, ///< [in] handle of the program instance
    const char *KernelName,      ///< [in] pointer to null-terminated string.
    ur_kernel_handle_t
        *RetKernel ///< [out] pointer to handle of kernel object created.
) {
  std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
  try {
    ur_kernel_handle_t_ *UrKernel = new ur_kernel_handle_t_(true, Program);
    *RetKernel = reinterpret_cast<ur_kernel_handle_t>(UrKernel);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  for (auto &Dev : Program->AssociatedDevices) {
    auto ZeDevice = Dev->ZeDevice;
    // Program may be associated with all devices from the context but built
    // only for subset of devices.
    if (Program->getState(ZeDevice) != ur_program_handle_t_::state::Exe)
      continue;

    auto ZeModule = Program->getZeModuleHandle(ZeDevice);
    ZeStruct<ze_kernel_desc_t> ZeKernelDesc;
    ZeKernelDesc.flags = 0;
    ZeKernelDesc.pKernelName = KernelName;

    ze_kernel_handle_t ZeKernel;
    auto ZeResult =
        ZE_CALL_NOCHECK(zeKernelCreate, (ZeModule, &ZeKernelDesc, &ZeKernel));
    // Gracefully handle the case that kernel create fails.
    if (ZeResult != ZE_RESULT_SUCCESS) {
      delete *RetKernel;
      *RetKernel = nullptr;
      return ze2urResult(ZeResult);
    }

    // Store the kernel in the ZeKernelMap so the correct
    // kernel can be retrieved later for a specific device
    // where a queue is being submitted.
    (*RetKernel)->ZeKernelMap[ZeDevice] = ZeKernel;
    (*RetKernel)->ZeKernels.push_back(ZeKernel);

    // If the device used to create the module's kernel is a root-device
    // then store the kernel also using the sub-devices, since application
    // could submit the root-device's kernel to a sub-device's queue.
    uint32_t SubDevicesCount = 0;
    zeDeviceGetSubDevices(ZeDevice, &SubDevicesCount, nullptr);
    std::vector<ze_device_handle_t> ZeSubDevices(SubDevicesCount);
    zeDeviceGetSubDevices(ZeDevice, &SubDevicesCount, ZeSubDevices.data());
    for (auto ZeSubDevice : ZeSubDevices) {
      (*RetKernel)->ZeKernelMap[ZeSubDevice] = ZeKernel;
    }
  }
  // There is no any successfully built executable for program.
  if ((*RetKernel)->ZeKernelMap.empty())
    return UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE;

  (*RetKernel)->ZeKernel = (*RetKernel)->ZeKernelMap.begin()->second;

  UR_CALL((*RetKernel)->initialize());

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelSetArgValue(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex, ///< [in] argument index in range [0, num args - 1]
    size_t ArgSize,    ///< [in] size of argument type
    const ur_kernel_arg_value_properties_t
        *Properties, ///< [in][optional] argument properties
    const void
        *PArgValue ///< [in] argument value represented as matching arg type.
) {
  std::ignore = Properties;

  UR_ASSERT(Kernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  // OpenCL: "the arg_value pointer can be NULL or point to a NULL value
  // in which case a NULL value will be used as the value for the argument
  // declared as a pointer to global or constant memory in the kernel"
  //
  // We don't know the type of the argument but it seems that the only time
  // SYCL RT would send a pointer to NULL in 'arg_value' is when the argument
  // is a NULL pointer. Treat a pointer to NULL in 'arg_value' as a NULL.
  if (ArgSize == sizeof(void *) && PArgValue &&
      *(void **)(const_cast<void *>(PArgValue)) == nullptr) {
    PArgValue = nullptr;
  }

  if (ArgIndex > Kernel->ZeKernelProperties->numKernelArgs - 1) {
    return UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX;
  }

  std::scoped_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  ze_result_t ZeResult = ZE_RESULT_SUCCESS;
  if (Kernel->ZeKernelMap.empty()) {
    auto ZeKernel = Kernel->ZeKernel;
    ZeResult = ZE_CALL_NOCHECK(zeKernelSetArgumentValue,
                               (ZeKernel, ArgIndex, ArgSize, PArgValue));
  } else {
    for (auto It : Kernel->ZeKernelMap) {
      auto ZeKernel = It.second;
      ZeResult = ZE_CALL_NOCHECK(zeKernelSetArgumentValue,
                                 (ZeKernel, ArgIndex, ArgSize, PArgValue));
    }
  }

  if (ZeResult == ZE_RESULT_ERROR_INVALID_ARGUMENT) {
    return UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE;
  }

  return ze2urResult(ZeResult);
}

ur_result_t urKernelSetArgLocal(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex, ///< [in] argument index in range [0, num args - 1]
    size_t ArgSize,    ///< [in] size of the local buffer to be allocated by the
                       ///< runtime
    const ur_kernel_arg_local_properties_t
        *Properties ///< [in][optional] argument properties
) {
  std::ignore = Properties;

  UR_CALL(ur::level_zero::urKernelSetArgValue(Kernel, ArgIndex, ArgSize,
                                              nullptr, nullptr));

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelGetInfo(
    ur_kernel_handle_t Kernel,  ///< [in] handle of the Kernel object
    ur_kernel_info_t ParamName, ///< [in] name of the Kernel property to query
    size_t PropSize,            ///< [in] the size of the Kernel property value.
    void *KernelInfo, ///< [in,out][optional] array of bytes holding the kernel
                      ///< info property. If propSize is not equal to or
                      ///< greater than the real number of bytes needed to
                      ///< return the info then the
                      ///< ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                      ///< pKernelInfo is not used.
    size_t *PropSizeRet ///< [out][optional] pointer to the actual size in
                        ///< bytes of data being queried by propName.
) {

  UrReturnHelper ReturnValue(PropSize, KernelInfo, PropSizeRet);

  std::shared_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  switch (ParamName) {
  case UR_KERNEL_INFO_CONTEXT:
    return ReturnValue(ur_context_handle_t{Kernel->Program->Context});
  case UR_KERNEL_INFO_PROGRAM:
    return ReturnValue(ur_program_handle_t{Kernel->Program});
  case UR_KERNEL_INFO_FUNCTION_NAME:
    try {
      std::string &KernelName = Kernel->ZeKernelName.get();
      return ReturnValue(static_cast<const char *>(KernelName.c_str()));
    } catch (const std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
  case UR_KERNEL_INFO_NUM_REGS:
  case UR_KERNEL_INFO_NUM_ARGS:
    return ReturnValue(uint32_t{Kernel->ZeKernelProperties->numKernelArgs});
  case UR_KERNEL_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{Kernel->RefCount.load()});
  case UR_KERNEL_INFO_ATTRIBUTES:
    try {
      uint32_t Size;
      ZE2UR_CALL(zeKernelGetSourceAttributes,
                 (Kernel->ZeKernel, &Size, nullptr));
      char *attributes = new char[Size];
      ZE2UR_CALL(zeKernelGetSourceAttributes,
                 (Kernel->ZeKernel, &Size, &attributes));
      auto Res = ReturnValue(static_cast<const char *>(attributes));
      delete[] attributes;
      return Res;
    } catch (const std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
  default:
    logger::error(
        "Unsupported ParamName in urKernelGetInfo: ParamName={}(0x{})",
        ParamName, logger::toHex(ParamName));
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelGetGroupInfo(
    ur_kernel_handle_t Kernel, ///< [in] handle of the Kernel object
    ur_device_handle_t Device, ///< [in] handle of the Device object
    ur_kernel_group_info_t
        ParamName, ///< [in] name of the work Group property to query
    size_t
        ParamValueSize, ///< [in] size of the Kernel Work Group property value
    void *ParamValue,   ///< [in,out][optional][range(0, propSize)] value of the
                        ///< Kernel Work Group property.
    size_t *ParamValueSizeRet ///< [out][optional] pointer to the actual size in
                              ///< bytes of data being queried by propName.
) {
  UrReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  std::shared_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  switch (ParamName) {
  case UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE: {
    // TODO: To revisit after level_zero/issues/262 is resolved
    struct {
      size_t Arr[3];
    } GlobalWorkSize = {{(Device->ZeDeviceComputeProperties->maxGroupSizeX *
                          Device->ZeDeviceComputeProperties->maxGroupCountX),
                         (Device->ZeDeviceComputeProperties->maxGroupSizeY *
                          Device->ZeDeviceComputeProperties->maxGroupCountY),
                         (Device->ZeDeviceComputeProperties->maxGroupSizeZ *
                          Device->ZeDeviceComputeProperties->maxGroupCountZ)}};
    return ReturnValue(GlobalWorkSize);
  }
  case UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
    ZeStruct<ze_kernel_max_group_size_properties_ext_t> workGroupProperties;
    workGroupProperties.maxGroupSize = 0;

    ZeStruct<ze_kernel_properties_t> kernelProperties;
    kernelProperties.pNext = &workGroupProperties;
    // Set the Kernel to use as the ZeKernel initally for native handle support.
    // This makes the assumption that this device is the same device where this
    // kernel was created.
    auto ZeKernelDevice = Kernel->ZeKernel;
    auto It = Kernel->ZeKernelMap.find(Device->ZeDevice);
    if (It != Kernel->ZeKernelMap.end()) {
      ZeKernelDevice = Kernel->ZeKernelMap[Device->ZeDevice];
    }
    if (ZeKernelDevice) {
      auto ZeResult = ZE_CALL_NOCHECK(zeKernelGetProperties,
                                      (ZeKernelDevice, &kernelProperties));
      if (ZeResult || workGroupProperties.maxGroupSize == 0) {
        return ReturnValue(
            uint64_t{Device->ZeDeviceComputeProperties->maxTotalGroupSize});
      }
      return ReturnValue(workGroupProperties.maxGroupSize);
    } else {
      return ReturnValue(
          uint64_t{Device->ZeDeviceComputeProperties->maxTotalGroupSize});
    }
  }
  case UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE: {
    struct {
      size_t Arr[3];
    } WgSize = {{Kernel->ZeKernelProperties->requiredGroupSizeX,
                 Kernel->ZeKernelProperties->requiredGroupSizeY,
                 Kernel->ZeKernelProperties->requiredGroupSizeZ}};
    return ReturnValue(WgSize);
  }
  case UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE:
    return ReturnValue(uint32_t{Kernel->ZeKernelProperties->localMemSize});
  case UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
    return ReturnValue(size_t{Device->ZeDeviceProperties->physicalEUSimdWidth});
  }
  case UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
    return ReturnValue(uint32_t{Kernel->ZeKernelProperties->privateMemSize});
  }
  case UR_KERNEL_GROUP_INFO_COMPILE_MAX_WORK_GROUP_SIZE:
  case UR_KERNEL_GROUP_INFO_COMPILE_MAX_LINEAR_WORK_GROUP_SIZE:
    // No corresponding enumeration in Level Zero
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  default: {
    logger::error(
        "Unknown ParamName in urKernelGetGroupInfo: ParamName={}(0x{})",
        ParamName, logger::toHex(ParamName));
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelGetSubGroupInfo(
    ur_kernel_handle_t Kernel, ///< [in] handle of the Kernel object
    ur_device_handle_t Device, ///< [in] handle of the Device object
    ur_kernel_sub_group_info_t
        PropName,       ///< [in] name of the SubGroup property to query
    size_t PropSize,    ///< [in] size of the Kernel SubGroup property value
    void *PropValue,    ///< [in,out][range(0, propSize)][optional] value of the
                        ///< Kernel SubGroup property.
    size_t *PropSizeRet ///< [out][optional] pointer to the actual size in
                        ///< bytes of data being queried by propName.
) {
  std::ignore = Device;

  UrReturnHelper ReturnValue(PropSize, PropValue, PropSizeRet);

  std::shared_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  if (PropName == UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE) {
    ReturnValue(uint32_t{Kernel->ZeKernelProperties->maxSubgroupSize});
  } else if (PropName == UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS) {
    ReturnValue(uint32_t{Kernel->ZeKernelProperties->maxNumSubgroups});
  } else if (PropName == UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS) {
    ReturnValue(uint32_t{Kernel->ZeKernelProperties->requiredNumSubGroups});
  } else if (PropName == UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL) {
    ReturnValue(uint32_t{Kernel->ZeKernelProperties->requiredSubgroupSize});
  } else {
    die("urKernelGetSubGroupInfo: parameter not implemented");
    return {};
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelRetain(
    ur_kernel_handle_t Kernel ///< [in] handle for the Kernel to retain
) {
  Kernel->RefCount.increment();

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelRelease(
    ur_kernel_handle_t Kernel ///< [in] handle for the Kernel to release
) {
  if (!Kernel->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  auto KernelProgram = Kernel->Program;
  if (Kernel->OwnNativeHandle) {
    for (auto &ZeKernel : Kernel->ZeKernels) {
      auto ZeResult = ZE_CALL_NOCHECK(zeKernelDestroy, (ZeKernel));
      // Gracefully handle the case that L0 was already unloaded.
      if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
        return ze2urResult(ZeResult);
    }
  }
  Kernel->ZeKernelMap.clear();
  if (IndirectAccessTrackingEnabled) {
    UR_CALL(ur::level_zero::urContextRelease(KernelProgram->Context));
  }
  // do a release on the program this kernel was part of without delete of the
  // program handle
  KernelProgram->ur_release_program_resources(false);

  delete Kernel;

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelSetArgPointer(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_pointer_properties_t
        *Properties,     ///< [in][optional] argument properties
    const void *ArgValue ///< [in][optional] SVM pointer to memory location
                         ///< holding the argument value. If null then argument
                         ///< value is considered null.
) {
  std::ignore = Properties;

  // KernelSetArgValue is expecting a pointer to the argument
  UR_CALL(ur::level_zero::urKernelSetArgValue(
      Kernel, ArgIndex, sizeof(const void *), nullptr, &ArgValue));
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelSetExecInfo(
    ur_kernel_handle_t Kernel,      ///< [in] handle of the kernel object
    ur_kernel_exec_info_t PropName, ///< [in] name of the execution attribute
    size_t PropSize,                ///< [in] size in byte the attribute value
    const ur_kernel_exec_info_properties_t
        *Properties, ///< [in][optional] pointer to execution info properties
    const void *PropValue ///< [in][range(0, propSize)] pointer to memory
                          ///< location holding the property value.
) {
  std::ignore = PropSize;
  std::ignore = Properties;

  std::scoped_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  for (auto &ZeKernel : Kernel->ZeKernels) {
    if (PropName == UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS &&
        *(static_cast<const ur_bool_t *>(PropValue)) == true) {
      // The whole point for users really was to not need to know anything
      // about the types of allocations kernel uses. So in DPC++ we always
      // just set all 3 modes for each kernel.
      ze_kernel_indirect_access_flags_t IndirectFlags =
          ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST |
          ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE |
          ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
      ZE2UR_CALL(zeKernelSetIndirectAccess, (ZeKernel, IndirectFlags));
    } else if (PropName == UR_KERNEL_EXEC_INFO_CACHE_CONFIG) {
      ze_cache_config_flag_t ZeCacheConfig{};
      auto CacheConfig =
          *(static_cast<const ur_kernel_cache_config_t *>(PropValue));
      if (CacheConfig == UR_KERNEL_CACHE_CONFIG_LARGE_SLM)
        ZeCacheConfig = ZE_CACHE_CONFIG_FLAG_LARGE_SLM;
      else if (CacheConfig == UR_KERNEL_CACHE_CONFIG_LARGE_DATA)
        ZeCacheConfig = ZE_CACHE_CONFIG_FLAG_LARGE_DATA;
      else if (CacheConfig == UR_KERNEL_CACHE_CONFIG_DEFAULT)
        ZeCacheConfig = static_cast<ze_cache_config_flag_t>(0);
      else
        // Unexpected cache configuration value.
        return UR_RESULT_ERROR_INVALID_VALUE;
      ZE2UR_CALL(zeKernelSetCacheConfig, (ZeKernel, ZeCacheConfig););
    } else {
      logger::error("urKernelSetExecInfo: unsupported ParamName");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelSetArgSampler(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_sampler_properties_t
        *Properties,             ///< [in][optional] argument properties
    ur_sampler_handle_t ArgValue ///< [in] handle of Sampler object.
) {
  std::ignore = Properties;
  std::scoped_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  if (ArgIndex > Kernel->ZeKernelProperties->numKernelArgs - 1) {
    return UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX;
  }
  ZE2UR_CALL(zeKernelSetArgumentValue, (Kernel->ZeKernel, ArgIndex,
                                        sizeof(void *), &ArgValue->ZeSampler));

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelSetArgMemObj(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_mem_obj_properties_t
        *Properties, ///< [in][optional] pointer to Memory object properties.
    ur_mem_handle_t ArgValue ///< [in][optional] handle of Memory object.
) {
  std::ignore = Properties;

  std::scoped_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  // The ArgValue may be a NULL pointer in which case a NULL value is used for
  // the kernel argument declared as a pointer to global or constant memory.

  if (ArgIndex > Kernel->ZeKernelProperties->numKernelArgs - 1) {
    return UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX;
  }

  ur_mem_handle_t_ *UrMem = ur_cast<ur_mem_handle_t_ *>(ArgValue);

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
  auto Arg = UrMem ? UrMem : nullptr;
  Kernel->PendingArguments.push_back(
      {ArgIndex, sizeof(void *), Arg, UrAccessMode});

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelGetNativeHandle(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel.
    ur_native_handle_t
        *NativeKernel ///< [out] a pointer to the native handle of the kernel.
) {
  std::shared_lock<ur_shared_mutex> Guard(Kernel->Mutex);

  *NativeKernel = reinterpret_cast<ur_native_handle_t>(Kernel->ZeKernel);
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelSuggestMaxCooperativeGroupCountExp(
    ur_kernel_handle_t hKernel, ur_device_handle_t hDevice, uint32_t workDim,
    const size_t *pLocalWorkSize, size_t dynamicSharedMemorySize,
    uint32_t *pGroupCountRet) {
  (void)dynamicSharedMemorySize;
  std::shared_lock<ur_shared_mutex> Guard(hKernel->Mutex);

  uint32_t WG[3];
  WG[0] = ur_cast<uint32_t>(pLocalWorkSize[0]);
  WG[1] = workDim >= 2 ? ur_cast<uint32_t>(pLocalWorkSize[1]) : 1;
  WG[2] = workDim == 3 ? ur_cast<uint32_t>(pLocalWorkSize[2]) : 1;
  ZE2UR_CALL(zeKernelSetGroupSize, (hKernel->ZeKernel, WG[0], WG[1], WG[2]));

  uint32_t TotalGroupCount = 0;
  ze_kernel_handle_t ZeKernel;
  UR_CALL(getZeKernel(hDevice->ZeDevice, hKernel, &ZeKernel));
  ZE2UR_CALL(zeKernelSuggestMaxCooperativeGroupCount,
             (ZeKernel, &TotalGroupCount));
  *pGroupCountRet = TotalGroupCount;
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelCreateWithNativeHandle(
    ur_native_handle_t NativeKernel, ///< [in] the native handle of the kernel.
    ur_context_handle_t Context,     ///< [in] handle of the context object
    ur_program_handle_t Program,
    const ur_kernel_native_properties_t *Properties,
    ur_kernel_handle_t *
        RetKernel ///< [out] pointer to the handle of the kernel object created.
) {
  if (!Program) {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }
  ze_kernel_handle_t ZeKernel = ur_cast<ze_kernel_handle_t>(NativeKernel);
  ur_kernel_handle_t_ *Kernel = nullptr;
  try {
    Kernel = new ur_kernel_handle_t_(ZeKernel, Properties->isNativeHandleOwned,
                                     Context);
    if (Properties->isNativeHandleOwned) {
      // If ownership is passed to the adapter we need to pass the kernel
      // to this vector which is then used during ZeKernelRelease.
      Kernel->ZeKernels.push_back(ZeKernel);
    }

    *RetKernel = reinterpret_cast<ur_kernel_handle_t>(Kernel);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  Kernel->Program = Program;

  UR_CALL(Kernel->initialize());

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelSetSpecializationConstants(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t Count, ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t
        *SpecConstants ///< [in] array of specialization constant value
                       ///< descriptions
) {
  std::ignore = Kernel;
  std::ignore = Count;
  std::ignore = SpecConstants;
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::level_zero

ur_result_t ur_kernel_handle_t_::initialize() {
  // Retain the program and context to show it's used by this kernel.
  UR_CALL(ur::level_zero::urProgramRetain(Program));

  if (IndirectAccessTrackingEnabled)
    // TODO: do piContextRetain without the guard
    UR_CALL(ur::level_zero::urContextRetain(Program->Context));

  // Set up how to obtain kernel properties when needed.
  ZeKernelProperties.Compute = [this](ze_kernel_properties_t &Properties) {
    ZE_CALL_NOCHECK(zeKernelGetProperties, (ZeKernel, &Properties));
  };

  // Cache kernel name.
  ZeKernelName.Compute = [this](std::string &Name) {
    size_t Size = 0;
    ZE_CALL_NOCHECK(zeKernelGetName, (ZeKernel, &Size, nullptr));
    char *KernelName = new char[Size];
    ZE_CALL_NOCHECK(zeKernelGetName, (ZeKernel, &Size, KernelName));
    Name = KernelName;
    delete[] KernelName;
  };

  return UR_RESULT_SUCCESS;
}
