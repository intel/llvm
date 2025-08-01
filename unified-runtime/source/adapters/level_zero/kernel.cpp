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

inline ur_result_t KernelSetArgValueHelper(
    ur_kernel_handle_t Kernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t ArgIndex,
    /// [in] size of argument type
    size_t ArgSize,
    /// [in] argument value represented as matching arg type.
    const void *PArgValue) {
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

inline ur_result_t KernelSetArgMemObjHelper(
    /// [in] handle of the kernel object
    ur_kernel_handle_t Kernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t ArgIndex,
    /// [in][optional] pointer to Memory object properties.
    const ur_kernel_arg_mem_obj_properties_t *Properties,
    /// [in][optional] handle of Memory object.
    ur_mem_handle_t ArgValue) {
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
    case 0:
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

ur_result_t urEnqueueKernelLaunchWithArgsExp(
    /// [in] handle of the queue object
    ur_queue_handle_t Queue,
    /// [in] handle of the kernel object
    ur_kernel_handle_t Kernel,
    /// [in] number of dimensions, from 1 to 3, to specify the global and
    /// work-group work-items
    uint32_t workDim,
    /// [in][optional] pointer to an array of workDim unsigned values that
    /// specify the offset used to calculate the global ID of a work-item
    const size_t *GlobalWorkOffset,
    /// [in] pointer to an array of workDim unsigned values that specify the
    /// number of global work-items in workDim that will execute the kernel
    /// function
    const size_t *GlobalWorkSize,
    /// [in][optional] pointer to an array of workDim unsigned values that
    /// specify the number of local work-items forming a work-group that will
    /// execute the kernel function.
    /// If nullptr, the runtime implementation will choose the work-group size.
    const size_t *LocalWorkSize,
    /// [in] size of the event wait list
    uint32_t NumArgs,
    /// [in][optional][range(0, numArgs)] pointer to a list of kernel arg
    /// properties.
    const ur_exp_kernel_arg_properties_t *Args,
    /// [in] size of the launch prop list
    uint32_t NumPropsInLaunchPropList,
    /// [in][range(0, numPropsInLaunchPropList)] pointer to a list of launch
    /// properties
    const ur_kernel_launch_property_t *LaunchPropList,
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular kernel execution instance.
    ur_event_handle_t *OutEvent) {
  {
    std::scoped_lock<ur_shared_mutex> Guard(Kernel->Mutex);
    for (uint32_t i = 0; i < NumArgs; i++) {
      switch (Args[i].type) {
      case UR_EXP_KERNEL_ARG_TYPE_LOCAL:
        UR_CALL(KernelSetArgValueHelper(Kernel, Args[i].index, Args[i].size,
                                        nullptr));
        break;
      case UR_EXP_KERNEL_ARG_TYPE_VALUE:
        UR_CALL(KernelSetArgValueHelper(Kernel, Args[i].index, Args[i].size,
                                        Args[i].value.value));
        break;
      case UR_EXP_KERNEL_ARG_TYPE_POINTER:
        UR_CALL(KernelSetArgValueHelper(Kernel, Args[i].index, Args[i].size,
                                        &Args[i].value.pointer));
        break;
      case UR_EXP_KERNEL_ARG_TYPE_MEM_OBJ: {
        ur_kernel_arg_mem_obj_properties_t Properties = {
            UR_STRUCTURE_TYPE_KERNEL_ARG_MEM_OBJ_PROPERTIES, nullptr,
            Args[i].value.memObjTuple.flags};
        UR_CALL(KernelSetArgMemObjHelper(Kernel, Args[i].index, &Properties,
                                         Args[i].value.memObjTuple.hMem));
        break;
      }
      case UR_EXP_KERNEL_ARG_TYPE_SAMPLER: {
        UR_CALL(KernelSetArgValueHelper(Kernel, Args[i].index, Args[i].size,
                                        &Args[i].value.sampler->ZeSampler));
        break;
      }
      default:
        return UR_RESULT_ERROR_INVALID_ENUMERATION;
      }
    }
  }
  // Normalize so each dimension has at least one work item
  return level_zero::urEnqueueKernelLaunch(
      Queue, Kernel, workDim, GlobalWorkOffset, GlobalWorkSize, LocalWorkSize,
      NumPropsInLaunchPropList, LaunchPropList, NumEventsInWaitList,
      EventWaitList, OutEvent);
}

ur_result_t urEnqueueKernelLaunch(
    /// [in] handle of the queue object
    ur_queue_handle_t Queue,
    /// [in] handle of the kernel object
    ur_kernel_handle_t Kernel,
    /// [in] number of dimensions, from 1 to 3, to specify the global and
    /// work-group work-items
    uint32_t WorkDim,
    /// [in][optional] pointer to an array of workDim unsigned values that
    /// specify the offset used to calculate the global ID of a work-item
    const size_t *GlobalWorkOffset,
    /// [in] pointer to an array of workDim unsigned values that specify the
    /// number of global work-items in workDim that will execute the kernel
    /// function
    const size_t *GlobalWorkSize,
    /// [in][optional] pointer to an array of workDim unsigned values that
    /// specify the number of local work-items forming a work-group that
    /// will execute the kernel function. If nullptr, the runtime
    /// implementation will choose the work-group size.
    const size_t *LocalWorkSize,
    /// [in] size of the launch prop list
    uint32_t NumPropsInLaunchPropList,
    /// [in][range(0, numPropsInLaunchPropList)] pointer to a list of launch
    /// properties
    const ur_kernel_launch_property_t *LaunchPropList,
    /// [in] size of the event wait list
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular kernel execution instance.
    ur_event_handle_t *OutEvent) {
  using ZeKernelLaunchFuncT = ze_result_t (*)(
      ze_command_list_handle_t, ze_kernel_handle_t, const ze_group_count_t *,
      ze_event_handle_t, uint32_t, ze_event_handle_t *);
  ZeKernelLaunchFuncT ZeKernelLaunchFunc = &zeCommandListAppendLaunchKernel;
  for (uint32_t PropIndex = 0; PropIndex < NumPropsInLaunchPropList;
       PropIndex++) {
    if (LaunchPropList[PropIndex].id ==
            UR_KERNEL_LAUNCH_PROPERTY_ID_COOPERATIVE &&
        LaunchPropList[PropIndex].value.cooperative) {
      ZeKernelLaunchFunc = &zeCommandListAppendLaunchCooperativeKernel;
    }
    if (LaunchPropList[PropIndex].id != UR_KERNEL_LAUNCH_PROPERTY_ID_IGNORE &&
        LaunchPropList[PropIndex].id !=
            UR_KERNEL_LAUNCH_PROPERTY_ID_COOPERATIVE) {
      // We don't support any other properties.
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }
  }
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
  ur_ze_event_list_t TmpWaitList;
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
    ZE2UR_CALL(ZeKernelLaunchFunc,
               (CommandList->first, ZeKernel, &ZeThreadGroupDimensions, ZeEvent,
                (*Event)->WaitList.Length, (*Event)->WaitList.ZeEventList));
  } else {
    // Add the command to the command list for later submission.
    // No lock is needed here, unlike the immediate commandlist case above,
    // because the kernels are not actually submitted yet. Kernels will be
    // submitted only when the comamndlist is closed. Then, a lock is held.
    ZE2UR_CALL(ZeKernelLaunchFunc,
               (CommandList->first, ZeKernel, &ZeThreadGroupDimensions, ZeEvent,
                (*Event)->WaitList.Length, (*Event)->WaitList.ZeEventList));
  }

  UR_LOG(DEBUG, "calling zeCommandListAppendLaunchKernel() with ZeEvent {}",
         ur_cast<std::uintptr_t>(ZeEvent));
  printZeEventList((*Event)->WaitList);

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  UR_CALL(Queue->executeCommandList(CommandList, false /*IsBlocking*/,
                                    true /*OKToBatchCommand*/));

  return UR_RESULT_SUCCESS;
}

ur_result_t urEnqueueDeviceGlobalVariableWrite(
    /// [in] handle of the queue to submit to.
    ur_queue_handle_t Queue,
    /// [in] handle of the program containing the device global variable.
    ur_program_handle_t Program,
    /// [in] the unique identifier for the device global variable.
    const char *Name,
    /// [in] indicates if this operation should block.
    bool BlockingWrite,
    /// [in] the number of bytes to copy.
    size_t Count,
    /// [in] the byte offset into the device global variable to start copying.
    size_t Offset,
    /// [in] pointer to where the data must be copied from.
    const void *Src,
    /// [in] size of the event wait list.
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular kernel execution instance.
    ur_event_handle_t *Event) {
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
    /// [in] handle of the queue to submit to.
    ur_queue_handle_t Queue,
    /// [in] handle of the program containing the device global variable.
    ur_program_handle_t Program,
    const char
        /// [in] the unique identifier for the device global variable.
        *Name,
    /// [in] indicates if this operation should block.
    bool BlockingRead,
    /// [in] the number of bytes to copy.
    size_t Count,
    /// [in] the byte offset into the device global variable to start copying.
    size_t Offset,
    /// [in] pointer to where the data must be copied to.
    void *Dst,
    /// [in] size of the event wait list.
    uint32_t NumEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *EventWaitList,
    /// [in,out][optional] return an event object that identifies this
    /// particular kernel execution instance.
    ur_event_handle_t *Event) {
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
    /// [in] handle of the program instance
    ur_program_handle_t Program,
    /// [in] pointer to null-terminated string.
    const char *KernelName,
    /// [out] pointer to handle of kernel object created.
    ur_kernel_handle_t *RetKernel) {
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
    /// [in] handle of the kernel object
    ur_kernel_handle_t Kernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t ArgIndex,
    /// [in] size of argument type
    size_t ArgSize,
    /// [in][optional] argument properties
    const ur_kernel_arg_value_properties_t * /*Properties*/,
    /// [in] argument value represented as matching arg type.
    const void *PArgValue) {

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
    /// [in] handle of the kernel object
    ur_kernel_handle_t Kernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t ArgIndex,
    /// [in] size of the local buffer to be allocated by the runtime
    size_t ArgSize,
    /// [in][optional] argument properties
    const ur_kernel_arg_local_properties_t * /*Properties*/) {

  UR_CALL(ur::level_zero::urKernelSetArgValue(Kernel, ArgIndex, ArgSize,
                                              nullptr, nullptr));

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelGetInfo(
    /// [in] handle of the Kernel object
    ur_kernel_handle_t Kernel,
    /// [in] name of the Kernel property to query
    ur_kernel_info_t ParamName,
    /// [in] the size of the Kernel property value.
    size_t PropSize,
    /// [in,out][optional] array of bytes holding the kernel info property. If
    /// propSize is not equal to or greater than the real number of bytes needed
    /// to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    /// returned and pKernelInfo is not used.
    void *KernelInfo,
    /// [out][optional] pointer to the actual size in bytes of data being
    /// queried by propName.
    size_t *PropSizeRet) {

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
  case UR_KERNEL_INFO_SPILL_MEM_SIZE: {
    try {
      std::vector<uint32_t> Spills;
      Spills.reserve(Kernel->ZeKernels.size());
      for (auto &ZeKernel : Kernel->ZeKernels) {
        ze_kernel_properties_t props;
        props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
        props.pNext = nullptr;
        ZE2UR_CALL(zeKernelGetProperties, (ZeKernel, &props));
        uint32_t spillMemSize = props.spillMemSize;
        Spills.push_back(spillMemSize);
      }
      return ReturnValue(static_cast<const uint32_t *>(Spills.data()),
                         Spills.size());
    } catch (const std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
  }
  case UR_KERNEL_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{Kernel->RefCount.getCount()});
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
    UR_LOG(ERR, "Unsupported ParamName in urKernelGetInfo: ParamName={}(0x{})",
           ParamName, logger::toHex(ParamName));
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelGetGroupInfo(
    /// [in] handle of the Kernel object
    ur_kernel_handle_t Kernel,
    /// [in] handle of the Device object
    ur_device_handle_t Device,
    /// [in] name of the work Group property to query
    ur_kernel_group_info_t ParamName,
    /// [in] size of the Kernel Work Group property value
    size_t ParamValueSize,
    /// [in,out][optional][range(0, propSize)] value of the Kernel Work Group
    /// property.
    void *ParamValue,
    /// [out][optional] pointer to the actual size in bytes of data being
    /// queried by propName.
    size_t *ParamValueSizeRet) {
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
            size_t{Device->ZeDeviceComputeProperties->maxTotalGroupSize});
      }
      // Specification states this returns a size_t.
      return ReturnValue(size_t{workGroupProperties.maxGroupSize});
    } else {
      return ReturnValue(
          size_t{Device->ZeDeviceComputeProperties->maxTotalGroupSize});
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
    return ReturnValue(size_t{Kernel->ZeKernelProperties->localMemSize});
  case UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
    return ReturnValue(size_t{Device->ZeDeviceProperties->physicalEUSimdWidth});
  }
  case UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
    return ReturnValue(size_t{Kernel->ZeKernelProperties->privateMemSize});
  }
  case UR_KERNEL_GROUP_INFO_COMPILE_MAX_WORK_GROUP_SIZE:
  case UR_KERNEL_GROUP_INFO_COMPILE_MAX_LINEAR_WORK_GROUP_SIZE:
    // No corresponding enumeration in Level Zero
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  default: {
    UR_LOG(ERR, "Unknown ParamName in urKernelGetGroupInfo: ParamName={}(0x{})",
           ParamName, logger::toHex(ParamName));
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelGetSubGroupInfo(
    /// [in] handle of the Kernel object
    ur_kernel_handle_t Kernel,
    /// [in] handle of the Device object
    ur_device_handle_t /*Device*/,
    /// [in] name of the SubGroup property to query
    ur_kernel_sub_group_info_t PropName,
    /// [in] size of the Kernel SubGroup property value
    size_t PropSize,
    /// [in,out][range(0, propSize)][optional] value of the Kernel SubGroup
    /// property.
    void *PropValue,
    /// [out][optional] pointer to the actual size in bytes of data being
    /// queried by propName.
    size_t *PropSizeRet) {

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
    /// [in] handle for the Kernel to retain
    ur_kernel_handle_t Kernel) {
  Kernel->RefCount.retain();

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelRelease(
    /// [in] handle for the Kernel to release
    ur_kernel_handle_t Kernel) {
  if (!Kernel->RefCount.release())
    return UR_RESULT_SUCCESS;

  auto KernelProgram = Kernel->Program;
  if (Kernel->OwnNativeHandle) {
    for (auto &ZeKernel : Kernel->ZeKernels) {
      if (checkL0LoaderTeardown()) {
        auto ZeResult = ZE_CALL_NOCHECK(zeKernelDestroy, (ZeKernel));
        // Gracefully handle the case that L0 was already unloaded.
        if (ZeResult && (ZeResult != ZE_RESULT_ERROR_UNINITIALIZED &&
                         ZeResult != ZE_RESULT_ERROR_UNKNOWN))
          return ze2urResult(ZeResult);
        if (ZeResult == ZE_RESULT_ERROR_UNKNOWN) {
          ZeResult = ZE_RESULT_ERROR_UNINITIALIZED;
        }
      }
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
    /// [in] handle of the kernel object
    ur_kernel_handle_t Kernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t ArgIndex,
    /// [in][optional] argument properties
    const ur_kernel_arg_pointer_properties_t * /*Properties*/,
    /// [in][optional] SVM pointer to memory location holding the argument
    /// value. If null then argument value is considered null.
    const void *ArgValue) {

  // KernelSetArgValue is expecting a pointer to the argument
  UR_CALL(ur::level_zero::urKernelSetArgValue(
      Kernel, ArgIndex, sizeof(const void *), nullptr, &ArgValue));
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelSetExecInfo(
    /// [in] handle of the kernel object
    ur_kernel_handle_t Kernel,
    /// [in] name of the execution attribute
    ur_kernel_exec_info_t PropName,
    /// [in] size in byte the attribute value
    size_t /*PropSize*/,
    /// [in][optional] pointer to execution info properties
    const ur_kernel_exec_info_properties_t * /*Properties*/,
    /// [in][range(0, propSize)] pointer to memory location holding the property
    /// value.
    const void *PropValue) {

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
      UR_LOG(ERR, "urKernelSetExecInfo: unsupported ParamName");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelSetArgSampler(
    /// [in] handle of the kernel object
    ur_kernel_handle_t Kernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t ArgIndex,
    /// [in][optional] argument properties
    const ur_kernel_arg_sampler_properties_t * /*Properties*/,
    /// [in] handle of Sampler object.
    ur_sampler_handle_t ArgValue) {
  std::scoped_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  if (ArgIndex > Kernel->ZeKernelProperties->numKernelArgs - 1) {
    return UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX;
  }
  ZE2UR_CALL(zeKernelSetArgumentValue, (Kernel->ZeKernel, ArgIndex,
                                        sizeof(void *), &ArgValue->ZeSampler));

  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelSetArgMemObj(
    /// [in] handle of the kernel object
    ur_kernel_handle_t Kernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t ArgIndex,
    /// [in][optional] pointer to Memory object properties.
    const ur_kernel_arg_mem_obj_properties_t *Properties,
    /// [in][optional] handle of Memory object.
    ur_mem_handle_t ArgValue) {

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
    /// [in] handle of the kernel.
    ur_kernel_handle_t Kernel,
    /// [out] a pointer to the native handle of the kernel.
    ur_native_handle_t *NativeKernel) {
  std::shared_lock<ur_shared_mutex> Guard(Kernel->Mutex);

  *NativeKernel = reinterpret_cast<ur_native_handle_t>(Kernel->ZeKernel);
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelSuggestMaxCooperativeGroupCount(
    ur_kernel_handle_t hKernel, ur_device_handle_t hDevice, uint32_t workDim,
    const size_t *pLocalWorkSize, size_t dynamicSharedMemorySize,
    uint32_t *pGroupCountRet) {
  (void)dynamicSharedMemorySize;
  std::shared_lock<ur_shared_mutex> Guard(hKernel->Mutex);

  ze_kernel_handle_t ZeKernel;
  UR_CALL(getZeKernel(hDevice->ZeDevice, hKernel, &ZeKernel));

  uint32_t WG[3];
  WG[0] = ur_cast<uint32_t>(pLocalWorkSize[0]);
  WG[1] = workDim >= 2 ? ur_cast<uint32_t>(pLocalWorkSize[1]) : 1;
  WG[2] = workDim == 3 ? ur_cast<uint32_t>(pLocalWorkSize[2]) : 1;
  ZE2UR_CALL(zeKernelSetGroupSize, (ZeKernel, WG[0], WG[1], WG[2]));

  uint32_t TotalGroupCount = 0;
  ZE2UR_CALL(zeKernelSuggestMaxCooperativeGroupCount,
             (ZeKernel, &TotalGroupCount));
  *pGroupCountRet = TotalGroupCount;
  return UR_RESULT_SUCCESS;
}

ur_result_t urKernelCreateWithNativeHandle(
    /// [in] the native handle of the kernel.
    ur_native_handle_t NativeKernel,
    /// [in] handle of the context object
    ur_context_handle_t Context, ur_program_handle_t Program,
    const ur_kernel_native_properties_t *Properties,
    /// [out] pointer to the handle of the kernel object created.
    ur_kernel_handle_t *RetKernel) {
  if (!Program) {
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }
  ze_kernel_handle_t ZeKernel = ur_cast<ze_kernel_handle_t>(NativeKernel);
  ur_kernel_handle_t_ *Kernel = nullptr;
  try {
    auto OwnNativeHandle = Properties ? Properties->isNativeHandleOwned : false;
    Kernel = new ur_kernel_handle_t_(ZeKernel, OwnNativeHandle, Context);
    if (OwnNativeHandle) {
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
    /// [in] handle of the kernel object
    ur_kernel_handle_t /*Kernel*/,
    /// [in] the number of elements in the pSpecConstants array
    uint32_t /*Count*/,
    const ur_specialization_constant_info_t
        /// [in] array of specialization constant value descriptions
        * /*SpecConstants*/) {
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
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
