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
#include "ur_level_zero.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(
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
  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex, ur_shared_mutex> Lock(
      Queue->Mutex, Kernel->Mutex, Kernel->Program->Mutex);
  if (GlobalWorkOffset != NULL) {
    if (!Queue->Device->Platform->ZeDriverGlobalOffsetExtensionFound) {
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
                                        Queue->Device));
    }
    ZE2UR_CALL(zeKernelSetArgumentValue,
               (Kernel->ZeKernel, Arg.Index, Arg.Size, ZeHandlePtr));
  }
  Kernel->PendingArguments.clear();

  ze_group_count_t ZeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3]{};

  // global_work_size of unused dimensions must be set to 1
  UR_ASSERT(WorkDim == 3 || GlobalWorkSize[2] == 1,
            UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT(WorkDim >= 2 || GlobalWorkSize[1] == 1,
            UR_RESULT_ERROR_INVALID_VALUE);
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
            Queue->Device->ZeDeviceComputeProperties->maxGroupSizeX,
            Queue->Device->ZeDeviceComputeProperties->maxGroupSizeY,
            Queue->Device->ZeDeviceComputeProperties->maxGroupSizeZ};
        GroupSize[I] = (std::min)(size_t(GroupSize[I]), GlobalWorkSize[I]);
        while (GlobalWorkSize[I] % GroupSize[I]) {
          --GroupSize[I];
        }
        if (GlobalWorkSize[I] / GroupSize[I] > UINT32_MAX) {
          urPrint("urEnqueueKernelLaunch: can't find a WG size "
                  "suitable for global work size > UINT32_MAX\n");
          return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
        }
        WG[I] = GroupSize[I];
      }
      urPrint("urEnqueueKernelLaunch: using computed WG size = {%d, %d, %d}\n",
              WG[0], WG[1], WG[2]);
    }
  }

  // TODO: assert if sizes do not fit into 32-bit?

  switch (WorkDim) {
  case 3:
    ZeThreadGroupDimensions.groupCountX =
        static_cast<uint32_t>(GlobalWorkSize[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        static_cast<uint32_t>(GlobalWorkSize[1] / WG[1]);
    ZeThreadGroupDimensions.groupCountZ =
        static_cast<uint32_t>(GlobalWorkSize[2] / WG[2]);
    break;
  case 2:
    ZeThreadGroupDimensions.groupCountX =
        static_cast<uint32_t>(GlobalWorkSize[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        static_cast<uint32_t>(GlobalWorkSize[1] / WG[1]);
    WG[2] = 1;
    break;
  case 1:
    ZeThreadGroupDimensions.groupCountX =
        static_cast<uint32_t>(GlobalWorkSize[0] / WG[0]);
    WG[1] = WG[2] = 1;
    break;

  default:
    urPrint("urEnqueueKernelLaunch: unsupported work_dim\n");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  // Error handling for non-uniform group size case
  if (GlobalWorkSize[0] !=
      size_t(ZeThreadGroupDimensions.groupCountX) * WG[0]) {
    urPrint("urEnqueueKernelLaunch: invalid work_dim. The range is not a "
            "multiple of the group size in the 1st dimension\n");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize[1] !=
      size_t(ZeThreadGroupDimensions.groupCountY) * WG[1]) {
    urPrint("urEnqueueKernelLaunch: invalid work_dim. The range is not a "
            "multiple of the group size in the 2nd dimension\n");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize[2] !=
      size_t(ZeThreadGroupDimensions.groupCountZ) * WG[2]) {
    urPrint("urEnqueueKernelLaunch: invalid work_dim. The range is not a "
            "multiple of the group size in the 3rd dimension\n");
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }

  ZE2UR_CALL(zeKernelSetGroupSize, (Kernel->ZeKernel, WG[0], WG[1], WG[2]));

  bool UseCopyEngine = false;
  _ur_ze_event_list_t TmpWaitList;
  UR_CALL(TmpWaitList.createAndRetainUrZeEventList(
      NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine));

  // Get a new command list to be used on this call
  ur_command_list_ptr_t CommandList{};
  UR_CALL(Queue->Context->getAvailableCommandList(
      Queue, CommandList, UseCopyEngine, true /* AllowBatching */));

  ze_event_handle_t ZeEvent = nullptr;
  ur_event_handle_t InternalEvent{};
  bool IsInternal = OutEvent == nullptr;
  ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;

  UR_CALL(createEventAndAssociateQueue(Queue, Event, UR_COMMAND_KERNEL_LAUNCH,
                                       CommandList, IsInternal));
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  // Save the kernel in the event, so that when the event is signalled
  // the code can do a urKernelRelease on this kernel.
  (*Event)->CommandData = (void *)Kernel;

  // Increment the reference count of the Kernel and indicate that the Kernel is
  // in use. Once the event has been signalled, the code in
  // CleanupCompletedEvent(Event) will do a urKernelRelease to update the
  // reference count on the kernel, using the kernel saved in CommandData.
  UR_CALL(urKernelRetain(Kernel));

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
               (CommandList->first, Kernel->ZeKernel, &ZeThreadGroupDimensions,
                ZeEvent, (*Event)->WaitList.Length,
                (*Event)->WaitList.ZeEventList));
  } else {
    // Add the command to the command list for later submission.
    // No lock is needed here, unlike the immediate commandlist case above,
    // because the kernels are not actually submitted yet. Kernels will be
    // submitted only when the comamndlist is closed. Then, a lock is held.
    ZE2UR_CALL(zeCommandListAppendLaunchKernel,
               (CommandList->first, Kernel->ZeKernel, &ZeThreadGroupDimensions,
                ZeEvent, (*Event)->WaitList.Length,
                (*Event)->WaitList.ZeEventList));
  }

  urPrint("calling zeCommandListAppendLaunchKernel() with"
          "  ZeEvent %#" PRIxPTR "\n",
          ur_cast<std::uintptr_t>(ZeEvent));
  printZeEventList((*Event)->WaitList);

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  UR_CALL(Queue->executeCommandList(CommandList, false, true));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
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
  ZE2UR_CALL(zeModuleGetGlobalPointer,
             (Program->ZeModule, Name, &GlobalVarSize, &GlobalVarPtr));
  if (GlobalVarSize < Offset + Count) {
    setErrorMessage("Write device global variable is out of range.",
                    UR_RESULT_ERROR_INVALID_VALUE,
                    static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_ARGUMENT));
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  }

  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = !IsDevicePointer(Queue->Context, Src);

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyHelper(UR_COMMAND_DEVICE_GLOBAL_VARIABLE_WRITE, Queue,
                              ur_cast<char *>(GlobalVarPtr) + Offset,
                              BlockingWrite, Count, Src, NumEventsInWaitList,
                              EventWaitList, Event, PreferCopyEngine);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
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

  // Find global variable pointer
  size_t GlobalVarSize = 0;
  void *GlobalVarPtr = nullptr;
  ZE2UR_CALL(zeModuleGetGlobalPointer,
             (Program->ZeModule, Name, &GlobalVarSize, &GlobalVarPtr));
  if (GlobalVarSize < Offset + Count) {
    setErrorMessage("Read from device global variable is out of range.",
                    UR_RESULT_ERROR_INVALID_VALUE,
                    static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_ARGUMENT));
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  }

  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = !IsDevicePointer(Queue->Context, Dst);

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyHelper(
      UR_COMMAND_DEVICE_GLOBAL_VARIABLE_READ, Queue, Dst, BlockingRead, Count,
      ur_cast<char *>(GlobalVarPtr) + Offset, NumEventsInWaitList,
      EventWaitList, Event, PreferCopyEngine);
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelCreate(
    ur_program_handle_t Program, ///< [in] handle of the program instance
    const char *KernelName,      ///< [in] pointer to null-terminated string.
    ur_kernel_handle_t
        *RetKernel ///< [out] pointer to handle of kernel object created.
) {
  std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
  if (Program->State != ur_program_handle_t_::state::Exe) {
    return UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE;
  }

  ZeStruct<ze_kernel_desc_t> ZeKernelDesc;
  ZeKernelDesc.flags = 0;
  ZeKernelDesc.pKernelName = KernelName;

  ze_kernel_handle_t ZeKernel;
  ZE2UR_CALL(zeKernelCreate, (Program->ZeModule, &ZeKernelDesc, &ZeKernel));

  try {
    ur_kernel_handle_t_ *UrKernel =
        new ur_kernel_handle_t_(ZeKernel, true, Program);
    *RetKernel = reinterpret_cast<ur_kernel_handle_t>(UrKernel);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  UR_CALL((*RetKernel)->initialize());

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgValue(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex, ///< [in] argument index in range [0, num args - 1]
    size_t ArgSize,    ///< [in] size of argument type
    const ur_kernel_arg_value_properties_t
        *Properties, ///< [in][optional] argument properties
    const void
        *PArgValue ///< [in] argument value represented as matching arg type.
) {
  std::ignore = Properties;

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

  std::scoped_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  ZE2UR_CALL(zeKernelSetArgumentValue,
             (Kernel->ZeKernel, ArgIndex, ArgSize, PArgValue));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgLocal(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex, ///< [in] argument index in range [0, num args - 1]
    size_t ArgSize,    ///< [in] size of the local buffer to be allocated by the
                       ///< runtime
    const ur_kernel_arg_local_properties_t
        *Properties ///< [in][optional] argument properties
) {
  std::ignore = Properties;

  UR_CALL(urKernelSetArgValue(Kernel, ArgIndex, ArgSize, nullptr, nullptr));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetInfo(
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
      std::string &KernelName = *Kernel->ZeKernelName.operator->();
      return ReturnValue(static_cast<const char *>(KernelName.c_str()));
    } catch (const std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
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
      auto Res = ReturnValue(attributes);
      delete[] attributes;
      return Res;
    } catch (const std::bad_alloc &) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
  default:
    urPrint("Unsupported ParamName in urKernelGetInfo: ParamName=%d(0x%x)\n",
            ParamName, ParamName);
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetGroupInfo(
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
    // As of right now, L0 is missing API to query kernel and device specific
    // max work group size.
    return ReturnValue(
        uint64_t{Device->ZeDeviceComputeProperties->maxTotalGroupSize});
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
  default: {
    urPrint("Unknown ParamName in urKernelGetGroupInfo: ParamName=%d(0x%x)\n",
            ParamName, ParamName);
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetSubGroupInfo(
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

UR_APIEXPORT ur_result_t UR_APICALL urKernelRetain(
    ur_kernel_handle_t Kernel ///< [in] handle for the Kernel to retain
) {
  Kernel->RefCount.increment();

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelRelease(
    ur_kernel_handle_t Kernel ///< [in] handle for the Kernel to release
) {
  if (!Kernel->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  auto KernelProgram = Kernel->Program;
  if (Kernel->OwnNativeHandle) {
    auto ZeResult = ZE_CALL_NOCHECK(zeKernelDestroy, (Kernel->ZeKernel));
    // Gracefully handle the case that L0 was already unloaded.
    if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
      return ze2urResult(ZeResult);
  }
  if (IndirectAccessTrackingEnabled) {
    UR_CALL(urContextRelease(KernelProgram->Context));
  }
  // do a release on the program this kernel was part of
  UR_CALL(urProgramRelease(KernelProgram));
  delete Kernel;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgPointer(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_pointer_properties_t
        *Properties,     ///< [in][optional] argument properties
    const void *ArgValue ///< [in][optional] SVM pointer to memory location
                         ///< holding the argument value. If null then argument
                         ///< value is considered null.
) {
  std::ignore = Properties;

  UR_CALL(urKernelSetArgValue(Kernel, ArgIndex, sizeof(const void *), nullptr,
                              ArgValue));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetExecInfo(
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
  if (PropName == UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS &&
      *(static_cast<const ur_bool_t *>(PropValue)) == true) {
    // The whole point for users really was to not need to know anything
    // about the types of allocations kernel uses. So in DPC++ we always
    // just set all 3 modes for each kernel.
    ze_kernel_indirect_access_flags_t IndirectFlags =
        ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST |
        ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE |
        ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
    ZE2UR_CALL(zeKernelSetIndirectAccess, (Kernel->ZeKernel, IndirectFlags));
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
    ZE2UR_CALL(zeKernelSetCacheConfig, (Kernel->ZeKernel, ZeCacheConfig););
  } else {
    urPrint("urKernelSetExecInfo: unsupported ParamName\n");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgSampler(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t ArgIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_sampler_properties_t
        *Properties,             ///< [in][optional] argument properties
    ur_sampler_handle_t ArgValue ///< [in] handle of Sampler object.
) {
  std::ignore = Properties;
  std::scoped_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  ZE2UR_CALL(zeKernelSetArgumentValue, (Kernel->ZeKernel, ArgIndex,
                                        sizeof(void *), &ArgValue->ZeSampler));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgMemObj(
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

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetNativeHandle(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel.
    ur_native_handle_t
        *NativeKernel ///< [out] a pointer to the native handle of the kernel.
) {
  std::shared_lock<ur_shared_mutex> Guard(Kernel->Mutex);

  *NativeKernel = reinterpret_cast<ur_native_handle_t>(Kernel->ZeKernel);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelCreateWithNativeHandle(
    ur_native_handle_t NativeKernel, ///< [in] the native handle of the kernel.
    ur_context_handle_t Context,     ///< [in] handle of the context object
    ur_program_handle_t Program,
    const ur_kernel_native_properties_t *Properties,
    ur_kernel_handle_t *
        RetKernel ///< [out] pointer to the handle of the kernel object created.
) {
  ze_kernel_handle_t ZeKernel = ur_cast<ze_kernel_handle_t>(NativeKernel);
  ur_kernel_handle_t_ *Kernel = nullptr;
  try {
    Kernel = new ur_kernel_handle_t_(ZeKernel, Properties->isNativeHandleOwned,
                                     Context);
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

ur_result_t ur_kernel_handle_t_::initialize() {
  // Retain the program and context to show it's used by this kernel.
  UR_CALL(urProgramRetain(Program));

  if (IndirectAccessTrackingEnabled)
    // TODO: do piContextRetain without the guard
    UR_CALL(urContextRetain(Program->Context));

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

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetSpecializationConstants(
    ur_kernel_handle_t Kernel, ///< [in] handle of the kernel object
    uint32_t Count, ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t
        *SpecConstants ///< [in] array of specialization constant value
                       ///< descriptions
) {
  std::ignore = Kernel;
  std::ignore = Count;
  std::ignore = SpecConstants;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
