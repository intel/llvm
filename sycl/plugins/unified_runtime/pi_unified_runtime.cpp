//===--- pi_unified_runtime.cpp - Unified Runtime PI Plugin  ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstring>

// #include "ur/adapters/level_zero/ur_level_zero_common.hpp"
#include <pi2ur.hpp>
#include <pi_unified_runtime.hpp>

// Stub function to where all not yet supported PI API are bound
static void DieUnsupported() {
  die("Unified Runtime: functionality is not supported");
}

// Adapters may be released by piTearDown being called, or the global dtors
// being called first. Handle releasing the adapters exactly once.
static void releaseAdapters(std::vector<ur_adapter_handle_t> &Vec) noexcept {
  static std::once_flag ReleaseFlag{};
  try {
    std::call_once(ReleaseFlag, [&]() {
      for (auto Adapter : Vec) {
        urAdapterRelease(Adapter);
      }
      urLoaderTearDown();
    });
  } catch (...) {
    // Ignore any potential exceptions on teardown. Worst case scenario
    // this just leaks some memory on exit.
  }
}

struct AdapterHolder {
  ~AdapterHolder() { releaseAdapters(Vec); }
  std::vector<ur_adapter_handle_t> Vec{};
} Adapters;

// All PI API interfaces are C interfaces
extern "C" {
__SYCL_EXPORT pi_result piPlatformsGet(pi_uint32 NumEntries,
                                       pi_platform *Platforms,
                                       pi_uint32 *NumPlatforms) {
  // Get all the platforms from all available adapters
  urPlatformGet(Adapters.Vec.data(), static_cast<uint32_t>(Adapters.Vec.size()),
                NumEntries, reinterpret_cast<ur_platform_handle_t *>(Platforms),
                NumPlatforms);

  return PI_SUCCESS;
}

__SYCL_EXPORT pi_result piPlatformGetInfo(pi_platform Platform,
                                          pi_platform_info ParamName,
                                          size_t ParamValueSize,
                                          void *ParamValue,
                                          size_t *ParamValueSizeRet) {
  return pi2ur::piPlatformGetInfo(Platform, ParamName, ParamValueSize,
                                  ParamValue, ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piDevicesGet(pi_platform Platform,
                                     pi_device_type DeviceType,
                                     pi_uint32 NumEntries, pi_device *Devices,
                                     pi_uint32 *NumDevices) {
  return pi2ur::piDevicesGet(Platform, DeviceType, NumEntries, Devices,
                             NumDevices);
}

__SYCL_EXPORT pi_result piDeviceRetain(pi_device Device) {
  return pi2ur::piDeviceRetain(Device);
}

__SYCL_EXPORT pi_result piDeviceRelease(pi_device Device) {
  return pi2ur::piDeviceRelease(Device);
}

__SYCL_EXPORT pi_result piDeviceGetInfo(pi_device Device,
                                        pi_device_info ParamName,
                                        size_t ParamValueSize, void *ParamValue,
                                        size_t *ParamValueSizeRet) {
  return pi2ur::piDeviceGetInfo(Device, ParamName, ParamValueSize, ParamValue,
                                ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piDevicePartition(
    pi_device Device, const pi_device_partition_property *Properties,
    pi_uint32 NumDevices, pi_device *OutDevices, pi_uint32 *OutNumDevices) {
  return pi2ur::piDevicePartition(Device, Properties, NumDevices, OutDevices,
                                  OutNumDevices);
}

// Stub for the not yet supported API
__SYCL_EXPORT pi_result piextDeviceSelectBinary(pi_device Device,
                                                pi_device_binary *Binaries,
                                                pi_uint32 NumBinaries,
                                                pi_uint32 *SelectedBinaryInd) {
  return pi2ur::piextDeviceSelectBinary(Device, Binaries, NumBinaries,
                                        SelectedBinaryInd);
}

__SYCL_EXPORT pi_result
piContextCreate(const pi_context_properties *Properties, pi_uint32 NumDevices,
                const pi_device *Devices,
                void (*PFnNotify)(const char *ErrInfo, const void *PrivateInfo,
                                  size_t CB, void *UserData),
                void *UserData, pi_context *RetContext) {
  return pi2ur::piContextCreate(Properties, NumDevices, Devices, PFnNotify,
                                UserData, RetContext);
}

__SYCL_EXPORT pi_result piContextGetInfo(pi_context Context,
                                         pi_context_info ParamName,
                                         size_t ParamValueSize,
                                         void *ParamValue,
                                         size_t *ParamValueSizeRet) {
  return pi2ur::piContextGetInfo(Context, ParamName, ParamValueSize, ParamValue,
                                 ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piContextRelease(pi_context Context) {
  return pi2ur::piContextRelease(Context);
}

__SYCL_EXPORT pi_result piQueueCreate(pi_context Context, pi_device Device,
                                      pi_queue_properties Flags,
                                      pi_queue *Queue) {
  return pi2ur::piQueueCreate(Context, Device, Flags, Queue);
}

__SYCL_EXPORT pi_result piextQueueCreate(pi_context Context, pi_device Device,
                                         pi_queue_properties *Properties,
                                         pi_queue *Queue) {
  return pi2ur::piextQueueCreate(Context, Device, Properties, Queue);
}

__SYCL_EXPORT pi_result piQueueRelease(pi_queue Queue) {
  return pi2ur::piQueueRelease(Queue);
}

__SYCL_EXPORT pi_result piProgramCreate(pi_context Context, const void *ILBytes,
                                        size_t Length, pi_program *Program) {
  return pi2ur::piProgramCreate(Context, ILBytes, Length, Program);
}

__SYCL_EXPORT pi_result piProgramBuild(
    pi_program Program, pi_uint32 NumDevices, const pi_device *DeviceList,
    const char *Options, void (*PFnNotify)(pi_program Program, void *UserData),
    void *UserData) {
  return pi2ur::piProgramBuild(Program, NumDevices, DeviceList, Options,
                               PFnNotify, UserData);
}

__SYCL_EXPORT pi_result piextProgramSetSpecializationConstant(
    pi_program Prog, pi_uint32 SpecID, size_t Size, const void *SpecValue) {
  return pi2ur::piextProgramSetSpecializationConstant(Prog, SpecID, Size,
                                                      SpecValue);
}

__SYCL_EXPORT pi_result
piProgramLink(pi_context Context, pi_uint32 NumDevices,
              const pi_device *DeviceList, const char *Options,
              pi_uint32 NumInputPrograms, const pi_program *InputPrograms,
              void (*PFnNotify)(pi_program Program, void *UserData),
              void *UserData, pi_program *RetProgram) {
  return pi2ur::piProgramLink(Context, NumDevices, DeviceList, Options,
                              NumInputPrograms, InputPrograms, PFnNotify,
                              UserData, RetProgram);
}

__SYCL_EXPORT pi_result piKernelCreate(pi_program Program,
                                       const char *KernelName,
                                       pi_kernel *RetKernel) {
  return pi2ur::piKernelCreate(Program, KernelName, RetKernel);
}

// Special version of piKernelSetArg to accept pi_mem.
__SYCL_EXPORT pi_result piextKernelSetArgMemObj(
    pi_kernel Kernel, pi_uint32 ArgIndex,
    const pi_mem_obj_property *ArgProperties, const pi_mem *ArgValue) {

  return pi2ur::piextKernelSetArgMemObj(Kernel, ArgIndex, ArgProperties,
                                        ArgValue);
}

__SYCL_EXPORT pi_result piKernelSetArg(pi_kernel Kernel, pi_uint32 ArgIndex,
                                       size_t ArgSize, const void *ArgValue) {

  return pi2ur::piKernelSetArg(Kernel, ArgIndex, ArgSize, ArgValue);
}

__SYCL_EXPORT pi_result piKernelGetGroupInfo(pi_kernel Kernel, pi_device Device,
                                             pi_kernel_group_info ParamName,
                                             size_t ParamValueSize,
                                             void *ParamValue,
                                             size_t *ParamValueSizeRet) {
  return pi2ur::piKernelGetGroupInfo(Kernel, Device, ParamName, ParamValueSize,
                                     ParamValue, ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piMemBufferCreate(pi_context Context,
                                          pi_mem_flags Flags, size_t Size,
                                          void *HostPtr, pi_mem *RetMem,
                                          const pi_mem_properties *properties) {

  return pi2ur::piMemBufferCreate(Context, Flags, Size, HostPtr, RetMem,
                                  properties);
}

__SYCL_EXPORT pi_result piextUSMHostAlloc(void **ResultPtr, pi_context Context,
                                          pi_usm_mem_properties *Properties,
                                          size_t Size, pi_uint32 Alignment) {

  return pi2ur::piextUSMHostAlloc(ResultPtr, Context, Properties, Size,
                                  Alignment);
}

__SYCL_EXPORT pi_result piMemGetInfo(pi_mem Mem, pi_mem_info ParamName,
                                     size_t ParamValueSize, void *ParamValue,
                                     size_t *ParamValueSizeRet) {
  return pi2ur::piMemGetInfo(Mem, ParamName, ParamValueSize, ParamValue,
                             ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piMemImageCreate(pi_context Context, pi_mem_flags Flags,
                                         const pi_image_format *ImageFormat,
                                         const pi_image_desc *ImageDesc,
                                         void *HostPtr, pi_mem *RetImage) {

  return pi2ur::piMemImageCreate(Context, Flags, ImageFormat, ImageDesc,
                                 HostPtr, RetImage);
}

__SYCL_EXPORT pi_result piMemBufferPartition(
    pi_mem Buffer, pi_mem_flags Flags, pi_buffer_create_type BufferCreateType,
    void *BufferCreateInfo, pi_mem *RetMem) {
  return pi2ur::piMemBufferPartition(Buffer, Flags, BufferCreateType,
                                     BufferCreateInfo, RetMem);
}

__SYCL_EXPORT pi_result piextMemGetNativeHandle(
    pi_mem Mem, pi_device Dev, pi_native_handle *NativeHandle) {
  return pi2ur::piextMemGetNativeHandle(Mem, Dev, NativeHandle);
}

__SYCL_EXPORT pi_result
piEnqueueMemImageCopy(pi_queue Queue, pi_mem SrcImage, pi_mem DstImage,
                      pi_image_offset SrcOrigin, pi_image_offset DstOrigin,
                      pi_image_region Region, pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *Event) {
  return pi2ur::piEnqueueMemImageCopy(Queue, SrcImage, DstImage, SrcOrigin,
                                      DstOrigin, Region, NumEventsInWaitList,
                                      EventWaitList, Event);
}

__SYCL_EXPORT pi_result piextMemCreateWithNativeHandle(
    pi_native_handle NativeHandle, pi_context Context, bool ownNativeHandle,
    pi_mem *Mem) {
  return pi2ur::piextMemCreateWithNativeHandle(NativeHandle, Context,
                                               ownNativeHandle, Mem);
}

__SYCL_EXPORT pi_result piEnqueueKernelLaunch(
    pi_queue Queue, pi_kernel Kernel, pi_uint32 WorkDim,
    const size_t *GlobalWorkOffset, const size_t *GlobalWorkSize,
    const size_t *LocalWorkSize, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *OutEvent) {

  return pi2ur::piEnqueueKernelLaunch(
      Queue, Kernel, WorkDim, GlobalWorkOffset, GlobalWorkSize, LocalWorkSize,
      NumEventsInWaitList, EventWaitList, OutEvent);
}

__SYCL_EXPORT pi_result piextEnqueueKernelLaunchCustom(
    pi_queue Queue, pi_kernel Kernel, pi_uint32 WorkDim,
    const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
    pi_uint32 NumPropsInLaunchPropList,
    const pi_launch_property *LaunchPropList, pi_uint32 NumEventsInWaitList,
    const pi_event *EventsWaitList, pi_event *OutEvent) {

  return pi2ur::piextEnqueueKernelLaunchCustom(
      Queue, Kernel, WorkDim, GlobalWorkSize, LocalWorkSize,
      NumPropsInLaunchPropList, LaunchPropList, NumEventsInWaitList,
      EventsWaitList, OutEvent);
}

__SYCL_EXPORT pi_result piEnqueueMemImageWrite(
    pi_queue Queue, pi_mem Image, pi_bool BlockingWrite, pi_image_offset Origin,
    pi_image_region Region, size_t InputRowPitch, size_t InputSlicePitch,
    const void *Ptr, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {

  return pi2ur::piEnqueueMemImageWrite(
      Queue, Image, BlockingWrite, Origin, Region, InputRowPitch,
      InputSlicePitch, Ptr, NumEventsInWaitList, EventWaitList, Event);
}

__SYCL_EXPORT pi_result piEnqueueMemImageRead(
    pi_queue Queue, pi_mem Image, pi_bool BlockingRead, pi_image_offset Origin,
    pi_image_region Region, size_t RowPitch, size_t SlicePitch, void *Ptr,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *Event) {
  return pi2ur::piEnqueueMemImageRead(
      Queue, Image, BlockingRead, Origin, Region, RowPitch, SlicePitch, Ptr,
      NumEventsInWaitList, EventWaitList, Event);
}

__SYCL_EXPORT pi_result piextKernelCreateWithNativeHandle(
    pi_native_handle NativeHandle, pi_context Context, pi_program Program,
    bool OwnNativeHandle, pi_kernel *Kernel) {

  return pi2ur::piextKernelCreateWithNativeHandle(
      NativeHandle, Context, Program, OwnNativeHandle, Kernel);
}

__SYCL_EXPORT pi_result piEnqueueMemUnmap(pi_queue Queue, pi_mem Mem,
                                          void *MappedPtr,
                                          pi_uint32 NumEventsInWaitList,
                                          const pi_event *EventWaitList,
                                          pi_event *OutEvent) {

  return pi2ur::piEnqueueMemUnmap(Queue, Mem, MappedPtr, NumEventsInWaitList,
                                  EventWaitList, OutEvent);
}

__SYCL_EXPORT pi_result piEventsWait(pi_uint32 NumEvents,
                                     const pi_event *EventList) {

  return pi2ur::piEventsWait(NumEvents, EventList);
}

__SYCL_EXPORT pi_result piQueueFinish(pi_queue Queue) {
  return pi2ur::piQueueFinish(Queue);
}

__SYCL_EXPORT pi_result piEventGetInfo(pi_event Event, pi_event_info ParamName,
                                       size_t ParamValueSize, void *ParamValue,
                                       size_t *ParamValueSizeRet) {
  return pi2ur::piEventGetInfo(Event, ParamName, ParamValueSize, ParamValue,
                               ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piEnqueueMemBufferMap(
    pi_queue Queue, pi_mem Mem, pi_bool BlockingMap, pi_map_flags MapFlags,
    size_t Offset, size_t Size, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *OutEvent, void **RetMap) {

  return pi2ur::piEnqueueMemBufferMap(Queue, Mem, BlockingMap, MapFlags, Offset,
                                      Size, NumEventsInWaitList, EventWaitList,
                                      OutEvent, RetMap);
}

__SYCL_EXPORT pi_result piEnqueueMemBufferFill(
    pi_queue Queue, pi_mem Buffer, const void *Pattern, size_t PatternSize,
    size_t Offset, size_t Size, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {
  return pi2ur::piEnqueueMemBufferFill(Queue, Buffer, Pattern, PatternSize,
                                       Offset, Size, NumEventsInWaitList,
                                       EventWaitList, Event);
}

__SYCL_EXPORT pi_result piextUSMDeviceAlloc(void **ResultPtr,
                                            pi_context Context,
                                            pi_device Device,
                                            pi_usm_mem_properties *Properties,
                                            size_t Size, pi_uint32 Alignment) {

  return pi2ur::piextUSMDeviceAlloc(ResultPtr, Context, Device, Properties,
                                    Size, Alignment);
}

__SYCL_EXPORT pi_result piKernelRetain(pi_kernel Kernel) {
  return pi2ur::piKernelRetain(Kernel);
}

__SYCL_EXPORT pi_result piKernelRelease(pi_kernel Kernel) {

  return pi2ur::piKernelRelease(Kernel);
}

__SYCL_EXPORT pi_result piProgramRelease(pi_program Program) {
  return pi2ur::piProgramRelease(Program);
}

__SYCL_EXPORT pi_result piextUSMSharedAlloc(void **ResultPtr,
                                            pi_context Context,
                                            pi_device Device,
                                            pi_usm_mem_properties *Properties,
                                            size_t Size, pi_uint32 Alignment) {

  return pi2ur::piextUSMSharedAlloc(ResultPtr, Context, Device, Properties,
                                    Size, Alignment);
}

__SYCL_EXPORT pi_result piextUSMPitchedAlloc(
    void **ResultPtr, size_t *ResultPitch, pi_context Context, pi_device Device,
    pi_usm_mem_properties *Properties, size_t WidthInBytes, size_t Height,
    unsigned int ElementSizeBytes) {

  return pi2ur::piextUSMPitchedAlloc(ResultPtr, ResultPitch, Context, Device,
                                     Properties, WidthInBytes, Height,
                                     ElementSizeBytes);
}

__SYCL_EXPORT pi_result piextUSMFree(pi_context Context, void *Ptr) {
  return pi2ur::piextUSMFree(Context, Ptr);
}

__SYCL_EXPORT pi_result piextUSMImport(const void *HostPtr, size_t Size,
                                       pi_context Context) {
  return pi2ur::piextUSMImport(HostPtr, Size, Context);
}

__SYCL_EXPORT pi_result piextUSMRelease(const void *HostPtr,
                                        pi_context Context) {
  return pi2ur::piextUSMRelease(HostPtr, Context);
}

__SYCL_EXPORT pi_result piContextRetain(pi_context Context) {
  return pi2ur::piContextRetain(Context);
}

__SYCL_EXPORT pi_result piextKernelSetArgPointer(pi_kernel Kernel,
                                                 pi_uint32 ArgIndex,
                                                 size_t ArgSize,
                                                 const void *ArgValue) {
  return pi2ur::piextKernelSetArgPointer(Kernel, ArgIndex, ArgSize, ArgValue);
}

// Special version of piKernelSetArg to accept pi_sampler.
__SYCL_EXPORT pi_result piextKernelSetArgSampler(pi_kernel Kernel,
                                                 pi_uint32 ArgIndex,
                                                 const pi_sampler *ArgValue) {

  return pi2ur::piextKernelSetArgSampler(Kernel, ArgIndex, ArgValue);
}

__SYCL_EXPORT pi_result piKernelGetSubGroupInfo(
    pi_kernel Kernel, pi_device Device, pi_kernel_sub_group_info ParamName,
    size_t InputValueSize, const void *InputValue, size_t ParamValueSize,
    void *ParamValue, size_t *ParamValueSizeRet) {

  return pi2ur::piKernelGetSubGroupInfo(
      Kernel, Device, ParamName, InputValueSize, InputValue, ParamValueSize,
      ParamValue, ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piQueueGetInfo(pi_queue Queue, pi_queue_info ParamName,
                                       size_t ParamValueSize, void *ParamValue,
                                       size_t *ParamValueSizeRet) {

  return pi2ur::piQueueGetInfo(Queue, ParamName, ParamValueSize, ParamValue,
                               ParamValueSizeRet);
}

/// USM Fill API
///
/// \param queue is the queue to submit to
/// \param ptr is the ptr to fill
/// \param pattern is the ptr with the bytes of the pattern to set
/// \param patternSize is the size in bytes of the pattern to set
/// \param count is the size in bytes to fill
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
__SYCL_EXPORT pi_result piextUSMEnqueueFill(pi_queue Queue, void *Ptr,
                                            const void *Pattern,
                                            size_t PatternSize, size_t Count,
                                            pi_uint32 NumEventsInWaitlist,
                                            const pi_event *EventsWaitlist,
                                            pi_event *Event) {
  return pi2ur::piextUSMEnqueueFill(Queue, Ptr, Pattern, PatternSize, Count,
                                    NumEventsInWaitlist, EventsWaitlist, Event);
}

__SYCL_EXPORT pi_result piEnqueueMemBufferCopyRect(
    pi_queue Queue, pi_mem SrcMem, pi_mem DstMem, pi_buff_rect_offset SrcOrigin,
    pi_buff_rect_offset DstOrigin, pi_buff_rect_region Region,
    size_t SrcRowPitch, size_t SrcSlicePitch, size_t DstRowPitch,
    size_t DstSlicePitch, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {

  return pi2ur::piEnqueueMemBufferCopyRect(
      Queue, SrcMem, DstMem, SrcOrigin, DstOrigin, Region, SrcRowPitch,
      SrcSlicePitch, DstRowPitch, DstSlicePitch, NumEventsInWaitList,
      EventWaitList, Event);
}

__SYCL_EXPORT pi_result piEnqueueMemBufferCopy(pi_queue Queue, pi_mem SrcMem,
                                               pi_mem DstMem, size_t SrcOffset,
                                               size_t DstOffset, size_t Size,
                                               pi_uint32 NumEventsInWaitList,
                                               const pi_event *EventWaitList,
                                               pi_event *Event) {
  return pi2ur::piEnqueueMemBufferCopy(Queue, SrcMem, DstMem, SrcOffset,
                                       DstOffset, Size, NumEventsInWaitList,
                                       EventWaitList, Event);
}

__SYCL_EXPORT pi_result piextUSMEnqueueMemcpy(pi_queue Queue, pi_bool Blocking,
                                              void *DstPtr, const void *SrcPtr,
                                              size_t Size,
                                              pi_uint32 NumEventsInWaitlist,
                                              const pi_event *EventsWaitlist,
                                              pi_event *Event) {

  return pi2ur::piextUSMEnqueueMemcpy(Queue, Blocking, DstPtr, SrcPtr, Size,
                                      NumEventsInWaitlist, EventsWaitlist,
                                      Event);
}

__SYCL_EXPORT pi_result piEnqueueMemBufferWriteRect(
    pi_queue Queue, pi_mem Buffer, pi_bool BlockingWrite,
    pi_buff_rect_offset BufferOffset, pi_buff_rect_offset HostOffset,
    pi_buff_rect_region Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, const void *Ptr,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *Event) {

  return pi2ur::piEnqueueMemBufferWriteRect(
      Queue, Buffer, BlockingWrite, BufferOffset, HostOffset, Region,
      BufferRowPitch, BufferSlicePitch, HostRowPitch, HostSlicePitch, Ptr,
      NumEventsInWaitList, EventWaitList, Event);
}

__SYCL_EXPORT pi_result piEnqueueMemBufferWrite(
    pi_queue Queue, pi_mem Buffer, pi_bool BlockingWrite, size_t Offset,
    size_t Size, const void *Ptr, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {

  return pi2ur::piEnqueueMemBufferWrite(Queue, Buffer, BlockingWrite, Offset,
                                        Size, Ptr, NumEventsInWaitList,
                                        EventWaitList, Event);
}

__SYCL_EXPORT pi_result piEnqueueMemBufferReadRect(
    pi_queue Queue, pi_mem Buffer, pi_bool BlockingRead,
    pi_buff_rect_offset BufferOffset, pi_buff_rect_offset HostOffset,
    pi_buff_rect_region Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, void *Ptr,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *Event) {

  return pi2ur::piEnqueueMemBufferReadRect(
      Queue, Buffer, BlockingRead, BufferOffset, HostOffset, Region,
      BufferRowPitch, BufferSlicePitch, HostRowPitch, HostSlicePitch, Ptr,
      NumEventsInWaitList, EventWaitList, Event);
}

__SYCL_EXPORT pi_result piEnqueueMemBufferRead(
    pi_queue Queue, pi_mem Src, pi_bool BlockingRead, size_t Offset,
    size_t Size, void *Dst, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {

  return pi2ur::piEnqueueMemBufferRead(Queue, Src, BlockingRead, Offset, Size,
                                       Dst, NumEventsInWaitList, EventWaitList,
                                       Event);
}

__SYCL_EXPORT pi_result piEnqueueEventsWaitWithBarrier(
    pi_queue Queue, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *OutEvent) {

  return pi2ur::piEnqueueEventsWaitWithBarrier(Queue, NumEventsInWaitList,
                                               EventWaitList, OutEvent);
}

__SYCL_EXPORT pi_result piEnqueueEventsWait(pi_queue Queue,
                                            pi_uint32 NumEventsInWaitList,
                                            const pi_event *EventWaitList,
                                            pi_event *OutEvent) {

  return pi2ur::piEnqueueEventsWait(Queue, NumEventsInWaitList, EventWaitList,
                                    OutEvent);
}

__SYCL_EXPORT pi_result
piextEventGetNativeHandle(pi_event Event, pi_native_handle *NativeHandle) {

  return pi2ur::piextEventGetNativeHandle(Event, NativeHandle);
}

__SYCL_EXPORT pi_result piEventGetProfilingInfo(pi_event Event,
                                                pi_profiling_info ParamName,
                                                size_t ParamValueSize,
                                                void *ParamValue,
                                                size_t *ParamValueSizeRet) {

  return pi2ur::piEventGetProfilingInfo(Event, ParamName, ParamValueSize,
                                        ParamValue, ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piProgramRetain(pi_program Program) {
  return pi2ur::piProgramRetain(Program);
}

__SYCL_EXPORT pi_result piKernelSetExecInfo(pi_kernel Kernel,
                                            pi_kernel_exec_info ParamName,
                                            size_t ParamValueSize,
                                            const void *ParamValue) {

  return pi2ur::piKernelSetExecInfo(Kernel, ParamName, ParamValueSize,
                                    ParamValue);
}

__SYCL_EXPORT pi_result piKernelGetInfo(pi_kernel Kernel,
                                        pi_kernel_info ParamName,
                                        size_t ParamValueSize, void *ParamValue,
                                        size_t *ParamValueSizeRet) {
  return pi2ur::piKernelGetInfo(Kernel, ParamName, ParamValueSize, ParamValue,
                                ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piQueueRetain(pi_queue Queue) {
  return pi2ur::piQueueRetain(Queue);
}

__SYCL_EXPORT pi_result piQueueFlush(pi_queue Queue) {
  return pi2ur::piQueueFlush(Queue);
}

__SYCL_EXPORT pi_result piMemRetain(pi_mem Mem) {
  return pi2ur::piMemRetain(Mem);
}

__SYCL_EXPORT pi_result piProgramCreateWithBinary(
    pi_context Context, pi_uint32 NumDevices, const pi_device *DeviceList,
    const size_t *Lengths, const unsigned char **Binaries,
    size_t NumMetadataEntries, const pi_device_binary_property *Metadata,
    pi_int32 *BinaryStatus, pi_program *Program) {

  return pi2ur::piProgramCreateWithBinary(Context, NumDevices, DeviceList,
                                          Lengths, Binaries, NumMetadataEntries,
                                          Metadata, BinaryStatus, Program);
}

__SYCL_EXPORT pi_result piProgramGetInfo(pi_program Program,
                                         pi_program_info ParamName,
                                         size_t ParamValueSize,
                                         void *ParamValue,
                                         size_t *ParamValueSizeRet) {

  return pi2ur::piProgramGetInfo(Program, ParamName, ParamValueSize, ParamValue,
                                 ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piProgramCompile(
    pi_program Program, pi_uint32 NumDevices, const pi_device *DeviceList,
    const char *Options, pi_uint32 NumInputHeaders,
    const pi_program *InputHeaders, const char **HeaderIncludeNames,
    void (*PFnNotify)(pi_program Program, void *UserData), void *UserData) {

  return pi2ur::piProgramCompile(Program, NumDevices, DeviceList, Options,
                                 NumInputHeaders, InputHeaders,
                                 HeaderIncludeNames, PFnNotify, UserData);
}

__SYCL_EXPORT pi_result piProgramGetBuildInfo(
    pi_program Program, pi_device Device, pi_program_build_info ParamName,
    size_t ParamValueSize, void *ParamValue, size_t *ParamValueSizeRet) {

  return pi2ur::piProgramGetBuildInfo(Program, Device, ParamName,
                                      ParamValueSize, ParamValue,
                                      ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piEventCreate(pi_context Context, pi_event *RetEvent) {

  return pi2ur::piEventCreate(Context, RetEvent);
}

__SYCL_EXPORT pi_result piEventSetCallback(
    pi_event Event, pi_int32 CommandExecCallbackType,
    void (*PFnNotify)(pi_event Event, pi_int32 EventCommandStatus,
                      void *UserData),
    void *UserData) {
  return pi2ur::piEventSetCallback(Event, CommandExecCallbackType, PFnNotify,
                                   UserData);
}

__SYCL_EXPORT pi_result piEventSetStatus(pi_event Event,
                                         pi_int32 ExecutionStatus) {
  return pi2ur::piEventSetStatus(Event, ExecutionStatus);
}

__SYCL_EXPORT pi_result piEventRetain(pi_event Event) {
  return pi2ur::piEventRetain(Event);
}

__SYCL_EXPORT pi_result piEventRelease(pi_event Event) {
  return pi2ur::piEventRelease(Event);
}

__SYCL_EXPORT pi_result piextEventCreateWithNativeHandle(
    pi_native_handle NativeHandle, pi_context Context, bool OwnNativeHandle,
    pi_event *Event) {
  return pi2ur::piextEventCreateWithNativeHandle(NativeHandle, Context,
                                                 OwnNativeHandle, Event);
}

__SYCL_EXPORT pi_result piEnqueueTimestampRecordingExp(
    pi_queue Queue, pi_bool Blocking, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {
  return pi2ur::piEnqueueTimestampRecordingExp(
      Queue, Blocking, NumEventsInWaitList, EventWaitList, Event);
}

__SYCL_EXPORT pi_result piEnqueueMemImageFill(
    pi_queue Queue, pi_mem Image, const void *FillColor, const size_t *Origin,
    const size_t *Region, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {

  return pi2ur::piEnqueueMemImageFill(Queue, Image, FillColor, Origin, Region,
                                      NumEventsInWaitList, EventWaitList,
                                      Event);
}

__SYCL_EXPORT pi_result piextPlatformGetNativeHandle(
    pi_platform Platform, pi_native_handle *NativeHandle) {

  return pi2ur::piextPlatformGetNativeHandle(Platform, NativeHandle);
}

__SYCL_EXPORT pi_result piextPlatformCreateWithNativeHandle(
    pi_native_handle NativeHandle, pi_platform *Platform) {

  return pi2ur::piextPlatformCreateWithNativeHandle(NativeHandle, Platform);
}

__SYCL_EXPORT pi_result
piextDeviceGetNativeHandle(pi_device Device, pi_native_handle *NativeHandle) {

  return pi2ur::piextDeviceGetNativeHandle(Device, NativeHandle);
}

__SYCL_EXPORT pi_result piextDeviceCreateWithNativeHandle(
    pi_native_handle NativeHandle, pi_platform Platform, pi_device *Device) {

  return pi2ur::piextDeviceCreateWithNativeHandle(NativeHandle, Platform,
                                                  Device);
}

// FIXME: Dummy implementation to prevent link fail
__SYCL_EXPORT pi_result piextContextSetExtendedDeleter(
    pi_context Context, pi_context_extended_deleter Function, void *UserData) {
  return pi2ur::piextContextSetExtendedDeleter(Context, Function, UserData);
}

__SYCL_EXPORT pi_result piextContextGetNativeHandle(
    pi_context Context, pi_native_handle *NativeHandle) {

  return pi2ur::piextContextGetNativeHandle(Context, NativeHandle);
}

__SYCL_EXPORT pi_result piextContextCreateWithNativeHandle(
    pi_native_handle NativeHandle, pi_uint32 NumDevices,
    const pi_device *Devices, bool OwnNativeHandle, pi_context *RetContext) {
  return pi2ur::piextContextCreateWithNativeHandle(
      NativeHandle, NumDevices, Devices, OwnNativeHandle, RetContext);
}

__SYCL_EXPORT pi_result piextQueueGetNativeHandle(
    pi_queue Queue, pi_native_handle *NativeHandle, int32_t *NativeHandleDesc) {
  return pi2ur::piextQueueGetNativeHandle(Queue, NativeHandle,
                                          NativeHandleDesc);
}

__SYCL_EXPORT pi_result piextQueueCreateWithNativeHandle(
    pi_native_handle NativeHandle, int32_t NativeHandleDesc, pi_context Context,
    pi_device Device, bool OwnNativeHandle, pi_queue_properties *Properties,
    pi_queue *Queue) {
  return pi2ur::piextQueueCreateWithNativeHandle(
      NativeHandle, NativeHandleDesc, Context, Device, OwnNativeHandle,
      Properties, Queue);
}

__SYCL_EXPORT pi_result piMemRelease(pi_mem Mem) {
  return pi2ur::piMemRelease(Mem);
}

__SYCL_EXPORT pi_result piextGetDeviceFunctionPointer(
    pi_device Device, pi_program Program, const char *FunctionName,
    pi_uint64 *FunctionPointerRet) {

  return pi2ur::piextGetDeviceFunctionPointer(Device, Program, FunctionName,
                                              FunctionPointerRet);
}

__SYCL_EXPORT pi_result piextGetGlobalVariablePointer(
    pi_device Device, pi_program Program, const char *GlobalVariableName,
    size_t *GlobalVariableSize, void **GlobalVariablePointerRet) {

  return pi2ur::piextGetGlobalVariablePointer(
      Device, Program, GlobalVariableName, GlobalVariableSize,
      GlobalVariablePointerRet);
}

/// Hint to migrate memory to the device
///
/// @param Queue is the queue to submit to
/// @param Ptr points to the memory to migrate
/// @param Size is the number of bytes to migrate
/// @param Flags is a bitfield used to specify memory migration options
/// @param NumEventsInWaitlist is the number of events to wait on
/// @param EventsWaitlist is an array of events to wait on
/// @param Event is the event that represents this operation
__SYCL_EXPORT pi_result piextUSMEnqueuePrefetch(pi_queue Queue, const void *Ptr,
                                                size_t Size,
                                                pi_usm_migration_flags Flags,
                                                pi_uint32 NumEventsInWaitList,
                                                const pi_event *EventWaitList,
                                                pi_event *OutEvent) {

  return pi2ur::piextUSMEnqueuePrefetch(
      Queue, Ptr, Size, Flags, NumEventsInWaitList, EventWaitList, OutEvent);
}

/// USM memadvise API to govern behavior of automatic migration mechanisms
///
/// @param Queue is the queue to submit to
/// @param Ptr is the data to be advised
/// @param Length is the size in bytes of the meory to advise
/// @param Advice is device specific advice
/// @param Event is the event that represents this operation
///
__SYCL_EXPORT pi_result piextUSMEnqueueMemAdvise(pi_queue Queue,
                                                 const void *Ptr, size_t Length,
                                                 pi_mem_advice Advice,
                                                 pi_event *OutEvent) {

  return pi2ur::piextUSMEnqueueMemAdvise(Queue, Ptr, Length, Advice, OutEvent);
}

/// USM 2D Fill API
///
/// \param queue is the queue to submit to
/// \param ptr is the ptr to fill
/// \param pitch is the total width of the destination memory including padding
/// \param pattern is a pointer with the bytes of the pattern to set
/// \param pattern_size is the size in bytes of the pattern
/// \param width is width in bytes of each row to fill
/// \param height is height the columns to fill
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
__SYCL_EXPORT pi_result piextUSMEnqueueFill2D(pi_queue Queue, void *Ptr,
                                              size_t Pitch, size_t PatternSize,
                                              const void *Pattern, size_t Width,
                                              size_t Height,
                                              pi_uint32 NumEventsWaitList,
                                              const pi_event *EventsWaitList,
                                              pi_event *Event) {

  return pi2ur::piextUSMEnqueueFill2D(Queue, Ptr, Pitch, PatternSize, Pattern,
                                      Width, Height, NumEventsWaitList,
                                      EventsWaitList, Event);
}

/// USM 2D Memset API
///
/// \param queue is the queue to submit to
/// \param ptr is the ptr to fill
/// \param pitch is the total width of the destination memory including padding
/// \param pattern is a pointer with the bytes of the pattern to set
/// \param pattern_size is the size in bytes of the pattern
/// \param width is width in bytes of each row to fill
/// \param height is height the columns to fill
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
__SYCL_EXPORT pi_result piextUSMEnqueueMemset2D(pi_queue Queue, void *Ptr,
                                                size_t Pitch, int Value,
                                                size_t Width, size_t Height,
                                                pi_uint32 NumEventsWaitList,
                                                const pi_event *EventsWaitlist,
                                                pi_event *Event) {
  return pi2ur::piextUSMEnqueueMemset2D(Queue, Ptr, Pitch, Value, Width, Height,
                                        NumEventsWaitList, EventsWaitlist,
                                        Event);
}

/// API to query information about USM allocated pointers.
/// Valid Queries:
///   PI_MEM_ALLOC_TYPE returns host/device/shared pi_usm_type value
///   PI_MEM_ALLOC_BASE_PTR returns the base ptr of an allocation if
///                         the queried pointer fell inside an allocation.
///                         Result must fit in void *
///   PI_MEM_ALLOC_SIZE returns how big the queried pointer's
///                     allocation is in bytes. Result is a size_t.
///   PI_MEM_ALLOC_DEVICE returns the pi_device this was allocated against
///
/// @param Context is the pi_context
/// @param Ptr is the pointer to query
/// @param ParamName is the type of query to perform
/// @param ParamValueSize is the size of the result in bytes
/// @param ParamValue is the result
/// @param ParamValueRet is how many bytes were written
__SYCL_EXPORT pi_result piextUSMGetMemAllocInfo(
    pi_context Context, const void *Ptr, pi_mem_alloc_info ParamName,
    size_t ParamValueSize, void *ParamValue, size_t *ParamValueSizeRet) {
  return pi2ur::piextUSMGetMemAllocInfo(Context, Ptr, ParamName, ParamValueSize,
                                        ParamValue, ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piextPluginGetOpaqueData(void *opaque_data_param,
                                                 void **opaque_data_return) {
  return pi2ur::piextPluginGetOpaqueData(opaque_data_param, opaque_data_return);
}

__SYCL_EXPORT pi_result piextProgramGetNativeHandle(
    pi_program Program, pi_native_handle *NativeHandle) {

  return pi2ur::piextProgramGetNativeHandle(Program, NativeHandle);
}

__SYCL_EXPORT pi_result piextProgramCreateWithNativeHandle(
    pi_native_handle NativeHandle, // missing
    pi_context Context, bool ownNativeHandle, pi_program *Program) {
  return pi2ur::piextProgramCreateWithNativeHandle(NativeHandle, Context,
                                                   ownNativeHandle, Program);
}

__SYCL_EXPORT pi_result piSamplerCreate(
    pi_context Context, const pi_sampler_properties *SamplerProperties,
    pi_sampler *RetSampler) {
  return pi2ur::piSamplerCreate(Context, SamplerProperties, RetSampler);
}

__SYCL_EXPORT pi_result piSamplerGetInfo(pi_sampler Sampler,
                                         pi_sampler_info ParamName,
                                         size_t ParamValueSize,
                                         void *ParamValue,
                                         size_t *ParamValueSizeRet) {
  return pi2ur::piSamplerGetInfo(Sampler, ParamName, ParamValueSize, ParamValue,
                                 ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piSamplerRetain(pi_sampler Sampler) {
  return pi2ur::piSamplerRetain(Sampler);
}

__SYCL_EXPORT pi_result piSamplerRelease(pi_sampler Sampler) {
  return pi2ur::piSamplerRelease(Sampler);
}

__SYCL_EXPORT pi_result piMemImageGetInfo(pi_mem Image, pi_image_info ParamName,
                                          size_t ParamValueSize,
                                          void *ParamValue,
                                          size_t *ParamValueSizeRet) {
  return pi2ur::piMemImageGetInfo(Image, ParamName, ParamValueSize, ParamValue,
                                  ParamValueSizeRet);
}

/// USM 2D Memcpy API
///
/// \param queue is the queue to submit to
/// \param blocking is whether this operation should block the host
/// \param dst_ptr is the location the data will be copied
/// \param dst_pitch is the total width of the destination memory including
/// padding
/// \param src_ptr is the data to be copied
/// \param dst_pitch is the total width of the source memory including padding
/// \param width is width in bytes of each row to be copied
/// \param height is height the columns to be copied
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
__SYCL_EXPORT pi_result piextUSMEnqueueMemcpy2D(
    pi_queue Queue, pi_bool Blocking, void *DstPtr, size_t DstPitch,
    const void *SrcPtr, size_t SrcPitch, size_t Width, size_t Height,
    pi_uint32 NumEventsInWaitList, const pi_event *EventsWaitList,
    pi_event *Event) {

  return pi2ur::piextUSMEnqueueMemcpy2D(
      Queue, Blocking, DstPtr, DstPitch, SrcPtr, SrcPitch, Width, Height,
      NumEventsInWaitList, EventsWaitList, Event);
}

/// API for writing data from host to a device global variable.
///
/// \param Queue is the queue
/// \param Program is the program containing the device global variable
/// \param Name is the unique identifier for the device global variable
/// \param BlockingWrite is true if the write should block
/// \param Count is the number of bytes to copy
/// \param Offset is the byte offset into the device global variable to start
/// copying
/// \param Src is a pointer to where the data must be copied from
/// \param NumEventsInWaitList is a number of events in the wait list
/// \param EventWaitList is the wait list
/// \param Event is the resulting event
pi_result piextEnqueueDeviceGlobalVariableWrite(
    pi_queue Queue, pi_program Program, const char *Name, pi_bool BlockingWrite,
    size_t Count, size_t Offset, const void *Src, pi_uint32 NumEventsInWaitList,
    const pi_event *EventsWaitList, pi_event *Event) {
  return pi2ur::piextEnqueueDeviceGlobalVariableWrite(
      Queue, Program, Name, BlockingWrite, Count, Offset, Src,
      NumEventsInWaitList, EventsWaitList, Event);
}

/// API reading data from a device global variable to host.
///
/// \param Queue is the queue
/// \param Program is the program containing the device global variable
/// \param Name is the unique identifier for the device global variable
/// \param BlockingRead is true if the read should block
/// \param Count is the number of bytes to copy
/// \param Offset is the byte offset into the device global variable to start
/// copying
/// \param Dst is a pointer to where the data must be copied to
/// \param NumEventsInWaitList is a number of events in the wait list
/// \param EventWaitList is the wait list
/// \param Event is the resulting event
pi_result piextEnqueueDeviceGlobalVariableRead(
    pi_queue Queue, pi_program Program, const char *Name, pi_bool BlockingRead,
    size_t Count, size_t Offset, void *Dst, pi_uint32 NumEventsInWaitList,
    const pi_event *EventsWaitList, pi_event *Event) {

  return pi2ur::piextEnqueueDeviceGlobalVariableRead(
      Queue, Program, Name, BlockingRead, Count, Offset, Dst,
      NumEventsInWaitList, EventsWaitList, Event);
}

pi_result piextMemImageCreateWithNativeHandle(
    pi_native_handle NativeHandle, pi_context Context, bool OwnNativeHandle,
    const pi_image_format *ImageFormat, const pi_image_desc *ImageDesc,
    pi_mem *Img) {
  return pi2ur::piextMemImageCreateWithNativeHandle(
      NativeHandle, Context, OwnNativeHandle, ImageFormat, ImageDesc, Img);
}

// Command buffer extension
pi_result piextCommandBufferCreate(pi_context Context, pi_device Device,
                                   const pi_ext_command_buffer_desc *Desc,
                                   pi_ext_command_buffer *RetCommandBuffer) {
  return pi2ur::piextCommandBufferCreate(Context, Device, Desc,
                                         RetCommandBuffer);
}

pi_result piextCommandBufferRetain(pi_ext_command_buffer CommandBuffer) {
  return pi2ur::piextCommandBufferRetain(CommandBuffer);
}

pi_result piextCommandBufferRelease(pi_ext_command_buffer CommandBuffer) {
  return pi2ur::piextCommandBufferRelease(CommandBuffer);
}

pi_result piextCommandBufferFinalize(pi_ext_command_buffer CommandBuffer) {
  return pi2ur::piextCommandBufferFinalize(CommandBuffer);
}

pi_result piextCommandBufferNDRangeKernel(
    pi_ext_command_buffer CommandBuffer, pi_kernel Kernel, pi_uint32 WorkDim,
    const size_t *GlobalWorkOffset, const size_t *GlobalWorkSize,
    const size_t *LocalWorkSize, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint,
    pi_ext_command_buffer_command *Command) {
  return pi2ur::piextCommandBufferNDRangeKernel(
      CommandBuffer, Kernel, WorkDim, GlobalWorkOffset, GlobalWorkSize,
      LocalWorkSize, NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint,
      Command);
}

pi_result piextCommandBufferMemcpyUSM(
    pi_ext_command_buffer CommandBuffer, void *DstPtr, const void *SrcPtr,
    size_t Size, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  return pi2ur::piextCommandBufferMemcpyUSM(CommandBuffer, DstPtr, SrcPtr, Size,
                                            NumSyncPointsInWaitList,
                                            SyncPointWaitList, SyncPoint);
}

pi_result piextCommandBufferMemBufferCopy(
    pi_ext_command_buffer CommandBuffer, pi_mem SrcMem, pi_mem DstMem,
    size_t SrcOffset, size_t DstOffset, size_t Size,
    pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  return pi2ur::piextCommandBufferMemBufferCopy(
      CommandBuffer, SrcMem, DstMem, SrcOffset, DstOffset, Size,
      NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

pi_result piextCommandBufferMemBufferCopyRect(
    pi_ext_command_buffer CommandBuffer, pi_mem SrcMem, pi_mem DstMem,
    pi_buff_rect_offset SrcOrigin, pi_buff_rect_offset DstOrigin,
    pi_buff_rect_region Region, size_t SrcRowPitch, size_t SrcSlicePitch,
    size_t DstRowPitch, size_t DstSlicePitch, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  return pi2ur::piextCommandBufferMemBufferCopyRect(
      CommandBuffer, SrcMem, DstMem, SrcOrigin, DstOrigin, Region, SrcRowPitch,
      SrcSlicePitch, DstRowPitch, DstSlicePitch, NumSyncPointsInWaitList,
      SyncPointWaitList, SyncPoint);
}

pi_result piextCommandBufferMemBufferRead(
    pi_ext_command_buffer CommandBuffer, pi_mem Buffer, size_t Offset,
    size_t Size, void *Dst, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  return pi2ur::piextCommandBufferMemBufferRead(
      CommandBuffer, Buffer, Offset, Size, Dst, NumSyncPointsInWaitList,
      SyncPointWaitList, SyncPoint);
}

pi_result piextCommandBufferMemBufferReadRect(
    pi_ext_command_buffer CommandBuffer, pi_mem Buffer,
    pi_buff_rect_offset BufferOffset, pi_buff_rect_offset HostOffset,
    pi_buff_rect_region Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, void *Ptr,
    pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  return pi2ur::piextCommandBufferMemBufferReadRect(
      CommandBuffer, Buffer, BufferOffset, HostOffset, Region, BufferRowPitch,
      BufferSlicePitch, HostRowPitch, HostSlicePitch, Ptr,
      NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

pi_result piextCommandBufferMemBufferWrite(
    pi_ext_command_buffer CommandBuffer, pi_mem Buffer, size_t Offset,
    size_t Size, const void *Ptr, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  return pi2ur::piextCommandBufferMemBufferWrite(
      CommandBuffer, Buffer, Offset, Size, Ptr, NumSyncPointsInWaitList,
      SyncPointWaitList, SyncPoint);
}

pi_result piextCommandBufferMemBufferWriteRect(
    pi_ext_command_buffer CommandBuffer, pi_mem Buffer,
    pi_buff_rect_offset BufferOffset, pi_buff_rect_offset HostOffset,
    pi_buff_rect_region Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, const void *Ptr,
    pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  return pi2ur::piextCommandBufferMemBufferWriteRect(
      CommandBuffer, Buffer, BufferOffset, HostOffset, Region, BufferRowPitch,
      BufferSlicePitch, HostRowPitch, HostSlicePitch, Ptr,
      NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

pi_result piextCommandBufferMemBufferFill(
    pi_ext_command_buffer CommandBuffer, pi_mem Buffer, const void *Pattern,
    size_t PatternSize, size_t Offset, size_t Size,
    pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  return pi2ur::piextCommandBufferMemBufferFill(
      CommandBuffer, Buffer, Pattern, PatternSize, Offset, Size,
      NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
}

pi_result piextCommandBufferFillUSM(pi_ext_command_buffer CommandBuffer,
                                    void *Ptr, const void *Pattern,
                                    size_t PatternSize, size_t Size,
                                    pi_uint32 NumSyncPointsInWaitList,
                                    const pi_ext_sync_point *SyncPointWaitList,
                                    pi_ext_sync_point *SyncPoint) {
  return pi2ur::piextCommandBufferFillUSM(
      CommandBuffer, Ptr, Pattern, PatternSize, Size, NumSyncPointsInWaitList,
      SyncPointWaitList, SyncPoint);
}

pi_result piextCommandBufferPrefetchUSM(
    pi_ext_command_buffer CommandBuffer, const void *Ptr, size_t Size,
    pi_usm_migration_flags Flags, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  return pi2ur::piextCommandBufferPrefetchUSM(CommandBuffer, Ptr, Size, Flags,
                                              NumSyncPointsInWaitList,
                                              SyncPointWaitList, SyncPoint);
}

pi_result piextCommandBufferAdviseUSM(
    pi_ext_command_buffer CommandBuffer, const void *Ptr, size_t Length,
    pi_mem_advice Advice, pi_uint32 NumSyncPointsInWaitList,
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  return pi2ur::piextCommandBufferAdviseUSM(CommandBuffer, Ptr, Length, Advice,
                                            NumSyncPointsInWaitList,
                                            SyncPointWaitList, SyncPoint);
}

pi_result piextEnqueueCommandBuffer(pi_ext_command_buffer CommandBuffer,
                                    pi_queue Queue,
                                    pi_uint32 NumEventsInWaitList,
                                    const pi_event *EventWaitList,
                                    pi_event *Event) {
  return pi2ur::piextEnqueueCommandBuffer(
      CommandBuffer, Queue, NumEventsInWaitList, EventWaitList, Event);
}

pi_result piextCommandBufferUpdateKernelLaunch(
    pi_ext_command_buffer_command Command,
    pi_ext_command_buffer_update_kernel_launch_desc *Desc) {
  return pi2ur::piextCommandBufferUpdateKernelLaunch(Command, Desc);
}

pi_result
piextCommandBufferRetainCommand(pi_ext_command_buffer_command Command) {
  return pi2ur::piextCommandBufferRetainCommand(Command);
}

pi_result
piextCommandBufferReleaseCommand(pi_ext_command_buffer_command Command) {
  return pi2ur::piextCommandBufferReleaseCommand(Command);
}

__SYCL_EXPORT pi_result piextVirtualMemGranularityGetInfo(
    pi_context Context, pi_device Device,
    pi_virtual_mem_granularity_info ParamName, size_t ParamValueSize,
    void *ParamValue, size_t *ParamValueSizeRet) {
  return pi2ur::piextVirtualMemGranularityGetInfo(Context, Device, ParamName,
                                                  ParamValueSize, ParamValue,
                                                  ParamValueSizeRet);
}

__SYCL_EXPORT pi_result
piextPhysicalMemCreate(pi_context Context, pi_device Device, size_t MemSize,
                       pi_physical_mem *RetPhsycialMem) {
  return pi2ur::piextPhysicalMemCreate(Context, Device, MemSize,
                                       RetPhsycialMem);
}

__SYCL_EXPORT pi_result piextPhysicalMemRetain(pi_physical_mem PhysicalMem) {
  return pi2ur::piextPhysicalMemRetain(PhysicalMem);
}

__SYCL_EXPORT pi_result piextPhysicalMemRelease(pi_physical_mem PhysicalMem) {
  return pi2ur::piextPhysicalMemRelease(PhysicalMem);
}

__SYCL_EXPORT pi_result piextVirtualMemReserve(pi_context Context,
                                               const void *Start,
                                               size_t RangeSize,
                                               void **RetPtr) {
  return pi2ur::piextVirtualMemReserve(Context, Start, RangeSize, RetPtr);
}

__SYCL_EXPORT pi_result piextVirtualMemFree(pi_context Context, const void *Ptr,
                                            size_t RangeSize) {
  return pi2ur::piextVirtualMemFree(Context, Ptr, RangeSize);
}

__SYCL_EXPORT pi_result
piextVirtualMemSetAccess(pi_context Context, const void *Ptr, size_t RangeSize,
                         pi_virtual_access_flags Flags) {
  return pi2ur::piextVirtualMemSetAccess(Context, Ptr, RangeSize, Flags);
}

__SYCL_EXPORT pi_result piextVirtualMemMap(pi_context Context, const void *Ptr,
                                           size_t RangeSize,
                                           pi_physical_mem PhysicalMem,
                                           size_t Offset,
                                           pi_virtual_access_flags Flags) {
  return pi2ur::piextVirtualMemMap(Context, Ptr, RangeSize, PhysicalMem, Offset,
                                   Flags);
}

__SYCL_EXPORT pi_result piextVirtualMemUnmap(pi_context Context,
                                             const void *Ptr,
                                             size_t RangeSize) {
  return pi2ur::piextVirtualMemUnmap(Context, Ptr, RangeSize);
}

__SYCL_EXPORT pi_result
piextVirtualMemGetInfo(pi_context Context, const void *Ptr, size_t RangeSize,
                       pi_virtual_mem_info ParamName, size_t ParamValueSize,
                       void *ParamValue, size_t *ParamValueSizeRet) {
  return pi2ur::piextVirtualMemGetInfo(Context, Ptr, RangeSize, ParamName,
                                       ParamValueSize, ParamValue,
                                       ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piGetDeviceAndHostTimer(pi_device Device,
                                                uint64_t *DeviceTime,
                                                uint64_t *HostTime) {
  return pi2ur::piGetDeviceAndHostTimer(Device, DeviceTime, HostTime);
}

__SYCL_EXPORT pi_result piPluginGetBackendOption(pi_platform platform,
                                                 const char *frontend_option,
                                                 const char **backend_option) {
  return pi2ur::piPluginGetBackendOption(platform, frontend_option,
                                         backend_option);
}

__SYCL_EXPORT pi_result piextEnablePeerAccess(pi_device command_device,
                                              pi_device peer_device) {

  return pi2ur::piextEnablePeerAccess(command_device, peer_device);
}

__SYCL_EXPORT pi_result piextDisablePeerAccess(pi_device command_device,
                                               pi_device peer_device) {

  return pi2ur::piextDisablePeerAccess(command_device, peer_device);
}

__SYCL_EXPORT pi_result piextPeerAccessGetInfo(
    pi_device command_device, pi_device peer_device, pi_peer_attr attr,
    size_t ParamValueSize, void *ParamValue, size_t *ParamValueSizeRet) {
  return pi2ur::piextPeerAccessGetInfo(command_device, peer_device, attr,
                                       ParamValueSize, ParamValue,
                                       ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piTearDown(void *) {
  releaseAdapters(Adapters.Vec);
  return PI_SUCCESS;
}

__SYCL_EXPORT pi_result piextMemImageAllocate(pi_context Context,
                                              pi_device Device,
                                              pi_image_format *ImageFormat,
                                              pi_image_desc *ImageDesc,
                                              pi_image_mem_handle *RetMem) {
  return pi2ur::piextMemImageAllocate(Context, Device, ImageFormat, ImageDesc,
                                      RetMem);
}

__SYCL_EXPORT pi_result piextMemUnsampledImageCreate(
    pi_context Context, pi_device Device, pi_image_mem_handle ImgMem,
    pi_image_format *ImageFormat, pi_image_desc *ImageDesc,
    pi_image_handle *RetHandle) {
  return pi2ur::piextMemUnsampledImageCreate(Context, Device, ImgMem,
                                             ImageFormat, ImageDesc, RetHandle);
}

__SYCL_EXPORT pi_result piextMemSampledImageCreate(
    pi_context Context, pi_device Device, pi_image_mem_handle ImgMem,
    pi_image_format *ImageFormat, pi_image_desc *ImageDesc, pi_sampler Sampler,
    pi_image_handle *RetHandle) {
  return pi2ur::piextMemSampledImageCreate(Context, Device, ImgMem, ImageFormat,
                                           ImageDesc, Sampler, RetHandle);
}

__SYCL_EXPORT pi_result piextBindlessImageSamplerCreate(
    pi_context Context, const pi_sampler_properties *SamplerProperties,
    float MinMipmapLevelClamp, float MaxMipmapLevelClamp, float MaxAnisotropy,
    pi_sampler *RetSampler) {
  return pi2ur::piextBindlessImageSamplerCreate(
      Context, SamplerProperties, MinMipmapLevelClamp, MaxMipmapLevelClamp,
      MaxAnisotropy, RetSampler);
}

__SYCL_EXPORT pi_result piextMemMipmapGetLevel(pi_context Context,
                                               pi_device Device,
                                               pi_image_mem_handle MipMem,
                                               unsigned int Level,
                                               pi_image_mem_handle *RetMem) {
  return pi2ur::piextMemMipmapGetLevel(Context, Device, MipMem, Level, RetMem);
}

__SYCL_EXPORT pi_result piextMemImageFree(pi_context Context, pi_device Device,
                                          pi_image_mem_handle MemoryHandle) {
  return pi2ur::piextMemImageFree(Context, Device, MemoryHandle);
}

__SYCL_EXPORT pi_result piextMemMipmapFree(pi_context Context, pi_device Device,
                                           pi_image_mem_handle MemoryHandle) {
  return pi2ur::piextMemMipmapFree(Context, Device, MemoryHandle);
}

__SYCL_EXPORT pi_result piextMemImageCopy(
    pi_queue Queue, void *DstPtr, const void *SrcPtr,
    const pi_image_format *ImageFormat, const pi_image_desc *ImageDesc,
    const pi_image_copy_flags Flags, pi_image_offset SrcOffset,
    pi_image_offset DstOffset, pi_image_region CopyExtent,
    pi_image_region HostExtent, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {
  return pi2ur::piextMemImageCopy(Queue, DstPtr, SrcPtr, ImageFormat, ImageDesc,
                                  Flags, SrcOffset, DstOffset, CopyExtent,
                                  HostExtent, NumEventsInWaitList,
                                  EventWaitList, Event);
}

__SYCL_EXPORT pi_result piextMemUnsampledImageHandleDestroy(
    pi_context Context, pi_device Device, pi_image_handle Handle) {
  return pi2ur::piextMemUnsampledImageHandleDestroy(Context, Device, Handle);
}

__SYCL_EXPORT pi_result piextMemSampledImageHandleDestroy(
    pi_context Context, pi_device Device, pi_image_handle Handle) {
  return pi2ur::piextMemSampledImageHandleDestroy(Context, Device, Handle);
}

__SYCL_EXPORT pi_result piextMemImageGetInfo(pi_context Context,
                                             pi_image_mem_handle MemHandle,
                                             pi_image_info ParamName,
                                             void *ParamValue,
                                             size_t *ParamValueSizeRet) {
  return pi2ur::piextMemImageGetInfo(Context, MemHandle, ParamName, ParamValue,
                                     ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piextImportExternalMemory(
    pi_context Context, pi_device Device, pi_external_mem_descriptor *MemDesc,
    pi_interop_mem_handle *RetHandle) {
  return pi2ur::piextImportExternalMemory(Context, Device, MemDesc, RetHandle);
}

__SYCL_EXPORT pi_result piextMemMapExternalArray(
    pi_context Context, pi_device Device, pi_image_format *ImageFormat,
    pi_image_desc *ImageDesc, pi_interop_mem_handle MemHandle,
    pi_image_mem_handle *RetMem) {
  return pi2ur::piextMemMapExternalArray(Context, Device, ImageFormat,
                                         ImageDesc, MemHandle, RetMem);
}

__SYCL_EXPORT pi_result piextMemReleaseInterop(pi_context Context,
                                               pi_device Device,
                                               pi_interop_mem_handle ExtMem) {
  return pi2ur::piextMemReleaseInterop(Context, Device, ExtMem);
}

__SYCL_EXPORT pi_result
piextImportExternalSemaphore(pi_context Context, pi_device Device,
                             pi_external_semaphore_descriptor *SemDesc,
                             pi_interop_semaphore_handle *RetHandle) {
  return pi2ur::piextImportExternalSemaphore(Context, Device, SemDesc,
                                             RetHandle);
}

__SYCL_EXPORT pi_result
piextReleaseExternalSemaphore(pi_context Context, pi_device Device,
                              pi_interop_semaphore_handle SemHandle) {
  return pi2ur::piextReleaseExternalSemaphore(Context, Device, SemHandle);
}

__SYCL_EXPORT pi_result piextWaitExternalSemaphore(
    pi_queue Queue, pi_interop_semaphore_handle SemHandle, bool HasWaitValue,
    pi_uint64 WaitValue, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {
  return pi2ur::piextWaitExternalSemaphore(Queue, SemHandle, HasWaitValue,
                                           WaitValue, NumEventsInWaitList,
                                           EventWaitList, Event);
}

__SYCL_EXPORT pi_result piextSignalExternalSemaphore(
    pi_queue Queue, pi_interop_semaphore_handle SemHandle, bool HasSignalValue,
    pi_uint64 SignalValue, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {
  return pi2ur::piextSignalExternalSemaphore(Queue, SemHandle, HasSignalValue,
                                             SignalValue, NumEventsInWaitList,
                                             EventWaitList, Event);
}

pi_result
piextEnqueueNativeCommand(pi_queue Queue, pi_enqueue_native_command_function Fn,
                          void *Data, pi_uint32 NumMems, const pi_mem *Mems,
                          pi_uint32 NumEventsInWaitList,
                          const pi_event *EventWaitList, pi_event *Event) {
  return pi2ur::piextEnqueueNativeCommand(Queue, Fn, Data, NumMems, Mems,
                                          NumEventsInWaitList, EventWaitList,
                                          Event);
}

// This interface is not in Unified Runtime currently
__SYCL_EXPORT pi_result piPluginInit(pi_plugin *PluginInit) {
  PI_ASSERT(PluginInit, PI_ERROR_INVALID_VALUE);

  const char SupportedVersion[] = _PI_UNIFIED_RUNTIME_PLUGIN_VERSION_STRING;

  // Check that the major version matches in PiVersion and SupportedVersion
  _PI_PLUGIN_VERSION_CHECK(PluginInit->PiVersion, SupportedVersion);

  // TODO: handle versioning/targets properly.
  size_t PluginVersionSize = sizeof(PluginInit->PluginVersion);

  PI_ASSERT(strlen(_PI_UNIFIED_RUNTIME_PLUGIN_VERSION_STRING) <
                PluginVersionSize,
            PI_ERROR_INVALID_VALUE);

  strncpy(PluginInit->PluginVersion, SupportedVersion, PluginVersionSize);

  // Initialize UR and discover adapters
  ur_loader_config_handle_t LoaderConfig{};
  HANDLE_ERRORS(urLoaderConfigCreate(&LoaderConfig));

  if (PluginInit->SanitizeType == _PI_SANITIZE_TYPE_ADDRESS) {
    auto Result = urLoaderConfigEnableLayer(LoaderConfig, "UR_LAYER_ASAN");
    if (Result != UR_RESULT_SUCCESS) {
      urLoaderConfigRelease(LoaderConfig);
      return ur2piResult(Result);
    }
  }

  HANDLE_ERRORS(urLoaderInit(0, LoaderConfig));
  HANDLE_ERRORS(urLoaderConfigRelease(LoaderConfig));

  uint32_t NumAdapters;
  HANDLE_ERRORS(urAdapterGet(0, nullptr, &NumAdapters));
  if (NumAdapters > 0) {
    Adapters.Vec.resize(NumAdapters);
    HANDLE_ERRORS(urAdapterGet(NumAdapters, Adapters.Vec.data(), nullptr));
  }

  // Bind interfaces that are already supported and "die" for unsupported ones
#define _PI_API(api)                                                           \
  (PluginInit->PiFunctionTable).api = (decltype(&::api))(&DieUnsupported);
#include <sycl/detail/pi.def>

#define _PI_API(api)                                                           \
  (PluginInit->PiFunctionTable).api = (decltype(&::api))(&api);

  _PI_API(piPlatformsGet)
  _PI_API(piPlatformGetInfo)
  _PI_API(piDevicesGet)
  _PI_API(piDeviceRetain)
  _PI_API(piDeviceRelease)
  _PI_API(piDeviceGetInfo)
  _PI_API(piDevicePartition)
  _PI_API(piextDeviceSelectBinary)
  _PI_API(piGetDeviceAndHostTimer)
  _PI_API(piextPlatformGetNativeHandle)
  _PI_API(piextPlatformCreateWithNativeHandle)
  _PI_API(piextDeviceGetNativeHandle)
  _PI_API(piextDeviceCreateWithNativeHandle)
  _PI_API(piPluginGetBackendOption)

  _PI_API(piContextCreate)
  _PI_API(piContextRelease)
  _PI_API(piContextRetain)
  _PI_API(piContextGetInfo)
  _PI_API(piextContextSetExtendedDeleter)
  _PI_API(piextContextGetNativeHandle)
  _PI_API(piextContextCreateWithNativeHandle)

  _PI_API(piQueueCreate)
  _PI_API(piQueueRelease)
  _PI_API(piextQueueCreate)
  _PI_API(piQueueFinish)
  _PI_API(piQueueGetInfo)
  _PI_API(piQueueRetain)
  _PI_API(piQueueFlush)
  _PI_API(piextQueueGetNativeHandle)
  _PI_API(piextQueueCreateWithNativeHandle)

  _PI_API(piProgramCreate)
  _PI_API(piProgramBuild)
  _PI_API(piextProgramGetNativeHandle)
  _PI_API(piextProgramCreateWithNativeHandle)
  _PI_API(piextProgramSetSpecializationConstant)
  _PI_API(piProgramLink)
  _PI_API(piKernelCreate)
  _PI_API(piextKernelSetArgMemObj)
  _PI_API(piextKernelCreateWithNativeHandle)
  _PI_API(piProgramRetain)
  _PI_API(piKernelSetExecInfo)
  _PI_API(piKernelGetInfo)
  _PI_API(piKernelSetArg)
  _PI_API(piKernelGetGroupInfo)
  _PI_API(piKernelRetain)
  _PI_API(piKernelRelease)
  _PI_API(piProgramRelease)
  _PI_API(piextKernelSetArgPointer)
  _PI_API(piextKernelSetArgSampler)
  _PI_API(piKernelGetSubGroupInfo)
  _PI_API(piProgramCreateWithBinary)
  _PI_API(piProgramGetInfo)
  _PI_API(piProgramCompile)
  _PI_API(piProgramGetBuildInfo)
  _PI_API(piextGetDeviceFunctionPointer)
  _PI_API(piextGetGlobalVariablePointer)

  _PI_API(piMemBufferCreate)
  _PI_API(piMemGetInfo)
  _PI_API(piMemBufferPartition)
  _PI_API(piEnqueueMemImageCopy)
  _PI_API(piextMemGetNativeHandle)
  _PI_API(piextMemCreateWithNativeHandle)
  _PI_API(piMemRetain)
  _PI_API(piextUSMGetMemAllocInfo)
  _PI_API(piextUSMEnqueuePrefetch)
  _PI_API(piextUSMEnqueueFill2D)
  _PI_API(piextUSMEnqueueMemset2D)
  _PI_API(piextUSMEnqueueMemAdvise)
  _PI_API(piMemRelease)
  _PI_API(piMemImageCreate)
  _PI_API(piMemImageGetInfo)
  _PI_API(piextUSMEnqueueMemcpy2D)
  _PI_API(piextEnqueueDeviceGlobalVariableWrite)
  _PI_API(piextEnqueueDeviceGlobalVariableRead)

  _PI_API(piextUSMHostAlloc)
  _PI_API(piextUSMDeviceAlloc)
  _PI_API(piextUSMSharedAlloc)
  _PI_API(piextUSMFree)

  _PI_API(piextUSMImport)
  _PI_API(piextUSMRelease)

  _PI_API(piEnqueueKernelLaunch)
  _PI_API(piEnqueueMemImageWrite)
  _PI_API(piEnqueueMemImageRead)
  _PI_API(piEnqueueMemBufferMap)
  _PI_API(piEnqueueMemUnmap)
  _PI_API(piEnqueueMemBufferFill)
  _PI_API(piextUSMEnqueueFill)
  _PI_API(piEnqueueMemBufferCopyRect)
  _PI_API(piEnqueueMemBufferCopy)
  _PI_API(piextUSMEnqueueMemcpy)
  _PI_API(piEnqueueMemBufferWriteRect)
  _PI_API(piEnqueueMemBufferWrite)
  _PI_API(piEnqueueMemBufferReadRect)
  _PI_API(piEnqueueMemBufferRead)
  _PI_API(piEnqueueEventsWaitWithBarrier)
  _PI_API(piEnqueueEventsWait)
  _PI_API(piEnqueueMemImageFill)

  _PI_API(piEventSetCallback)
  _PI_API(piEventSetStatus)
  _PI_API(piEventRetain)
  _PI_API(piEventRelease)
  _PI_API(piextEventCreateWithNativeHandle)
  _PI_API(piEventsWait)
  _PI_API(piEventGetInfo)
  _PI_API(piextEventGetNativeHandle)
  _PI_API(piEventGetProfilingInfo)
  _PI_API(piEventCreate)
  _PI_API(piEnqueueTimestampRecordingExp)

  _PI_API(piSamplerCreate)
  _PI_API(piSamplerGetInfo)
  _PI_API(piSamplerRetain)
  _PI_API(piSamplerRelease)

  // Peer to Peer
  _PI_API(piextEnablePeerAccess)
  _PI_API(piextDisablePeerAccess)
  _PI_API(piextPeerAccessGetInfo)

  // Launch Properties
  _PI_API(piextEnqueueKernelLaunchCustom)

  _PI_API(piextPluginGetOpaqueData)
  _PI_API(piTearDown)

  return PI_SUCCESS;
}

} // extern "C
