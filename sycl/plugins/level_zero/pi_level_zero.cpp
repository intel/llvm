//===-------- pi_level_zero.cpp - Level Zero Plugin --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

/// \file pi_level_zero.cpp
/// Implementation of Level Zero Plugin.
///
/// \ingroup sycl_pi_level_zero

#include "pi_level_zero.hpp"
#include "ur_bindings.hpp"

// Defined in tracing.cpp
void enableZeTracing();
void disableZeTracing();

extern "C" {

// Forward declarations
decltype(piEventCreate) piEventCreate;

pi_result piPlatformsGet(pi_uint32 NumEntries, pi_platform *Platforms,
                         pi_uint32 *NumPlatforms) {
  return pi2ur::piPlatformsGet(NumEntries, Platforms, NumPlatforms);
}

pi_result piPlatformGetInfo(pi_platform Platform, pi_platform_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {
  // To distinguish this L0 platform from Unified Runtime one.
  if (ParamName == PI_PLATFORM_INFO_NAME) {
    ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
    return ReturnValue("Intel(R) Level-Zero");
  }
  return pi2ur::piPlatformGetInfo(Platform, ParamName, ParamValueSize,
                                  ParamValue, ParamValueSizeRet);
}

pi_result piextPlatformGetNativeHandle(pi_platform Platform,
                                       pi_native_handle *NativeHandle) {

  return pi2ur::piextPlatformGetNativeHandle(Platform, NativeHandle);
}

pi_result piextPlatformCreateWithNativeHandle(pi_native_handle NativeHandle,
                                              pi_platform *Platform) {

  return pi2ur::piextPlatformCreateWithNativeHandle(NativeHandle, Platform);
}

pi_result piPluginGetLastError(char **message) {
  return pi2ur::piPluginGetLastError(message);
}

// Returns plugin specific backend option.
// Return '-ze-opt-disable' for frontend_option = -O0.
// Return '-ze-opt-level=2' for frontend_option = -O1, O2 or -O3.
// Return '-igc_opts 'PartitionUnit=1,SubroutineThreshold=50000'' for
// frontend_option = -ftarget-compile-fast.
pi_result piPluginGetBackendOption(pi_platform platform,
                                   const char *frontend_option,
                                   const char **backend_option) {
  return pi2ur::piPluginGetBackendOption(platform, frontend_option,
                                         backend_option);
}

pi_result piDevicesGet(pi_platform Platform, pi_device_type DeviceType,
                       pi_uint32 NumEntries, pi_device *Devices,
                       pi_uint32 *NumDevices) {
  return pi2ur::piDevicesGet(Platform, DeviceType, NumEntries, Devices,
                             NumDevices);
}

pi_result piDeviceRetain(pi_device Device) {
  return pi2ur::piDeviceRetain(Device);
}

pi_result piDeviceRelease(pi_device Device) {
  return pi2ur::piDeviceRelease(Device);
}

pi_result piDeviceGetInfo(pi_device Device, pi_device_info ParamName,
                          size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet) {
  return pi2ur::piDeviceGetInfo(Device, ParamName, ParamValueSize, ParamValue,
                                ParamValueSizeRet);
}

pi_result piDevicePartition(pi_device Device,
                            const pi_device_partition_property *Properties,
                            pi_uint32 NumDevices, pi_device *OutDevices,
                            pi_uint32 *OutNumDevices) {
  return pi2ur::piDevicePartition(Device, Properties, NumDevices, OutDevices,
                                  OutNumDevices);
}

pi_result
piextDeviceSelectBinary(pi_device Device, // TODO: does this need to be context?
                        pi_device_binary *Binaries, pi_uint32 NumBinaries,
                        pi_uint32 *SelectedBinaryInd) {
  return pi2ur::piextDeviceSelectBinary(Device, Binaries, NumBinaries,
                                        SelectedBinaryInd);
}

pi_result piextDeviceGetNativeHandle(pi_device Device,
                                     pi_native_handle *NativeHandle) {

  return pi2ur::piextDeviceGetNativeHandle(Device, NativeHandle);
}

pi_result piextDeviceCreateWithNativeHandle(pi_native_handle NativeHandle,
                                            pi_platform Platform,
                                            pi_device *Device) {

  return pi2ur::piextDeviceCreateWithNativeHandle(NativeHandle, Platform,
                                                  Device);
}

pi_result piContextCreate(const pi_context_properties *Properties,
                          pi_uint32 NumDevices, const pi_device *Devices,
                          void (*PFnNotify)(const char *ErrInfo,
                                            const void *PrivateInfo, size_t CB,
                                            void *UserData),
                          void *UserData, pi_context *RetContext) {
  return pi2ur::piContextCreate(Properties, NumDevices, Devices, PFnNotify,
                                UserData, RetContext);
}

pi_result piContextGetInfo(pi_context Context, pi_context_info ParamName,
                           size_t ParamValueSize, void *ParamValue,
                           size_t *ParamValueSizeRet) {

  return pi2ur::piContextGetInfo(Context, ParamName, ParamValueSize, ParamValue,
                                 ParamValueSizeRet);
}

// FIXME: Dummy implementation to prevent link fail
pi_result piextContextSetExtendedDeleter(pi_context Context,
                                         pi_context_extended_deleter Function,
                                         void *UserData) {
  return pi2ur::piextContextSetExtendedDeleter(Context, Function, UserData);
}

pi_result piextContextGetNativeHandle(pi_context Context,
                                      pi_native_handle *NativeHandle) {
  return pi2ur::piextContextGetNativeHandle(Context, NativeHandle);
}

pi_result piextContextCreateWithNativeHandle(pi_native_handle NativeHandle,
                                             pi_uint32 NumDevices,
                                             const pi_device *Devices,
                                             bool OwnNativeHandle,
                                             pi_context *RetContext) {
  return pi2ur::piextContextCreateWithNativeHandle(
      NativeHandle, NumDevices, Devices, OwnNativeHandle, RetContext);
}

pi_result piContextRetain(pi_context Context) {

  return pi2ur::piContextRetain(Context);
}

pi_result piContextRelease(pi_context Context) {
  return pi2ur::piContextRelease(Context);
}

pi_result piQueueCreate(pi_context Context, pi_device Device,
                        pi_queue_properties Flags, pi_queue *Queue) {
  pi_queue_properties Properties[] = {PI_QUEUE_FLAGS, Flags, 0};
  return piextQueueCreate(Context, Device, Properties, Queue);
}

pi_result piextQueueCreate(pi_context Context, pi_device Device,
                           pi_queue_properties *Properties, pi_queue *Queue) {
  return pi2ur::piextQueueCreate(Context, Device, Properties, Queue);
}

pi_result piQueueGetInfo(pi_queue Queue, pi_queue_info ParamName,
                         size_t ParamValueSize, void *ParamValue,
                         size_t *ParamValueSizeRet) {

  return pi2ur::piQueueGetInfo(Queue, ParamName, ParamValueSize, ParamValue,
                               ParamValueSizeRet);
}

pi_result piQueueRetain(pi_queue Queue) { return pi2ur::piQueueRetain(Queue); }

pi_result piQueueRelease(pi_queue Queue) {
  return pi2ur::piQueueRelease(Queue);
}

pi_result piQueueFinish(pi_queue Queue) { return pi2ur::piQueueFinish(Queue); }

pi_result piQueueFlush(pi_queue Queue) { return pi2ur::piQueueFlush(Queue); }

pi_result piextQueueGetNativeHandle(pi_queue Queue,
                                    pi_native_handle *NativeHandle,
                                    int32_t *NativeHandleDesc) {

  return pi2ur::piextQueueGetNativeHandle(Queue, NativeHandle,
                                          NativeHandleDesc);
}

pi_result piextQueueCreateWithNativeHandle(pi_native_handle NativeHandle,
                                           int32_t NativeHandleDesc,
                                           pi_context Context, pi_device Device,
                                           bool OwnNativeHandle,
                                           pi_queue_properties *Properties,
                                           pi_queue *Queue) {

  return pi2ur::piextQueueCreateWithNativeHandle(
      NativeHandle, NativeHandleDesc, Context, Device, OwnNativeHandle,
      Properties, Queue);
}

pi_result piMemBufferCreate(pi_context Context, pi_mem_flags Flags, size_t Size,
                            void *HostPtr, pi_mem *RetMem,
                            const pi_mem_properties *properties) {
  return pi2ur::piMemBufferCreate(Context, Flags, Size, HostPtr, RetMem,
                                  properties);
}

pi_result piMemGetInfo(pi_mem Mem, pi_mem_info ParamName, size_t ParamValueSize,
                       void *ParamValue, size_t *ParamValueSizeRet) {
  return pi2ur::piMemGetInfo(Mem, ParamName, ParamValueSize, ParamValue,
                             ParamValueSizeRet);
}

pi_result piMemRetain(pi_mem Mem) { return pi2ur::piMemRetain(Mem); }

pi_result piMemRelease(pi_mem Mem) { return pi2ur::piMemRelease(Mem); }

pi_result piMemImageCreate(pi_context Context, pi_mem_flags Flags,
                           const pi_image_format *ImageFormat,
                           const pi_image_desc *ImageDesc, void *HostPtr,
                           pi_mem *RetImage) {

  return pi2ur::piMemImageCreate(Context, Flags, ImageFormat, ImageDesc,
                                 HostPtr, RetImage);
}

pi_result piextMemGetNativeHandle(pi_mem Mem, pi_native_handle *NativeHandle) {
  return pi2ur::piextMemGetNativeHandle(Mem, NativeHandle);
}

pi_result piextMemCreateWithNativeHandle(pi_native_handle NativeHandle,
                                         pi_context Context,
                                         bool ownNativeHandle, pi_mem *Mem) {
  return pi2ur::piextMemCreateWithNativeHandle(NativeHandle, Context,
                                               ownNativeHandle, Mem);
}

pi_result piProgramCreate(pi_context Context, const void *ILBytes,
                          size_t Length, pi_program *Program) {
  return pi2ur::piProgramCreate(Context, ILBytes, Length, Program);
}

pi_result piProgramCreateWithBinary(
    pi_context Context, pi_uint32 NumDevices, const pi_device *DeviceList,
    const size_t *Lengths, const unsigned char **Binaries,
    size_t NumMetadataEntries, const pi_device_binary_property *Metadata,
    pi_int32 *BinaryStatus, pi_program *Program) {

  return pi2ur::piProgramCreateWithBinary(Context, NumDevices, DeviceList,
                                          Lengths, Binaries, NumMetadataEntries,
                                          Metadata, BinaryStatus, Program);
}

pi_result piextMemImageCreateWithNativeHandle(
    pi_native_handle NativeHandle, pi_context Context, bool OwnNativeHandle,
    const pi_image_format *ImageFormat, const pi_image_desc *ImageDesc,
    pi_mem *Img) {
  return pi2ur::piextMemImageCreateWithNativeHandle(
      NativeHandle, Context, OwnNativeHandle, ImageFormat, ImageDesc, Img);
}

pi_result piProgramGetInfo(pi_program Program, pi_program_info ParamName,
                           size_t ParamValueSize, void *ParamValue,
                           size_t *ParamValueSizeRet) {

  return pi2ur::piProgramGetInfo(Program, ParamName, ParamValueSize, ParamValue,
                                 ParamValueSizeRet);
}

pi_result piProgramLink(pi_context Context, pi_uint32 NumDevices,
                        const pi_device *DeviceList, const char *Options,
                        pi_uint32 NumInputPrograms,
                        const pi_program *InputPrograms,
                        void (*PFnNotify)(pi_program Program, void *UserData),
                        void *UserData, pi_program *RetProgram) {
  return pi2ur::piProgramLink(Context, NumDevices, DeviceList, Options,
                              NumInputPrograms, InputPrograms, PFnNotify,
                              UserData, RetProgram);
}

pi_result piProgramCompile(
    pi_program Program, pi_uint32 NumDevices, const pi_device *DeviceList,
    const char *Options, pi_uint32 NumInputHeaders,
    const pi_program *InputHeaders, const char **HeaderIncludeNames,
    void (*PFnNotify)(pi_program Program, void *UserData), void *UserData) {

  return pi2ur::piProgramCompile(Program, NumDevices, DeviceList, Options,
                                 NumInputHeaders, InputHeaders,
                                 HeaderIncludeNames, PFnNotify, UserData);
}

pi_result piProgramBuild(pi_program Program, pi_uint32 NumDevices,
                         const pi_device *DeviceList, const char *Options,
                         void (*PFnNotify)(pi_program Program, void *UserData),
                         void *UserData) {
  return pi2ur::piProgramBuild(Program, NumDevices, DeviceList, Options,
                               PFnNotify, UserData);
}

pi_result piProgramGetBuildInfo(pi_program Program, pi_device Device,
                                pi_program_build_info ParamName,
                                size_t ParamValueSize, void *ParamValue,
                                size_t *ParamValueSizeRet) {

  return pi2ur::piProgramGetBuildInfo(Program, Device, ParamName,
                                      ParamValueSize, ParamValue,
                                      ParamValueSizeRet);
}

pi_result piProgramRetain(pi_program Program) {
  return pi2ur::piProgramRetain(Program);
}

pi_result piProgramRelease(pi_program Program) {
  return pi2ur::piProgramRelease(Program);
}

pi_result piextProgramGetNativeHandle(pi_program Program,
                                      pi_native_handle *NativeHandle) {
  return pi2ur::piextProgramGetNativeHandle(Program, NativeHandle);
}

pi_result piextProgramCreateWithNativeHandle(pi_native_handle NativeHandle,
                                             pi_context Context,
                                             bool OwnNativeHandle,
                                             pi_program *Program) {
  return pi2ur::piextProgramCreateWithNativeHandle(NativeHandle, Context,
                                                   OwnNativeHandle, Program);
}

pi_result piKernelCreate(pi_program Program, const char *KernelName,
                         pi_kernel *RetKernel) {

  return pi2ur::piKernelCreate(Program, KernelName, RetKernel);
}

pi_result piKernelSetArg(pi_kernel Kernel, pi_uint32 ArgIndex, size_t ArgSize,
                         const void *ArgValue) {

  return pi2ur::piKernelSetArg(Kernel, ArgIndex, ArgSize, ArgValue);
}

// Special version of piKernelSetArg to accept pi_mem.
pi_result piextKernelSetArgMemObj(pi_kernel Kernel, pi_uint32 ArgIndex,
                                  const pi_mem_obj_property *ArgProperties,
                                  const pi_mem *ArgValue) {
  return pi2ur::piextKernelSetArgMemObj(Kernel, ArgIndex, ArgProperties,
                                        ArgValue);
}

// Special version of piKernelSetArg to accept pi_sampler.
pi_result piextKernelSetArgSampler(pi_kernel Kernel, pi_uint32 ArgIndex,
                                   const pi_sampler *ArgValue) {

  return pi2ur::piextKernelSetArgSampler(Kernel, ArgIndex, ArgValue);
}

pi_result piKernelGetInfo(pi_kernel Kernel, pi_kernel_info ParamName,
                          size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet) {

  return pi2ur::piKernelGetInfo(Kernel, ParamName, ParamValueSize, ParamValue,
                                ParamValueSizeRet);
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
    pi_image_format *ImageFormat, pi_image_desc *ImageDesc, pi_mem *RetMem,
    pi_image_handle *RetHandle) {
  return pi2ur::piextMemUnsampledImageCreate(
      Context, Device, ImgMem, ImageFormat, ImageDesc, RetMem, RetHandle);
}

__SYCL_EXPORT pi_result piextMemSampledImageCreate(
    pi_context Context, pi_device Device, pi_image_mem_handle ImgMem,
    pi_image_format *ImageFormat, pi_image_desc *ImageDesc, pi_sampler Sampler,
    pi_mem *RetMem, pi_image_handle *RetHandle) {
  return pi2ur::piextMemSampledImageCreate(Context, Device, ImgMem, ImageFormat,
                                           ImageDesc, Sampler, RetMem,
                                           RetHandle);
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
    pi_queue Queue, void *DstPtr, void *SrcPtr,
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

__SYCL_EXPORT pi_result piextMemImageGetInfo(pi_image_mem_handle MemHandle,
                                             pi_image_info ParamName,
                                             void *ParamValue,
                                             size_t *ParamValueSizeRet) {
  return pi2ur::piextMemImageGetInfo(MemHandle, ParamName, ParamValue,
                                     ParamValueSizeRet);
}

__SYCL_EXPORT pi_result
piextMemImportOpaqueFD(pi_context Context, pi_device Device, size_t Size,
                       int FileDescriptor, pi_interop_mem_handle *RetHandle) {
  return pi2ur::piextMemImportOpaqueFD(Context, Device, Size, FileDescriptor,
                                       RetHandle);
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

__SYCL_EXPORT pi_result piextImportExternalSemaphoreOpaqueFD(
    pi_context Context, pi_device Device, int FileDescriptor,
    pi_interop_semaphore_handle *RetHandle) {
  return pi2ur::piextImportExternalSemaphoreOpaqueFD(Context, Device,
                                                     FileDescriptor, RetHandle);
}

__SYCL_EXPORT pi_result
piextDestroyExternalSemaphore(pi_context Context, pi_device Device,
                              pi_interop_semaphore_handle SemHandle) {
  return pi2ur::piextDestroyExternalSemaphore(Context, Device, SemHandle);
}

__SYCL_EXPORT pi_result piextWaitExternalSemaphore(
    pi_queue Queue, pi_interop_semaphore_handle SemHandle,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *Event) {
  return pi2ur::piextWaitExternalSemaphore(
      Queue, SemHandle, NumEventsInWaitList, EventWaitList, Event);
}

__SYCL_EXPORT pi_result piextSignalExternalSemaphore(
    pi_queue Queue, pi_interop_semaphore_handle SemHandle,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *Event) {
  return pi2ur::piextSignalExternalSemaphore(
      Queue, SemHandle, NumEventsInWaitList, EventWaitList, Event);
}

pi_result piKernelGetGroupInfo(pi_kernel Kernel, pi_device Device,
                               pi_kernel_group_info ParamName,
                               size_t ParamValueSize, void *ParamValue,
                               size_t *ParamValueSizeRet) {

  return pi2ur::piKernelGetGroupInfo(Kernel, Device, ParamName, ParamValueSize,
                                     ParamValue, ParamValueSizeRet);
}

pi_result piKernelGetSubGroupInfo(pi_kernel Kernel, pi_device Device,
                                  pi_kernel_sub_group_info ParamName,
                                  size_t InputValueSize, const void *InputValue,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {

  return pi2ur::piKernelGetSubGroupInfo(
      Kernel, Device, ParamName, InputValueSize, InputValue, ParamValueSize,
      ParamValue, ParamValueSizeRet);
}

pi_result piKernelRetain(pi_kernel Kernel) {

  return pi2ur::piKernelRetain(Kernel);
}

pi_result piKernelRelease(pi_kernel Kernel) {

  return pi2ur::piKernelRelease(Kernel);
}

pi_result
piEnqueueKernelLaunch(pi_queue Queue, pi_kernel Kernel, pi_uint32 WorkDim,
                      const size_t *GlobalWorkOffset,
                      const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
                      pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *OutEvent) {
  return pi2ur::piEnqueueKernelLaunch(
      Queue, Kernel, WorkDim, GlobalWorkOffset, GlobalWorkSize, LocalWorkSize,
      NumEventsInWaitList, EventWaitList, OutEvent);
}

pi_result piextEnqueueCooperativeKernelLaunch(
    pi_queue Queue, pi_kernel Kernel, pi_uint32 WorkDim,
    const size_t *GlobalWorkOffset, const size_t *GlobalWorkSize,
    const size_t *LocalWorkSize, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *OutEvent) {
  return pi2ur::piextEnqueueCooperativeKernelLaunch(
      Queue, Kernel, WorkDim, GlobalWorkOffset, GlobalWorkSize, LocalWorkSize,
      NumEventsInWaitList, EventWaitList, OutEvent);
}

pi_result piextKernelCreateWithNativeHandle(pi_native_handle NativeHandle,
                                            pi_context Context,
                                            pi_program Program,
                                            bool OwnNativeHandle,
                                            pi_kernel *Kernel) {

  return pi2ur::piextKernelCreateWithNativeHandle(
      NativeHandle, Context, Program, OwnNativeHandle, Kernel);
}

pi_result piextKernelGetNativeHandle(pi_kernel Kernel,
                                     pi_native_handle *NativeHandle) {
  return pi2ur::piextKernelGetNativeHandle(Kernel, NativeHandle);
}

pi_result piextKernelSuggestMaxCooperativeGroupCount(
    pi_kernel Kernel, size_t LocalWorkSize, size_t DynamicSharedMemorySize,
    pi_uint32 *GroupCountRet) {
  return pi2ur::piextKernelSuggestMaxCooperativeGroupCount(
      Kernel, LocalWorkSize, DynamicSharedMemorySize, GroupCountRet);
}

//
// Events
//

// External PI API entry
pi_result piEventCreate(pi_context Context, pi_event *RetEvent) {
  return pi2ur::piEventCreate(Context, RetEvent);
}

pi_result piEventGetInfo(pi_event Event, pi_event_info ParamName,
                         size_t ParamValueSize, void *ParamValue,
                         size_t *ParamValueSizeRet) {
  return pi2ur::piEventGetInfo(Event, ParamName, ParamValueSize, ParamValue,
                               ParamValueSizeRet);
}

pi_result piEventGetProfilingInfo(pi_event Event, pi_profiling_info ParamName,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {

  return pi2ur::piEventGetProfilingInfo(Event, ParamName, ParamValueSize,
                                        ParamValue, ParamValueSizeRet);
}

pi_result piEventsWait(pi_uint32 NumEvents, const pi_event *EventList) {
  return pi2ur::piEventsWait(NumEvents, EventList);
}

pi_result piEventSetCallback(pi_event Event, pi_int32 CommandExecCallbackType,
                             void (*PFnNotify)(pi_event Event,
                                               pi_int32 EventCommandStatus,
                                               void *UserData),
                             void *UserData) {
  return pi2ur::piEventSetCallback(Event, CommandExecCallbackType, PFnNotify,
                                   UserData);
}

pi_result piEventSetStatus(pi_event Event, pi_int32 ExecutionStatus) {
  return pi2ur::piEventSetStatus(Event, ExecutionStatus);
}

pi_result piEventRetain(pi_event Event) { return pi2ur::piEventRetain(Event); }

pi_result piEventRelease(pi_event Event) {
  return pi2ur::piEventRelease(Event);
}

pi_result piextEventGetNativeHandle(pi_event Event,
                                    pi_native_handle *NativeHandle) {

  return pi2ur::piextEventGetNativeHandle(Event, NativeHandle);
}

pi_result piextEventCreateWithNativeHandle(pi_native_handle NativeHandle,
                                           pi_context Context,
                                           bool OwnNativeHandle,
                                           pi_event *Event) {
  return pi2ur::piextEventCreateWithNativeHandle(NativeHandle, Context,
                                                 OwnNativeHandle, Event);
}

//
// Sampler
//
pi_result piSamplerCreate(pi_context Context,
                          const pi_sampler_properties *SamplerProperties,
                          pi_sampler *RetSampler) {
  return pi2ur::piSamplerCreate(Context, SamplerProperties, RetSampler);
}

pi_result piSamplerGetInfo(pi_sampler Sampler, pi_sampler_info ParamName,
                           size_t ParamValueSize, void *ParamValue,
                           size_t *ParamValueSizeRet) {

  return pi2ur::piSamplerGetInfo(Sampler, ParamName, ParamValueSize, ParamValue,
                                 ParamValueSizeRet);
}

pi_result piSamplerRetain(pi_sampler Sampler) {
  return pi2ur::piSamplerRetain(Sampler);
}

pi_result piSamplerRelease(pi_sampler Sampler) {
  return pi2ur::piSamplerRelease(Sampler);
}

//
// Queue Commands
//
pi_result piEnqueueEventsWait(pi_queue Queue, pi_uint32 NumEventsInWaitList,
                              const pi_event *EventWaitList,
                              pi_event *OutEvent) {

  return pi2ur::piEnqueueEventsWait(Queue, NumEventsInWaitList, EventWaitList,
                                    OutEvent);
}

pi_result piEnqueueEventsWaitWithBarrier(pi_queue Queue,
                                         pi_uint32 NumEventsInWaitList,
                                         const pi_event *EventWaitList,
                                         pi_event *OutEvent) {

  return pi2ur::piEnqueueEventsWaitWithBarrier(Queue, NumEventsInWaitList,
                                               EventWaitList, OutEvent);
}

pi_result piEnqueueMemBufferRead(pi_queue Queue, pi_mem Src,
                                 pi_bool BlockingRead, size_t Offset,
                                 size_t Size, void *Dst,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {

  return pi2ur::piEnqueueMemBufferRead(Queue, Src, BlockingRead, Offset, Size,
                                       Dst, NumEventsInWaitList, EventWaitList,
                                       Event);
}

pi_result piEnqueueMemBufferReadRect(
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

pi_result piEnqueueMemBufferWrite(pi_queue Queue, pi_mem Buffer,
                                  pi_bool BlockingWrite, size_t Offset,
                                  size_t Size, const void *Ptr,
                                  pi_uint32 NumEventsInWaitList,
                                  const pi_event *EventWaitList,
                                  pi_event *Event) {

  return pi2ur::piEnqueueMemBufferWrite(Queue, Buffer, BlockingWrite, Offset,
                                        Size, Ptr, NumEventsInWaitList,
                                        EventWaitList, Event);
}

pi_result piEnqueueMemBufferWriteRect(
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

pi_result piEnqueueMemBufferCopy(pi_queue Queue, pi_mem SrcMem, pi_mem DstMem,
                                 size_t SrcOffset, size_t DstOffset,
                                 size_t Size, pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {

  return pi2ur::piEnqueueMemBufferCopy(Queue, SrcMem, DstMem, SrcOffset,
                                       DstOffset, Size, NumEventsInWaitList,
                                       EventWaitList, Event);
}

pi_result piEnqueueMemBufferCopyRect(
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

pi_result piEnqueueMemBufferFill(pi_queue Queue, pi_mem Buffer,
                                 const void *Pattern, size_t PatternSize,
                                 size_t Offset, size_t Size,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {

  return pi2ur::piEnqueueMemBufferFill(Queue, Buffer, Pattern, PatternSize,
                                       Offset, Size, NumEventsInWaitList,
                                       EventWaitList, Event);
}

pi_result piEnqueueMemBufferMap(pi_queue Queue, pi_mem Mem, pi_bool BlockingMap,
                                pi_map_flags MapFlags, size_t Offset,
                                size_t Size, pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *OutEvent, void **RetMap) {

  return pi2ur::piEnqueueMemBufferMap(Queue, Mem, BlockingMap, MapFlags, Offset,
                                      Size, NumEventsInWaitList, EventWaitList,
                                      OutEvent, RetMap);
}

pi_result piEnqueueMemUnmap(pi_queue Queue, pi_mem Mem, void *MappedPtr,
                            pi_uint32 NumEventsInWaitList,
                            const pi_event *EventWaitList, pi_event *OutEvent) {

  return pi2ur::piEnqueueMemUnmap(Queue, Mem, MappedPtr, NumEventsInWaitList,
                                  EventWaitList, OutEvent);
}

pi_result piMemImageGetInfo(pi_mem Image, pi_image_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {

  return pi2ur::piMemImageGetInfo(Image, ParamName, ParamValueSize, ParamValue,
                                  ParamValueSizeRet);
}

pi_result piEnqueueMemImageRead(pi_queue Queue, pi_mem Image,
                                pi_bool BlockingRead, pi_image_offset Origin,
                                pi_image_region Region, size_t RowPitch,
                                size_t SlicePitch, void *Ptr,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {
  return pi2ur::piEnqueueMemImageRead(
      Queue, Image, BlockingRead, Origin, Region, RowPitch, SlicePitch, Ptr,
      NumEventsInWaitList, EventWaitList, Event);
}

pi_result piEnqueueMemImageWrite(pi_queue Queue, pi_mem Image,
                                 pi_bool BlockingWrite, pi_image_offset Origin,
                                 pi_image_region Region, size_t InputRowPitch,
                                 size_t InputSlicePitch, const void *Ptr,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {

  return pi2ur::piEnqueueMemImageWrite(
      Queue, Image, BlockingWrite, Origin, Region, InputRowPitch,
      InputSlicePitch, Ptr, NumEventsInWaitList, EventWaitList, Event);
}

pi_result
piEnqueueMemImageCopy(pi_queue Queue, pi_mem SrcImage, pi_mem DstImage,
                      pi_image_offset SrcOrigin, pi_image_offset DstOrigin,
                      pi_image_region Region, pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *Event) {
  return pi2ur::piEnqueueMemImageCopy(Queue, SrcImage, DstImage, SrcOrigin,
                                      DstOrigin, Region, NumEventsInWaitList,
                                      EventWaitList, Event);
}

pi_result piEnqueueMemImageFill(pi_queue Queue, pi_mem Image,
                                const void *FillColor, const size_t *Origin,
                                const size_t *Region,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {

  return pi2ur::piEnqueueMemImageFill(Queue, Image, FillColor, Origin, Region,
                                      NumEventsInWaitList, EventWaitList,
                                      Event);
}

pi_result piMemBufferPartition(pi_mem Buffer, pi_mem_flags Flags,
                               pi_buffer_create_type BufferCreateType,
                               void *BufferCreateInfo, pi_mem *RetMem) {

  return pi2ur::piMemBufferPartition(Buffer, Flags, BufferCreateType,
                                     BufferCreateInfo, RetMem);
}

// TODO: Check if the function_pointer_ret type can be converted to void**.
pi_result piextGetDeviceFunctionPointer(pi_device Device, pi_program Program,
                                        const char *FunctionName,
                                        pi_uint64 *FunctionPointerRet) {

  return pi2ur::piextGetDeviceFunctionPointer(Device, Program, FunctionName,
                                              FunctionPointerRet);
}

pi_result piextUSMDeviceAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {

  return pi2ur::piextUSMDeviceAlloc(ResultPtr, Context, Device, Properties,
                                    Size, Alignment);
}

pi_result piextUSMSharedAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {

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

pi_result piextUSMHostAlloc(void **ResultPtr, pi_context Context,
                            pi_usm_mem_properties *Properties, size_t Size,
                            pi_uint32 Alignment) {
  return pi2ur::piextUSMHostAlloc(ResultPtr, Context, Properties, Size,
                                  Alignment);
}

pi_result piextUSMFree(pi_context Context, void *Ptr) {

  return pi2ur::piextUSMFree(Context, Ptr);
}

pi_result piextKernelSetArgPointer(pi_kernel Kernel, pi_uint32 ArgIndex,
                                   size_t ArgSize, const void *ArgValue) {
  return pi2ur::piextKernelSetArgPointer(Kernel, ArgIndex, ArgSize, ArgValue);
}

/// USM Memset API
///
/// @param Queue is the queue to submit to
/// @param Ptr is the ptr to memset
/// @param Value is value to set.  It is interpreted as an 8-bit value and the
/// upper
///        24 bits are ignored
/// @param Count is the size in bytes to memset
/// @param NumEventsInWaitlist is the number of events to wait on
/// @param EventsWaitlist is an array of events to wait on
/// @param Event is the event that represents this operation
pi_result piextUSMEnqueueMemset(pi_queue Queue, void *Ptr, pi_int32 Value,
                                size_t Count, pi_uint32 NumEventsInWaitlist,
                                const pi_event *EventsWaitlist,
                                pi_event *Event) {
  return pi2ur::piextUSMEnqueueMemset(
      Queue, Ptr, Value, Count, NumEventsInWaitlist, EventsWaitlist, Event);
}

pi_result piextUSMEnqueueMemcpy(pi_queue Queue, pi_bool Blocking, void *DstPtr,
                                const void *SrcPtr, size_t Size,
                                pi_uint32 NumEventsInWaitlist,
                                const pi_event *EventsWaitlist,
                                pi_event *Event) {

  return pi2ur::piextUSMEnqueueMemcpy(Queue, Blocking, DstPtr, SrcPtr, Size,
                                      NumEventsInWaitlist, EventsWaitlist,
                                      Event);
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
pi_result piextUSMEnqueuePrefetch(pi_queue Queue, const void *Ptr, size_t Size,
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
pi_result piextUSMEnqueueMemAdvise(pi_queue Queue, const void *Ptr,
                                   size_t Length, pi_mem_advice Advice,
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
    pi_uint32 NumEventsInWaitlist, const pi_event *EventWaitlist,
    pi_event *Event) {

  return pi2ur::piextUSMEnqueueMemcpy2D(
      Queue, Blocking, DstPtr, DstPitch, SrcPtr, SrcPitch, Width, Height,
      NumEventsInWaitlist, EventWaitlist, Event);
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
pi_result piextUSMGetMemAllocInfo(pi_context Context, const void *Ptr,
                                  pi_mem_alloc_info ParamName,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {
  return pi2ur::piextUSMGetMemAllocInfo(Context, Ptr, ParamName, ParamValueSize,
                                        ParamValue, ParamValueSizeRet);
}

pi_result piextUSMImport(const void *HostPtr, size_t Size, pi_context Context) {
  return pi2ur::piextUSMImport(HostPtr, Size, Context);
}

pi_result piextUSMRelease(const void *HostPtr, pi_context Context) {
  return pi2ur::piextUSMRelease(HostPtr, Context);
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

  return PI_SUCCESS;
}
/// API for Read from host pipe.
///
/// \param Queue is the queue
/// \param Program is the program containing the device variable
/// \param PipeSymbol is the unique identifier for the device variable
/// \param Blocking is true if the write should block
/// \param Ptr is a pointer to where the data will be copied to
/// \param Size is size of the data that is read/written from/to pipe
/// \param NumEventsInWaitList is a number of events in the wait list
/// \param EventWaitList is the wait list
/// \param Event is the resulting event
pi_result piextEnqueueReadHostPipe(pi_queue Queue, pi_program Program,
                                   const char *PipeSymbol, pi_bool Blocking,
                                   void *Ptr, size_t Size,
                                   pi_uint32 NumEventsInWaitList,
                                   const pi_event *EventsWaitList,
                                   pi_event *Event) {
  (void)Queue;
  (void)Program;
  (void)PipeSymbol;
  (void)Blocking;
  (void)Ptr;
  (void)Size;
  (void)NumEventsInWaitList;
  (void)EventsWaitList;
  (void)Event;

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  die("piextEnqueueReadHostPipe: not implemented");
  return {};
}

/// API for write to pipe of a given name.
///
/// \param Queue is the queue
/// \param Program is the program containing the device variable
/// \param PipeSymbol is the unique identifier for the device variable
/// \param Blocking is true if the write should block
/// \param Ptr is a pointer to where the data must be copied from
/// \param Size is size of the data that is read/written from/to pipe
/// \param NumEventsInWaitList is a number of events in the wait list
/// \param EventWaitList is the wait list
/// \param Event is the resulting event
pi_result piextEnqueueWriteHostPipe(pi_queue Queue, pi_program Program,
                                    const char *PipeSymbol, pi_bool Blocking,
                                    void *Ptr, size_t Size,
                                    pi_uint32 NumEventsInWaitList,
                                    const pi_event *EventsWaitList,
                                    pi_event *Event) {
  (void)Queue;
  (void)Program;
  (void)PipeSymbol;
  (void)Blocking;
  (void)Ptr;
  (void)Size;
  (void)NumEventsInWaitList;
  (void)EventsWaitList;
  (void)Event;

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  die("piextEnqueueWriteHostPipe: not implemented");
  return {};
}

pi_result piKernelSetExecInfo(pi_kernel Kernel, pi_kernel_exec_info ParamName,
                              size_t ParamValueSize, const void *ParamValue) {

  return pi2ur::piKernelSetExecInfo(Kernel, ParamName, ParamValueSize,
                                    ParamValue);
}

pi_result piextProgramSetSpecializationConstant(pi_program Prog,
                                                pi_uint32 SpecID, size_t Size,
                                                const void *SpecValue) {
  return pi2ur::piextProgramSetSpecializationConstant(Prog, SpecID, Size,
                                                      SpecValue);
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
    const pi_ext_sync_point *SyncPointWaitList, pi_ext_sync_point *SyncPoint) {
  return pi2ur::piextCommandBufferNDRangeKernel(
      CommandBuffer, Kernel, WorkDim, GlobalWorkOffset, GlobalWorkSize,
      LocalWorkSize, NumSyncPointsInWaitList, SyncPointWaitList, SyncPoint);
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

pi_result piextEnqueueCommandBuffer(pi_ext_command_buffer CommandBuffer,
                                    pi_queue Queue,
                                    pi_uint32 NumEventsInWaitList,
                                    const pi_event *EventWaitList,
                                    pi_event *Event) {
  return pi2ur::piextEnqueueCommandBuffer(
      CommandBuffer, Queue, NumEventsInWaitList, EventWaitList, Event);
}

const char SupportedVersion[] = _PI_LEVEL_ZERO_PLUGIN_VERSION_STRING;

pi_result piPluginInit(pi_plugin *PluginInit) { // missing
  PI_ASSERT(PluginInit, PI_ERROR_INVALID_VALUE);

  // Check that the major version matches in PiVersion and SupportedVersion
  _PI_PLUGIN_VERSION_CHECK(PluginInit->PiVersion, SupportedVersion);

  // TODO: handle versioning/targets properly.
  size_t PluginVersionSize = sizeof(PluginInit->PluginVersion);

  PI_ASSERT(strlen(_PI_LEVEL_ZERO_PLUGIN_VERSION_STRING) < PluginVersionSize,
            PI_ERROR_INVALID_VALUE);

  strncpy(PluginInit->PluginVersion, SupportedVersion, PluginVersionSize);

#define _PI_API(api)                                                           \
  (PluginInit->PiFunctionTable).api = (decltype(&::api))(&api);
#include <sycl/detail/pi.def>

  enableZeTracing();
  return PI_SUCCESS;
}

pi_result piextPluginGetOpaqueData(void *opaque_data_param,
                                   void **opaque_data_return) {
  return pi2ur::piextPluginGetOpaqueData(opaque_data_param, opaque_data_return);
}

// SYCL RT calls this api to notify the end of plugin lifetime.
// Windows: dynamically loaded plugins might have been unloaded already
// when this is called. Sycl RT holds onto the PI plugin so it can be
// called safely. But this is not transitive. If the PI plugin in turn
// dynamically loaded a different DLL, that may have been unloaded.
// It can include all the jobs to tear down resources before
// the plugin is unloaded from memory.
pi_result piTearDown(void *PluginParameter) {
  return pi2ur::piTearDown(PluginParameter);
}

pi_result piGetDeviceAndHostTimer(pi_device Device, uint64_t *DeviceTime,
                                  uint64_t *HostTime) {
  return pi2ur::piGetDeviceAndHostTimer(Device, DeviceTime, HostTime);
}

pi_result piextEnablePeerAccess(pi_device command_device,
                                pi_device peer_device) {

  return pi2ur::piextEnablePeerAccess(command_device, peer_device);
}

pi_result piextDisablePeerAccess(pi_device command_device,
                                 pi_device peer_device) {

  return pi2ur::piextDisablePeerAccess(command_device, peer_device);
}

pi_result piextPeerAccessGetInfo(pi_device command_device,
                                 pi_device peer_device, pi_peer_attr attr,
                                 size_t ParamValueSize, void *ParamValue,
                                 size_t *ParamValueSizeRet) {

  return pi2ur::piextPeerAccessGetInfo(command_device, peer_device, attr,
                                       ParamValueSize, ParamValue,
                                       ParamValueSizeRet);
}

#ifdef _WIN32
#define __SYCL_PLUGIN_DLL_NAME "pi_level_zero.dll"
#include "../common_win_pi_trace/common_win_pi_trace.hpp"
#undef __SYCL_PLUGIN_DLL_NAME
#endif
} // extern "C"
