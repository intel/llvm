#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sycl/detail/cg_types.hpp> // NDRDescT
#include <sycl/detail/native_cpu.hpp>
#include <sycl/detail/pi.h>

#include "pi_native_cpu.hpp"

static bool PrintPiTrace = true;

// taken from pi_cuda.cpp
template <typename T, typename Assign>
pi_result getInfoImpl(size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret, T value, size_t value_size,
                      Assign &&assign_func) {

  if (param_value != nullptr) {
    if (param_value_size < value_size) {
      return PI_ERROR_INVALID_VALUE;
    }

    assign_func(param_value, value, value_size);
  }

  if (param_value_size_ret != nullptr) {
    *param_value_size_ret = value_size;
  }

  return PI_SUCCESS;
}

template <typename T>
pi_result getInfo(size_t param_value_size, void *param_value,
                  size_t *param_value_size_ret, T value) {

  auto assignment = [](void *param_value, T value, size_t value_size) {
    // Ignore unused parameter
    (void)value_size;

    *static_cast<T *>(param_value) = value;
  };

  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     sizeof(T), assignment);
}

template <typename T>
pi_result getInfoArray(size_t array_length, size_t param_value_size,
                       void *param_value, size_t *param_value_size_ret,
                       T *value) {
  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     array_length * sizeof(T), memcpy);
}

extern "C" {
#define DIE_NO_IMPLEMENTATION                                                  \
  if (PrintPiTrace) {                                                          \
    std::cerr << "Not Implemented : " << __FUNCTION__                          \
              << " - File : " << __FILE__;                                     \
    std::cerr << " / Line : " << __LINE__ << std::endl;                        \
  }                                                                            \
  return PI_ERROR_INVALID_OPERATION;

#define CONTINUE_NO_IMPLEMENTATION                                             \
  if (PrintPiTrace) {                                                          \
    std::cerr << "Warning : Not Implemented : " << __FUNCTION__                \
              << " - File : " << __FILE__;                                     \
    std::cerr << " / Line : " << __LINE__ << std::endl;                        \
  }                                                                            \
  return PI_SUCCESS;

#define CASE_PI_UNSUPPORTED(not_supported)                                     \
  case not_supported:                                                          \
    if (PrintPiTrace) {                                                        \
      std::cerr << std::endl                                                   \
                << "Unsupported PI case : " << #not_supported << " in "        \
                << __FUNCTION__ << ":" << __LINE__ << "(" << __FILE__ << ")"   \
                << std::endl;                                                  \
    }                                                                          \
    return PI_ERROR_INVALID_OPERATION;

pi_result piextUSMHostAlloc(void **result_ptr, pi_context,
                            pi_usm_mem_properties *, size_t size, pi_uint32) {
  // Todo: check properties and alignment.
  // Todo: error checking.
  *result_ptr = malloc(size);
  return PI_SUCCESS;
}

pi_result piextUSMDeviceAlloc(void **ResultPtr, pi_context, pi_device,
                              pi_usm_mem_properties *, size_t Size, pi_uint32) {
  // Todo: check properties and alignment.
  // Todo: error checking.
  *ResultPtr = malloc(Size);
  return PI_SUCCESS;
}

pi_result piextUSMSharedAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMFree(pi_context, void *Ptr) {
  // Todo: error checking
  free(Ptr);
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemset(pi_queue, void *ptr, pi_int32 value,
                                size_t count, pi_uint32, const pi_event *,
                                pi_event *) {
  // Todo: event dependency
  // Todo: error checking.
  memset(ptr, value, count);
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemcpy(pi_queue, pi_bool, void *dest, const void *src,
                                size_t len, pi_uint32, const pi_event *,
                                pi_event *) {
  // Todo: event dependency
  // Todo: error checking
  memcpy(dest, src, len);
  return PI_SUCCESS;
}

pi_result piextUSMEnqueueMemAdvise(pi_queue, const void *, size_t,
                                   pi_mem_advice, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMGetMemAllocInfo(pi_context, const void *, pi_mem_alloc_info,
                                  size_t, void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMEnqueuePrefetch(pi_queue, const void *, size_t,
                                  pi_usm_migration_flags, pi_uint32,
                                  const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMEnqueueMemcpy2D(pi_queue queue, pi_bool blocking,
                                  void *dst_ptr, size_t dst_pitch,
                                  const void *src_ptr, size_t src_pitch,
                                  size_t width, size_t height,
                                  pi_uint32 num_events_in_waitlist,
                                  const pi_event *events_waitlist,
                                  pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMEnqueueMemset2D(pi_queue queue, void *ptr, size_t pitch,
                                  int value, size_t width, size_t height,
                                  pi_uint32 num_events_in_waitlist,
                                  const pi_event *events_waitlist,
                                  pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextUSMEnqueueFill2D(pi_queue queue, void *ptr, size_t pitch,
                                size_t pattern_size, const void *pattern,
                                size_t width, size_t height,
                                pi_uint32 num_events_in_waitlist,
                                const pi_event *events_waitlist,
                                pi_event *event) {
  DIE_NO_IMPLEMENTATION;
}
pi_result piextCommandBufferCreate(pi_context, pi_device,
                                   const pi_ext_command_buffer_desc *,
                                   pi_ext_command_buffer *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferRetain(pi_ext_command_buffer) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferRelease(pi_ext_command_buffer) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferFinalize(pi_ext_command_buffer) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferNDRangeKernel(pi_ext_command_buffer, pi_kernel,
                                          pi_uint32, const size_t *,
                                          const size_t *, const size_t *,
                                          pi_uint32, const pi_ext_sync_point *,
                                          pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemcpyUSM(pi_ext_command_buffer, void *,
                                      const void *, size_t, pi_uint32,
                                      const pi_ext_sync_point *,
                                      pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferCopy(pi_ext_command_buffer, pi_mem, pi_mem,
                                          size_t, size_t, size_t, pi_uint32,
                                          const pi_ext_sync_point *,
                                          pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferCopyRect(
    pi_ext_command_buffer, pi_mem, pi_mem, pi_buff_rect_offset,
    pi_buff_rect_offset, pi_buff_rect_region, size_t, size_t, size_t, size_t,
    pi_uint32, const pi_ext_sync_point *, pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferRead(pi_ext_command_buffer, pi_mem, size_t,
                                          size_t, void *, pi_uint32,
                                          const pi_ext_sync_point *,
                                          pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferReadRect(
    pi_ext_command_buffer, pi_mem, pi_buff_rect_offset, pi_buff_rect_offset,
    pi_buff_rect_region, size_t, size_t, size_t, size_t, void *, pi_uint32,
    const pi_ext_sync_point *, pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferWrite(pi_ext_command_buffer, pi_mem,
                                           size_t, size_t, const void *,
                                           pi_uint32, const pi_ext_sync_point *,
                                           pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextCommandBufferMemBufferWriteRect(
    pi_ext_command_buffer, pi_mem, pi_buff_rect_offset, pi_buff_rect_offset,
    pi_buff_rect_region, size_t, size_t, size_t, size_t, const void *,
    pi_uint32, const pi_ext_sync_point *, pi_ext_sync_point *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEnqueueCommandBuffer(pi_ext_command_buffer, pi_queue, pi_uint32,
                                    const pi_event *, pi_event *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextEnablePeerAccess(pi_device, pi_device) { DIE_NO_IMPLEMENTATION; }

pi_result piextDisablePeerAccess(pi_device, pi_device) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piextPeerAccessGetInfo(pi_device, pi_device, pi_peer_attr, size_t,
                                 void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

pi_result piTearDown(void *) {
  // Todo: is it fine as a no-op?
  return PI_SUCCESS;
}

pi_result piPluginInit(pi_plugin *PluginInit) {

#define _PI_CL(pi_api, native_cpu_api)                                         \
  (PluginInit->PiFunctionTable).pi_api = (decltype(&::pi_api))(&native_cpu_api);

  // Platform
  _PI_CL(piPlatformsGet, pi2ur::piPlatformsGet)
  _PI_CL(piPlatformGetInfo, pi2ur::piPlatformGetInfo)
  _PI_CL(piPluginGetBackendOption, pi2ur::piPluginGetBackendOption)

  // Device
  _PI_CL(piDevicesGet, pi2ur::piDevicesGet)
  _PI_CL(piDeviceGetInfo, pi2ur::piDeviceGetInfo)
  _PI_CL(piDevicePartition, pi2ur::piDevicePartition)
  _PI_CL(piDeviceRetain, pi2ur::piDeviceRetain)
  _PI_CL(piDeviceRelease, pi2ur::piDeviceRelease)
  _PI_CL(piextDeviceSelectBinary, pi2ur::piextDeviceSelectBinary)
  _PI_CL(piextGetDeviceFunctionPointer, pi2ur::piextGetDeviceFunctionPointer)
  _PI_CL(piextDeviceGetNativeHandle, pi2ur::piextDeviceGetNativeHandle)
  _PI_CL(piextDeviceCreateWithNativeHandle,
         pi2ur::piextDeviceCreateWithNativeHandle)
  _PI_CL(piGetDeviceAndHostTimer, pi2ur::piGetDeviceAndHostTimer)

  // Context
  _PI_CL(piextContextSetExtendedDeleter, pi2ur::piextContextSetExtendedDeleter)
  _PI_CL(piContextCreate, pi2ur::piContextCreate)
  _PI_CL(piContextGetInfo, pi2ur::piContextGetInfo)
  _PI_CL(piContextRetain, pi2ur::piContextRetain)
  _PI_CL(piContextRelease, pi2ur::piContextRelease)
  _PI_CL(piextContextGetNativeHandle, pi2ur::piextContextGetNativeHandle)
  _PI_CL(piextContextCreateWithNativeHandle,
         pi2ur::piextContextCreateWithNativeHandle)
  // Queue
  _PI_CL(piQueueCreate, pi2ur::piQueueCreate)
  _PI_CL(piextQueueCreate, pi2ur::piextQueueCreate)
  _PI_CL(piQueueGetInfo, pi2ur::piQueueGetInfo)
  _PI_CL(piQueueFinish, pi2ur::piQueueFinish)
  _PI_CL(piQueueFlush, pi2ur::piQueueFlush)
  _PI_CL(piQueueRetain, pi2ur::piQueueRetain)
  _PI_CL(piQueueRelease, pi2ur::piQueueRelease)
  _PI_CL(piextQueueGetNativeHandle, pi2ur::piextQueueGetNativeHandle)
  _PI_CL(piextQueueCreateWithNativeHandle,
         pi2ur::piextQueueCreateWithNativeHandle)

  // Memory
  _PI_CL(piMemBufferCreate, pi2ur::piMemBufferCreate)
  _PI_CL(piMemImageCreate, pi2ur::piMemImageCreate)
  _PI_CL(piMemGetInfo, pi2ur::piMemGetInfo)
  _PI_CL(piMemImageGetInfo, pi2ur::piMemImageGetInfo)
  _PI_CL(piMemRetain, pi2ur::piMemRetain)
  _PI_CL(piMemRelease, pi2ur::piMemRelease)
  _PI_CL(piMemBufferPartition, pi2ur::piMemBufferPartition)
  _PI_CL(piextMemGetNativeHandle, pi2ur::piextMemGetNativeHandle)
  _PI_CL(piextMemCreateWithNativeHandle, pi2ur::piextMemCreateWithNativeHandle)

  // Program
  _PI_CL(piProgramCreate, pi2ur::piProgramCreate)
  _PI_CL(piclProgramCreateWithSource, pi2ur::piclProgramCreateWithSource)
  _PI_CL(piProgramCreateWithBinary, pi2ur::piProgramCreateWithBinary)
  _PI_CL(piProgramGetInfo, pi2ur::piProgramGetInfo)
  _PI_CL(piProgramCompile, pi2ur::piProgramCompile)
  _PI_CL(piProgramBuild, pi2ur::piProgramBuild)
  _PI_CL(piProgramLink, pi2ur::piProgramLink)
  _PI_CL(piProgramGetBuildInfo, pi2ur::piProgramGetBuildInfo)
  _PI_CL(piProgramRetain, pi2ur::piProgramRetain)
  _PI_CL(piProgramRelease, pi2ur::piProgramRelease)
  _PI_CL(piextProgramGetNativeHandle, pi2ur::piextProgramGetNativeHandle)
  _PI_CL(piextProgramCreateWithNativeHandle,
         pi2ur::piextProgramCreateWithNativeHandle)
  _PI_CL(piextProgramSetSpecializationConstant,
         pi2ur::piextProgramSetSpecializationConstant)

  // Kernel
  _PI_CL(piKernelCreate, pi2ur::piKernelCreate)
  _PI_CL(piKernelSetArg, pi2ur::piKernelSetArg)
  _PI_CL(piKernelGetInfo, pi2ur::piKernelGetInfo)
  _PI_CL(piKernelGetGroupInfo, pi2ur::piKernelGetGroupInfo)
  _PI_CL(piKernelGetSubGroupInfo, pi2ur::piKernelGetSubGroupInfo)
  _PI_CL(piKernelRetain, pi2ur::piKernelRetain)
  _PI_CL(piKernelRelease, pi2ur::piKernelRelease)
  _PI_CL(piextKernelGetNativeHandle, pi2ur::piextKernelGetNativeHandle)
  _PI_CL(piKernelSetExecInfo, pi2ur::piKernelSetExecInfo)
  _PI_CL(piextKernelSetArgPointer, pi2ur::piKernelSetArgPointer)
  _PI_CL(piextKernelCreateWithNativeHandle,
         pi2ur::piextKernelCreateWithNativeHandle)
  _PI_CL(piextKernelSetArgMemObj, pi2ur::piextKernelSetArgMemObj)
  _PI_CL(piextKernelSetArgSampler, pi2ur::piextKernelSetArgSampler)

  // Event
  _PI_CL(piEventCreate, pi2ur::piEventCreate)
  _PI_CL(piEventGetInfo, pi2ur::piEventGetInfo)
  _PI_CL(piEventGetProfilingInfo, pi2ur::piEventGetProfilingInfo)
  _PI_CL(piEventsWait, pi2ur::piEventsWait)
  _PI_CL(piEventSetCallback, pi2ur::piEventSetCallback)
  _PI_CL(piEventSetStatus, pi2ur::piEventSetStatus)
  _PI_CL(piEventRetain, pi2ur::piEventRetain)
  _PI_CL(piEventRelease, pi2ur::piEventRelease)
  _PI_CL(piextEventGetNativeHandle, pi2ur::piextEventGetNativeHandle)
  _PI_CL(piextEventCreateWithNativeHandle,
         pi2ur::piextEventCreateWithNativeHandle)

  // Sampler
  _PI_CL(piSamplerCreate, pi2ur::piSamplerCreate)
  _PI_CL(piSamplerGetInfo, pi2ur::piSamplerGetInfo)
  _PI_CL(piSamplerRetain, pi2ur::piSamplerRetain)
  _PI_CL(piSamplerRelease, pi2ur::piSamplerRelease)

  // Queue commands
  _PI_CL(piEnqueueKernelLaunch, pi2ur::piEnqueueKernelLaunch)
  _PI_CL(piEnqueueNativeKernel, pi2ur::piEnqueueNativeKernel)
  _PI_CL(piEnqueueEventsWait, pi2ur::piEnqueueEventsWait)
  _PI_CL(piEnqueueEventsWaitWithBarrier, pi2ur::piEnqueueEventsWaitWithBarrier)
  _PI_CL(piEnqueueMemBufferRead, pi2ur::piEnqueueMemBufferRead)
  _PI_CL(piEnqueueMemBufferReadRect, pi2ur::piEnqueueMemBufferReadRect)
  _PI_CL(piEnqueueMemBufferWrite, pi2ur::piEnqueueMemBufferWrite)
  _PI_CL(piEnqueueMemBufferWriteRect, pi2ur::piEnqueueMemBufferWriteRect)
  _PI_CL(piEnqueueMemBufferCopy, pi2ur::piEnqueueMemBufferCopy)
  _PI_CL(piEnqueueMemBufferCopyRect, pi2ur::piEnqueueMemBufferCopyRect)
  _PI_CL(piEnqueueMemBufferFill, pi2ur::piEnqueueMemBufferFill)
  _PI_CL(piEnqueueMemImageRead, pi2ur::piEnqueueMemImageRead)
  _PI_CL(piEnqueueMemImageWrite, pi2ur::piEnqueueMemImageWrite)
  _PI_CL(piEnqueueMemImageCopy, pi2ur::piEnqueueMemImageCopy)
  _PI_CL(piEnqueueMemImageFill, pi2ur::piEnqueueMemImageFill)
  _PI_CL(piEnqueueMemBufferMap, pi2ur::piEnqueueMemBufferMap)
  _PI_CL(piEnqueueMemUnmap, pi2ur::piEnqueueMemUnmap)
  // Device global variable
  _PI_CL(piextEnqueueDeviceGlobalVariableWrite,
         pi2ur::piextEnqueueDeviceGlobalVariableWrite)
  _PI_CL(piextEnqueueDeviceGlobalVariableRead,
         pi2ur::piextEnqueueDeviceGlobalVariableRead)
  // Host Pipe
  _PI_CL(piextEnqueueReadHostPipe, pi2ur::piextEnqueueReadHostPipe)
  _PI_CL(piextEnqueueWriteHostPipe, pi2ur::piextEnqueueWriteHostPipe)

  // USM
  _PI_CL(piextUSMHostAlloc, piextUSMHostAlloc)
  _PI_CL(piextUSMDeviceAlloc, piextUSMDeviceAlloc)
  _PI_CL(piextUSMSharedAlloc, piextUSMSharedAlloc)
  _PI_CL(piextUSMFree, piextUSMFree)
  _PI_CL(piextUSMEnqueueMemset, pi2ur::piextUSMEnqueueMemset)
  _PI_CL(piextUSMEnqueueMemcpy, pi2ur::piextUSMEnqueueMemcpy)
  _PI_CL(piextUSMEnqueuePrefetch, pi2ur::piextUSMEnqueuePrefetch)
  _PI_CL(piextUSMEnqueueMemAdvise, pi2ur::piextUSMEnqueueMemAdvise)
  _PI_CL(piextUSMEnqueueFill2D, pi2ur::piextUSMEnqueueFill2D)
  _PI_CL(piextUSMEnqueueMemset2D, pi2ur::piextUSMEnqueueMemset2D)
  _PI_CL(piextUSMEnqueueMemcpy2D, pi2ur::piextUSMEnqueueMemcpy2D)
  _PI_CL(piextUSMGetMemAllocInfo, piextUSMGetMemAllocInfo)

  _PI_CL(piPluginGetLastError, pi2ur::piPluginGetLastError)

  // P2P Access
  _PI_CL(piextEnablePeerAccess, piextEnablePeerAccess)
  _PI_CL(piextDisablePeerAccess, piextDisablePeerAccess)
  _PI_CL(piextPeerAccessGetInfo, piextPeerAccessGetInfo)

  // Runtime
  _PI_CL(piTearDown, pi2ur::piTearDown)

#undef _PI_CL
  return PI_SUCCESS;
}
}
