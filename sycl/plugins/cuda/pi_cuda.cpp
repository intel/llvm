//==---------- pi_cuda.cpp - CUDA Plugin -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi_cuda.cpp
/// Implementation of CUDA Plugin.
///
/// \ingroup sycl_pi_cuda

#include <pi_cuda.hpp>
#include <sycl/detail/cuda_definitions.hpp>
#include <sycl/detail/defines.hpp>
#include <sycl/detail/pi.hpp>

// Forward declarations
void enableCUDATracing();

//-- PI API implementation
extern "C" {

const char SupportedVersion[] = _PI_CUDA_PLUGIN_VERSION_STRING;

pi_result piPluginInit(pi_plugin *PluginInit) {
  // Check that the major version matches in PiVersion and SupportedVersion
  _PI_PLUGIN_VERSION_CHECK(PluginInit->PiVersion, SupportedVersion);

  // PI interface supports higher version or the same version.
  size_t PluginVersionSize = sizeof(PluginInit->PluginVersion);
  if (strlen(SupportedVersion) >= PluginVersionSize)
    return PI_ERROR_INVALID_VALUE;
  strncpy(PluginInit->PluginVersion, SupportedVersion, PluginVersionSize);

  // Set whole function table to zero to make it easier to detect if
  // functions are not set up below.
  std::memset(&(PluginInit->PiFunctionTable), 0,
              sizeof(PluginInit->PiFunctionTable));

  enableCUDATracing();

// Forward calls to CUDA RT.
#define _PI_CL(pi_api, cuda_api)                                               \
  (PluginInit->PiFunctionTable).pi_api = (decltype(&::pi_api))(&cuda_api);

  // Platform
  _PI_CL(piPlatformsGet, pi2ur::piPlatformsGet)
  _PI_CL(piPlatformGetInfo, pi2ur::piPlatformGetInfo)
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

  // USM
  _PI_CL(piextUSMHostAlloc, pi2ur::piextUSMHostAlloc)
  _PI_CL(piextUSMDeviceAlloc, pi2ur::piextUSMDeviceAlloc)
  _PI_CL(piextUSMSharedAlloc, pi2ur::piextUSMSharedAlloc)
  _PI_CL(piextUSMFree, pi2ur::piextUSMFree)
  _PI_CL(piextUSMEnqueueMemset, pi2ur::piextUSMEnqueueMemset)
  _PI_CL(piextUSMEnqueueMemcpy, pi2ur::piextUSMEnqueueMemcpy)
  _PI_CL(piextUSMEnqueuePrefetch, pi2ur::piextUSMEnqueuePrefetch)
  _PI_CL(piextUSMEnqueueMemAdvise, pi2ur::piextUSMEnqueueMemAdvise)
  _PI_CL(piextUSMEnqueueFill2D, pi2ur::piextUSMEnqueueFill2D)
  _PI_CL(piextUSMEnqueueMemset2D, pi2ur::piextUSMEnqueueMemset2D)
  _PI_CL(piextUSMEnqueueMemcpy2D, pi2ur::piextUSMEnqueueMemcpy2D)
  _PI_CL(piextUSMGetMemAllocInfo, pi2ur::piextUSMGetMemAllocInfo)
  // Device global variable
  _PI_CL(piextEnqueueDeviceGlobalVariableWrite,
         pi2ur::piextEnqueueDeviceGlobalVariableWrite)
  _PI_CL(piextEnqueueDeviceGlobalVariableRead,
         pi2ur::piextEnqueueDeviceGlobalVariableRead)

  // Host Pipe
  _PI_CL(piextEnqueueReadHostPipe, pi2ur::piextEnqueueReadHostPipe)
  _PI_CL(piextEnqueueWriteHostPipe, pi2ur::piextEnqueueWriteHostPipe)

  _PI_CL(piextKernelSetArgMemObj, pi2ur::piextKernelSetArgMemObj)
  _PI_CL(piextKernelSetArgSampler, pi2ur::piextKernelSetArgSampler)
  _PI_CL(piPluginGetLastError, pi2ur::piPluginGetLastError)
  _PI_CL(piTearDown, pi2ur::piTearDown)
  _PI_CL(piGetDeviceAndHostTimer, pi2ur::piGetDeviceAndHostTimer)
  _PI_CL(piPluginGetBackendOption, pi2ur::piPluginGetBackendOption)

#undef _PI_CL

  return PI_SUCCESS;
}

#ifdef _WIN32
#define __SYCL_PLUGIN_DLL_NAME "pi_cuda.dll"
#include "../common_win_pi_trace/common_win_pi_trace.hpp"
#undef __SYCL_PLUGIN_DLL_NAME
#endif

} // extern "C"
