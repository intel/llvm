//==---------- pi_opencl.hpp - OpenCL substitute for PI --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is doing temporary redirection of PI to OpenCL at compile-time.
// TODO: when PI is ready get rid of this file.
//
#pragma once

#include <CL/opencl.h>
#include <CL/sycl/detail/pi.h>

namespace cl {
namespace sycl {
namespace detail {

//
// TODO: there is no such functionality in OpenCL so call PI OpenCL
// plugin directly for now, the whole "opencl" class is temporary anyway.
//
extern "C" decltype(::piextDeviceSelectBinary) ocl_piextDeviceSelectBinary;
using cl_device_binary_type = ::pi_device_binary_type;

// Mapping of PI interfaces to OpenCL at compile-time.
// This is the default config until the entire SYCL RT is transferred to PI.
// TODO: we can just remove this when default is change to PI.
//
namespace pi_opencl {

  using PiResult              = cl_int;
  using PiPlatform            = cl_platform_id;
  using PiDevice              = cl_device_id;
  using PiDeviceType          = cl_device_type;
  using PiDeviceInfo          = cl_device_info;
  using PiDeviceBinaryType    = cl_device_binary_type;
  using PiContext             = cl_context;
  using PiProgram             = cl_program;
  using PiKernel              = cl_kernel;
  using PiQueue               = cl_command_queue;
  using PiMem                 = cl_mem;
  using PiMemFlags            = cl_mem_flags;
  using PiEvent               = cl_event;
  using PiSampler             = cl_sampler;

  // Convinience macro to have mapping look like a compact table.
  #define _PI_CL(pi_api, cl_api) \
    static constexpr decltype(cl_api) * pi_api = &cl_api;

  // Platform
  _PI_CL(piPlatformsGet,       clGetPlatformIDs)
  _PI_CL(piPlatformGetInfo,    clGetPlatformInfo)
  // Device
  _PI_CL(piDevicesGet,         clGetDeviceIDs)
  _PI_CL(piDeviceGetInfo,      clGetDeviceInfo)
  _PI_CL(piDevicePartition,    clCreateSubDevices)
  _PI_CL(piDeviceRetain,       clRetainDevice)
  _PI_CL(piDeviceRelease,      clReleaseDevice)
  _PI_CL(piextDeviceSelectBinary,  ocl_piextDeviceSelectBinary)
    // Context
  _PI_CL(piContextCreate,     clCreateContext)
  _PI_CL(piContextGetInfo,    clGetContextInfo)
  _PI_CL(piContextRetain,     clRetainContext)
  _PI_CL(piContextRelease,    clReleaseContext)
  // Queue
  _PI_CL(piQueueCreate,       clCreateCommandQueueWithProperties)
  _PI_CL(piQueueGetInfo,      clGetCommandQueueInfo)
  _PI_CL(piQueueFinish,       clFinish)
  _PI_CL(piQueueRetain,       clRetainCommandQueue)
  _PI_CL(piQueueRelease,      clReleaseCommandQueue)
  // Memory
  _PI_CL(piMemCreate,         clCreateBuffer)
  _PI_CL(piMemGetInfo,        clGetMemObjectInfo)
  _PI_CL(piMemRetain,         clRetainMemObject)
  _PI_CL(piMemRelease,        clReleaseMemObject)
  // Program
  _PI_CL(piProgramCreate,             clCreateProgramWithIL)
  _PI_CL(piclProgramCreateWithSource, clCreateProgramWithSource)
  _PI_CL(piclProgramCreateWithBinary, clCreateProgramWithBinary)
  _PI_CL(piProgramGetInfo,            clGetProgramInfo)
  _PI_CL(piProgramCompile,            clCompileProgram)
  _PI_CL(piProgramBuild,              clBuildProgram)
  _PI_CL(piProgramLink,               clLinkProgram)
  _PI_CL(piProgramGetBuildInfo,       clGetProgramBuildInfo)
  _PI_CL(piProgramRetain,             clRetainProgram)
  _PI_CL(piProgramRelease,            clReleaseProgram)
  // Kernel
  _PI_CL(piKernelCreate,          clCreateKernel)
  _PI_CL(piKernelSetArg,          clSetKernelArg)
  _PI_CL(piKernelGetInfo,         clGetKernelInfo)
  _PI_CL(piKernelGetGroupInfo,    clGetKernelWorkGroupInfo)
  _PI_CL(piKernelGetSubGroupInfo, clGetKernelSubGroupInfo)
  _PI_CL(piKernelRetain,          clRetainKernel)
  _PI_CL(piKernelRelease,         clReleaseKernel)
  // Event
  _PI_CL(piEventCreate,           clCreateUserEvent)
  _PI_CL(piEventGetInfo,          clGetEventInfo)
  _PI_CL(piEventGetProfilingInfo, clGetEventProfilingInfo)
  _PI_CL(piEventsWait,            clWaitForEvents)
  _PI_CL(piEventSetCallback,      clSetEventCallback)
  _PI_CL(piEventSetStatus,        clSetUserEventStatus)
  _PI_CL(piEventRetain,           clRetainEvent)
  _PI_CL(piEventRelease,          clReleaseEvent)
  // Sampler
  _PI_CL(piSamplerCreate,         clCreateSamplerWithProperties)
  _PI_CL(piSamplerGetInfo,        clGetSamplerInfo)
  _PI_CL(piSamplerRetain,         clRetainSampler)
  _PI_CL(piSamplerRelease,        clReleaseSampler)
  // Queue commands
  _PI_CL(piEnqueueKernelLaunch,     clEnqueueNDRangeKernel)
  _PI_CL(piEnqueueEventsWait,       clEnqueueMarkerWithWaitList)
  _PI_CL(piEnqueueMemRead,          clEnqueueReadBuffer)
  _PI_CL(piEnqueueMemReadRect,      clEnqueueReadBufferRect)
  _PI_CL(piEnqueueMemWrite,         clEnqueueWriteBuffer)
  _PI_CL(piEnqueueMemWriteRect,     clEnqueueWriteBufferRect)
  _PI_CL(piEnqueueMemCopy,          clEnqueueCopyBuffer)
  _PI_CL(piEnqueueMemCopyRect,      clEnqueueCopyBufferRect)
  _PI_CL(piEnqueueMemFill,          clEnqueueFillBuffer)
  _PI_CL(piEnqueueMemMap,           clEnqueueMapBuffer)
  _PI_CL(piEnqueueMemUnmap,         clEnqueueUnmapMemObject)

  #undef _PI_CL
} // namespace pi_opencl

} // namespace detail
} // namespace sycl
} // namespace cl

