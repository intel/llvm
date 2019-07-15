//==---------- pi_opencl.cpp - OpenCL Plugin -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "CL/opencl.h"
#include <CL/sycl/detail/pi.hpp>
#include <cassert>
#include <cstring>

namespace cl {
namespace sycl {
namespace detail {

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Convinience macro makes source code search easier
#define OCL(pi_api) ocl_##pi_api

// Example of a PI interface that does not map exactly to an OpenCL one.
pi_result OCL(piPlatformsGet)(pi_uint32      num_entries,
                              pi_platform *  platforms,
                              pi_uint32 *    num_platforms) {
  cl_int result =
    clGetPlatformIDs(pi_cast<cl_uint>           (num_entries),
                     pi_cast<cl_platform_id *>  (platforms),
                     pi_cast<cl_uint *>         (num_platforms));

  // Absorb the CL_PLATFORM_NOT_FOUND_KHR and just return 0 in num_platforms
  if (result == CL_PLATFORM_NOT_FOUND_KHR) {
    piAssert(num_platforms != 0);
    *num_platforms = 0;
    result = CL_SUCCESS;
  }
  return pi_cast<pi_result>(result);
}


// Example of a PI interface that does not map exactly to an OpenCL one.
pi_result OCL(piDevicesGet)(pi_platform      platform,
                            pi_device_type   device_type,
                            pi_uint32        num_entries,
                            pi_device *      devices,
                            pi_uint32 *      num_devices) {
  cl_int result =
    clGetDeviceIDs(pi_cast<cl_platform_id> (platform),
                   pi_cast<cl_device_type> (device_type),
                   pi_cast<cl_uint>        (num_entries),
                   pi_cast<cl_device_id *> (devices),
                   pi_cast<cl_uint *>      (num_devices));

  // Absorb the CL_DEVICE_NOT_FOUND and just return 0 in num_devices
  if (result == CL_DEVICE_NOT_FOUND) {
    piAssert(num_devices != 0);
    *num_devices = 0;
    result = CL_SUCCESS;
  }
  return pi_cast<pi_result>(result);
}

pi_result OCL(piextDeviceSelectBinary)(
  pi_device           device, // TODO: does this need to be context?
  pi_device_binary *  images,
  pi_uint32           num_images,
  pi_device_binary *  selected_image) {

  // TODO dummy implementation.
  // Real implementaion will use the same mechanism OpenCL ICD dispatcher
  // uses. Somthing like:
  //   PI_VALIDATE_HANDLE_RETURN_HANDLE(ctx, PI_INVALID_CONTEXT);
  //     return context->dispatch->piextDeviceSelectIR(
  //       ctx, images, num_images, selected_image);
  // where context->dispatch is set to the dispatch table provided by PI
  // plugin for platform/device the ctx was created for.

  *selected_image = num_images > 0 ? images[0] : nullptr;
  return PI_SUCCESS;
}

pi_program OCL(piProgramCreate)(pi_context context, const void *il,
                                size_t length, pi_result *err) {

  size_t deviceCount;
  cl_program resProgram;

  cl_int ret_err = clGetContextInfo(pi_cast<cl_context>(context),
                                    CL_CONTEXT_DEVICES, 0, NULL, &deviceCount);

  std::vector<cl_device_id> devicesInCtx;
  devicesInCtx.reserve(deviceCount);

  ret_err = clGetContextInfo(pi_cast<cl_context>(context), CL_CONTEXT_DEVICES,
                             deviceCount * sizeof(cl_device_id),
                             devicesInCtx.data(), NULL);

  if (ret_err != CL_SUCCESS || deviceCount < 1) {
    if (err != nullptr)
      *err = pi_cast<pi_result>(CL_INVALID_CONTEXT);
    return pi_cast<pi_program>(resProgram);
  }

  cl_platform_id curPlatform;
  ret_err = clGetDeviceInfo(devicesInCtx[0], CL_DEVICE_PLATFORM,
                            sizeof(cl_platform_id), &curPlatform, NULL);

  if (ret_err != CL_SUCCESS) {
    if (err != nullptr)
      *err = pi_cast<pi_result>(CL_INVALID_CONTEXT);
    return pi_cast<pi_program>(resProgram);
  }

  size_t devVerSize;
  ret_err =
      clGetPlatformInfo(curPlatform, CL_PLATFORM_VERSION, 0, NULL, &devVerSize);
  std::string devVer(devVerSize, '\0');
  ret_err = clGetPlatformInfo(curPlatform, CL_PLATFORM_VERSION, devVerSize,
                              &devVer.front(), NULL);

  if (ret_err != CL_SUCCESS) {
    if (err != nullptr)
      *err = pi_cast<pi_result>(CL_INVALID_CONTEXT);
    return pi_cast<pi_program>(resProgram);
  }

  if (devVer.find("OpenCL 1.0") == std::string::npos &&
      devVer.find("OpenCL 1.1") == std::string::npos &&
      devVer.find("OpenCL 1.2") == std::string::npos &&
      devVer.find("OpenCL 2.0") == std::string::npos) {
    resProgram = clCreateProgramWithIL(pi_cast<cl_context>(context), il, length,
                                       pi_cast<cl_int *>(err));
    return pi_cast<pi_program>(resProgram);
  }

  size_t extSize;
  ret_err = clGetPlatformInfo(curPlatform, CL_PLATFORM_EXTENSIONS, 0, NULL,
                                &extSize);
  std::string extStr(extSize, '\0');
  ret_err = clGetPlatformInfo(curPlatform, CL_PLATFORM_EXTENSIONS,
                                extSize, &extStr.front(), NULL);

  if (ret_err != CL_SUCCESS ||
      extStr.find("cl_khr_il_program") == std::string::npos) {
    if (err != nullptr)
      *err = pi_cast<pi_result>(CL_INVALID_CONTEXT);
    return pi_cast<pi_program>(resProgram);
  }

  using apiFuncT =
      cl_program(CL_API_CALL *)(cl_context, const void *, size_t, cl_int *);
  apiFuncT funcPtr =
      reinterpret_cast<apiFuncT>(clGetExtensionFunctionAddressForPlatform(
          curPlatform, "clCreateProgramWithILKHR"));

  assert(funcPtr != nullptr);
  resProgram = funcPtr(pi_cast<cl_context>(context), il, length,
                         pi_cast<cl_int *>(err));

  return pi_cast<pi_program>(resProgram);
}

// TODO: implement portable call forwarding (ifunc is a GNU extension).
// TODO: reuse same PI -> OCL mapping in pi_opencl.hpp, or maybe just
//       wait until that one is completely removed.
//
#define _PI_CL(pi_api, ocl_api)             \
static void *__resolve_##pi_api(void) {     \
  return (void*) (ocl_api);                 \
}                                           \
decltype(ocl_api) OCL(pi_api) __attribute__((ifunc ("__resolve_" #pi_api)));

// Platform
//_PI_CL(piPlatformsGet,       clGetPlatformIDs)
_PI_CL(piPlatformGetInfo,    clGetPlatformInfo)
// Device
//_PI_CL(piDevicesGet,         clGetDeviceIDs)
_PI_CL(piDeviceGetInfo,      clGetDeviceInfo)
_PI_CL(piDevicePartition,    clCreateSubDevices)
_PI_CL(piDeviceRetain,       clRetainDevice)
_PI_CL(piDeviceRelease,      clReleaseDevice)
//_PI_CL(piextDeviceSelectBinary,  ocl_piextDeviceSelectBinary)
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
//_PI_CL(piProgramCreate,             clCreateProgramWithIL)
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
_PI_CL(piEnqueueMemCopyRect,   clEnqueueCopyBufferRect)
_PI_CL(piEnqueueMemFill,          clEnqueueFillBuffer)
_PI_CL(piEnqueueMemMap,           clEnqueueMapBuffer)
_PI_CL(piEnqueueMemUnmap,         clEnqueueUnmapMemObject)

#undef _PI_CL

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

} // namespace detail
} // namespace sycl
} // namespace cl
