//==---------------- cl.h - Include OpenCL headers -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/detail/ur.hpp>

// Suppress a compiler message about undefined CL_TARGET_OPENCL_VERSION
// and define all symbols up to OpenCL 3.0
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

// Include symbols for beta extensions
#ifndef CL_ENABLE_BETA_EXTENSIONS
#define CL_ENABLE_BETA_EXTENSIONS
#endif

// Don't include the OpenCL headers when compiling for the SYCL device with the
// internal API, as they only define the host-side API. The opaque type aliases
// defined below stand in so the SYCL headers that include this file still
// parse.
#if !defined(__SYCL_DEVICE_ONLY__) || !defined(__SYCL_INTERNAL_API)
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

namespace sycl {
inline namespace _V1 {
#if defined(__SYCL_DEVICE_ONLY__)
// Don't include the OpenCL headers when compiling for SYCL device, as they only
// define the host-side API. Instead, define the necessary types as opaque
// pointers to not break the SYCL headers that include this header.
using OpenCLCommandQueueT = void *;
using OpenCLContextT = void *;
using OpenCLDeviceIdT = void *;
using OpenCLEventT = void *;
using OpenCLKernelT = void *;
using OpenCLMemT = void *;
using OpenCLPlatformT = void *;
using OpenCLProgramT = void *;
using OpenCLSamplerT = void *;
namespace detail {
inline void retainOpenCLCommandQueue(ur_native_handle_t) {}
inline void retainOpenCLContext(ur_native_handle_t) {}
inline void retainOpenCLDevice(ur_native_handle_t) {}
inline void retainOpenCLEvent(ur_native_handle_t) {}
inline void retainOpenCLKernel(ur_native_handle_t) {}
inline void retainOpenCLMemObject(ur_native_handle_t) {}
inline void retainOpenCLProgram(ur_native_handle_t) {}
} // namespace detail
#else  // !defined(__SYCL_DEVICE_ONLY__)
using OpenCLCommandQueueT = cl_command_queue;
using OpenCLContextT = cl_context;
using OpenCLDeviceIdT = cl_device_id;
using OpenCLEventT = cl_event;
using OpenCLKernelT = cl_kernel;
using OpenCLMemT = cl_mem;
using OpenCLPlatformT = cl_platform_id;
using OpenCLProgramT = cl_program;
using OpenCLSamplerT = cl_sampler;
namespace detail {
inline void retainOpenCLCommandQueue(ur_native_handle_t Queue) {
  __SYCL_OCL_CALL(clRetainCommandQueue, ur::cast<OpenCLCommandQueueT>(Queue));
}
inline void retainOpenCLContext(ur_native_handle_t Context) {
  __SYCL_OCL_CALL(clRetainContext, ur::cast<OpenCLContextT>(Context));
}
inline void retainOpenCLDevice(ur_native_handle_t Device) {
  __SYCL_OCL_CALL(clRetainDevice, ur::cast<OpenCLDeviceIdT>(Device));
}
inline void retainOpenCLEvent(ur_native_handle_t Event) {
  __SYCL_OCL_CALL(clRetainEvent, ur::cast<OpenCLEventT>(Event));
}
inline void retainOpenCLKernel(ur_native_handle_t Kernel) {
  __SYCL_OCL_CALL(clRetainKernel, ur::cast<OpenCLKernelT>(Kernel));
}
inline void retainOpenCLMemObject(ur_native_handle_t MemObject) {
  __SYCL_OCL_CALL(clRetainMemObject, ur::cast<OpenCLMemT>(MemObject));
}
inline void retainOpenCLProgram(ur_native_handle_t Program) {
  __SYCL_OCL_CALL(clRetainProgram, ur::cast<OpenCLProgramT>(Program));
}
} // namespace detail
#endif // defined(__SYCL_DEVICE_ONLY__)
} // namespace _V1
} // namespace sycl
