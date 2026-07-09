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

// FIXME: Drop OpenCL headers from both host and device.
// Can not do that under preview breaking changes, because
// SYCL CTS use -fpreview-breaking-changes and rely on
// SYCL headers to include OpenCL headers. We need to fix
// such tests before we can drop OpenCL headers.
#ifndef __SYCL_EXCLUDE_OCL_HDR__
#include <CL/cl.h>
#include <CL/cl_ext.h>
#else
// Forward declare the opaque handle types used by SYCL headers.
// We should not do "using cl_command_queue = void*" because that
// can cause redifinition errors if the user application also
// includes OpenCL headers.
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_context *cl_context;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_event *cl_event;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_program *cl_program;
typedef struct _cl_sampler *cl_sampler;
#endif

namespace sycl {
inline namespace _V1 {
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
#ifdef __SYCL_DEVICE_ONLY__
inline void retainOpenCLCommandQueue(ur_native_handle_t) {}
inline void retainOpenCLContext(ur_native_handle_t) {}
inline void retainOpenCLDevice(ur_native_handle_t) {}
inline void retainOpenCLEvent(ur_native_handle_t) {}
inline void retainOpenCLKernel(ur_native_handle_t) {}
inline void retainOpenCLMemObject(ur_native_handle_t) {}
inline void retainOpenCLProgram(ur_native_handle_t) {}
#else  // __SYCL_DEVICE_ONLY__
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
#endif // defined(__SYCL_DEVICE_ONLY__)
} // namespace detail
} // namespace _V1
} // namespace sycl
