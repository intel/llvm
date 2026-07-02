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

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_INTERNAL_API)
// Don't include the OpenCL headers when compiling for SYCL device, as they only
// define the host-side API. Instead, define the necessary types as opaque
// pointers to not break the SYCL headers that include this header.
using cl_command_queue = void *;
using cl_context = void *;
using cl_device_id = void *;
using cl_event = void *;
using cl_kernel = void *;
using cl_mem = void *;
using cl_platform_id = void *;
using cl_program = void *;
using cl_sampler = void *;
#else // !defined(__SYCL_DEVICE_ONLY__) || !defined(__SYCL_JIT__)
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_JIT__)

namespace sycl {
inline namespace _V1 {
namespace detail {
// Thin wrappers around the OpenCL `clRetain*` entry points. Every direct call
// into the host OpenCL runtime (via __SYCL_OCL_CALL) is consolidated here so
// that these calls can be compiled out entirely when building for a SYCL
// device, where there is no host OpenCL library to dynamically load. Each
// helper takes a `ur_native_handle_t` and casts it to the corresponding
// OpenCL handle type internally.
#ifndef __SYCL_DEVICE_ONLY__
inline void retainOpenCLCommandQueue(ur_native_handle_t Queue) {
  __SYCL_OCL_CALL(clRetainCommandQueue, ur::cast<cl_command_queue>(Queue));
}
inline void retainOpenCLContext(ur_native_handle_t Context) {
  __SYCL_OCL_CALL(clRetainContext, ur::cast<cl_context>(Context));
}
inline void retainOpenCLDevice(ur_native_handle_t Device) {
  __SYCL_OCL_CALL(clRetainDevice, ur::cast<cl_device_id>(Device));
}
inline void retainOpenCLEvent(ur_native_handle_t Event) {
  __SYCL_OCL_CALL(clRetainEvent, ur::cast<cl_event>(Event));
}
inline void retainOpenCLKernel(ur_native_handle_t Kernel) {
  __SYCL_OCL_CALL(clRetainKernel, ur::cast<cl_kernel>(Kernel));
}
inline void retainOpenCLMemObject(ur_native_handle_t MemObject) {
  __SYCL_OCL_CALL(clRetainMemObject, ur::cast<cl_mem>(MemObject));
}
inline void retainOpenCLProgram(ur_native_handle_t Program) {
  __SYCL_OCL_CALL(clRetainProgram, ur::cast<cl_program>(Program));
}
#else  // __SYCL_DEVICE_ONLY__
inline void retainOpenCLCommandQueue(ur_native_handle_t) {}
inline void retainOpenCLContext(ur_native_handle_t) {}
inline void retainOpenCLDevice(ur_native_handle_t) {}
inline void retainOpenCLEvent(ur_native_handle_t) {}
inline void retainOpenCLKernel(ur_native_handle_t) {}
inline void retainOpenCLMemObject(ur_native_handle_t) {}
inline void retainOpenCLProgram(ur_native_handle_t) {}
#endif // __SYCL_DEVICE_ONLY__
} // namespace detail
} // namespace _V1
} // namespace sycl
