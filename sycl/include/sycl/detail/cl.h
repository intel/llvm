//==---------------- cl.h - Include OpenCL headers -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Suppress a compiler message about undefined CL_TARGET_OPENCL_VERSION
// and define all symbols up to OpenCL 3.0
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

// Include symbols for beta extensions
#ifndef CL_ENABLE_BETA_EXTENSIONS
#define CL_ENABLE_BETA_EXTENSIONS
#endif

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_JIT__)
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
