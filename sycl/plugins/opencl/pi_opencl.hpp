//==---------- pi_opencl.hpp - OpenCL Plugin -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \defgroup sycl_pi_ocl OpenCL Plugin
/// \ingroup sycl_pi

/// \file pi_opencl.hpp
/// Declarations for vOpenCL Plugin. It is the interface between device-agnostic
/// SYCL runtime layer and underlying OpenCL runtime.
///
/// \ingroup sycl_pi_ocl

#ifndef PI_OPENCL_HPP
#define PI_OPENCL_HPP

// This version should be incremented for any change made to this file or its
// corresponding .cpp file.
#define _PI_OPENCL_PLUGIN_VERSION 1

#define _PI_OPENCL_PLUGIN_VERSION_STRING                                       \
  _PI_PLUGIN_VERSION_STRING(_PI_OPENCL_PLUGIN_VERSION)

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/sycl/detail/cl.h>
#include <CL/sycl/detail/pi.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_ERR_SET_NULL_RET(err, ptr, reterr)                               \
  if (err != CL_SUCCESS) {                                                     \
    if (ptr != nullptr)                                                        \
      *ptr = nullptr;                                                          \
    return cast<pi_result>(reterr);                                            \
  }

// Want all the needed casts be explicit, do not define conversion operators.
template <class To, class From> To cast(From value) {
  // TODO: see if more sanity checks are possible.
  static_assert(sizeof(From) == sizeof(To), "cast failed size check");
  return (To)(value);
}

// Older versions of GCC don't like "const" here
#if defined(__GNUC__) && (__GNUC__ < 7 || (__GNU__C == 7 && __GNUC_MINOR__ < 2))
#define CONSTFIX constexpr
#else
#define CONSTFIX const
#endif

// Names of USM functions that are queried from OpenCL
CONSTFIX char clHostMemAllocName[] = "clHostMemAllocINTEL";
CONSTFIX char clDeviceMemAllocName[] = "clDeviceMemAllocINTEL";
CONSTFIX char clSharedMemAllocName[] = "clSharedMemAllocINTEL";
CONSTFIX char clMemFreeName[] = "clMemFreeINTEL";
CONSTFIX char clMemBlockingFreeName[] = "clMemBlockingFreeINTEL";
CONSTFIX char clCreateBufferWithPropertiesName[] =
    "clCreateBufferWithPropertiesINTEL";
CONSTFIX char clSetKernelArgMemPointerName[] = "clSetKernelArgMemPointerINTEL";
CONSTFIX char clEnqueueMemsetName[] = "clEnqueueMemsetINTEL";
CONSTFIX char clEnqueueMemcpyName[] = "clEnqueueMemcpyINTEL";
CONSTFIX char clGetMemAllocInfoName[] = "clGetMemAllocInfoINTEL";
CONSTFIX char clSetProgramSpecializationConstantName[] =
    "clSetProgramSpecializationConstant";
CONSTFIX char clGetDeviceFunctionPointerName[] =
    "clGetDeviceFunctionPointerINTEL";

#undef CONSTFIX

// Global variables for PI_PLUGIN_SPECIFIC_ERROR
constexpr size_t MaxMessageSize = 256;
thread_local pi_result ErrorMessageCode = PI_SUCCESS;
thread_local char ErrorMessage[MaxMessageSize];

#endif // PI_OPENCL_HPP
