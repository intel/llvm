//==---------- pi_opencl.hpp - OpenCL Plugin for SYCL RT -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <CL/opencl.h>
#include <CL/sycl/detail/pi.hpp>

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
class opencl {
public:
  using pi_result             = cl_int;
  using pi_platform           = cl_platform_id;
  using pi_device             = cl_device_id;
  using pi_device_type        = cl_device_type;
  using pi_device_binary_type = cl_device_binary_type;
  using pi_device_info        = cl_device_info;
  using pi_program            = cl_program;

  // Convinience macro to have mapping look like a compact table.
  #define PI_CL(pi_api, cl_api) \
    static constexpr decltype(cl_api) * pi_api = &cl_api;

  // Platform
  PI_CL(piPlatformsGet,       clGetPlatformIDs)
  PI_CL(piPlatformGetInfo,    clGetPlatformInfo)
  // Device
  PI_CL(piDevicesGet,         clGetDeviceIDs)
  PI_CL(piDeviceGetInfo,      clGetDeviceInfo)
  PI_CL(piDevicePartition,    clCreateSubDevices)
  PI_CL(piDeviceRetain,       clRetainDevice)
  PI_CL(piDeviceRelease,      clReleaseDevice)
  // IR
  PI_CL(piextDeviceSelectBinary,  ocl_piextDeviceSelectBinary)

  #undef PI_CL
};

} // namespace detail
} // namespace sycl
} // namespace cl
