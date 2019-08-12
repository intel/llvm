//==------------------- findplatforms.hpp ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
bool findPlatformAndDevice(cl_device_type deviceType,
                           cl_platform_id &platformOut, cl_device_id &deviceOut) {
  cl_uint numPlatforms;
  cl_int errorCode;

  errorCode = clGetPlatformIDs(0, nullptr, &numPlatforms);
  if (errorCode != CL_SUCCESS) return false;

  std::vector<cl_platform_id> platforms(numPlatforms);
  errorCode = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  if (errorCode != CL_SUCCESS) return false;

  for (auto platform : platforms) {
    cl_uint numDevices = 0;
    errorCode =
      clGetDeviceIDs(platform, deviceType, 0, nullptr, &numDevices);
    if (errorCode != CL_SUCCESS) return false;

    std::vector<cl_device_id> devices(numDevices);
    errorCode = clGetDeviceIDs(platform, deviceType, numDevices,
                               devices.data(), nullptr);
    if (errorCode != CL_SUCCESS) return false;

    if (numDevices) {
      platformOut = platform;
      deviceOut = devices[0];
      return true;
    }
  }

  return false;
}
