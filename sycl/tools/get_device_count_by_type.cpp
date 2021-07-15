//==-- get_device_count_by_type.cpp - Get device count by type -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Suppress a compiler warning about undefined CL_TARGET_OPENCL_VERSION
// and define all symbols up to OpenCL 2.2
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 220
#endif

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <level_zero/zet_api.h>

#ifdef USE_PI_CUDA
#include <cuda.h>
#endif // USE_PI_CUDA

#ifdef USE_PI_ROCM
#include <hip/hip_runtime.h>
#endif // USE_PI_ROCM

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static const std::string help =
    "   Help\n"
    "   Example: ./get_device_count_by_type cpu opencl\n"
    "   Supported device types: cpu/gpu/accelerator/default/all\n"
    "   Supported backends: PI_CUDA/PI_ROCM/PI_OPENCL/PI_LEVEL_ZERO \n"
    "   Output format: <number_of_devices>:<additional_Information>";

// Return the string with all characters translated to lower case.
std::string lowerString(const std::string &str) {
  std::string result(str);
  std::transform(result.begin(), result.end(), result.begin(), ::tolower);
  return result;
}

static const char *deviceTypeToString(cl_device_type deviceType) {
  switch (deviceType) {
  case CL_DEVICE_TYPE_CPU:
    return "cpu";
  case CL_DEVICE_TYPE_GPU:
    return "gpu";
  case CL_DEVICE_TYPE_ACCELERATOR:
    return "accelerator";
  case CL_DEVICE_TYPE_CUSTOM:
    return "custom";
  case CL_DEVICE_TYPE_DEFAULT:
    return "default";
  case CL_DEVICE_TYPE_ALL:
    return "all";
  }

  return "unknown";
}

static bool queryOpenCL(cl_device_type deviceType, cl_uint &deviceCount,
                        std::string &msg) {
  deviceCount = 0u;

  cl_uint platformCount = 0;
  cl_int iRet = clGetPlatformIDs(0, nullptr, &platformCount);
  if (iRet != CL_SUCCESS) {
    if (iRet == CL_PLATFORM_NOT_FOUND_KHR) {
      msg = "OpenCL error runtime not found";
      return true;
    }
    std::stringstream stream;
    stream << "ERROR: OpenCL error calling clGetPlatformIDs " << iRet
           << std::endl;
    msg = stream.str();
    return true;
  }

  std::vector<cl_platform_id> platforms(platformCount);
  iRet = clGetPlatformIDs(platformCount, &platforms[0], nullptr);
  if (iRet != CL_SUCCESS) {
    std::stringstream stream;
    stream << "ERROR: OpenCL error calling clGetPlatformIDs " << iRet
           << std::endl;
    msg = stream.str();
    return true;
  }

  for (cl_uint i = 0; i < platformCount; i++) {
    const size_t MAX_PLATFORM_VENDOR = 100u;
    char info[MAX_PLATFORM_VENDOR];
    // get platform attribute value
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, MAX_PLATFORM_VENDOR,
                      info, NULL);
    const auto IsNVIDIAOpenCL = strstr(info, "NVIDIA") != NULL;
    if (IsNVIDIAOpenCL) {
      // Ignore NVIDIA OpenCL platform for testing
      continue;
    }

    cl_uint deviceCountPart = 0;
    iRet =
        clGetDeviceIDs(platforms[i], deviceType, 0, nullptr, &deviceCountPart);
    if (iRet == CL_SUCCESS || iRet == CL_DEVICE_NOT_FOUND) {
      deviceCount += deviceCountPart;
    }
  }

  msg = "opencl ";
  msg += deviceTypeToString(deviceType);
  return true;
}

static bool queryLevelZero(cl_device_type deviceType, cl_uint &deviceCount,
                           std::string &msg) {
  deviceCount = 0u;
  ze_result_t zeResult = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  if (zeResult != ZE_RESULT_SUCCESS) {
    msg = "ERROR: Level Zero initialization error";
    return true;
  }

  uint32_t zeDriverCount = 0;
  zeResult = zeDriverGet(&zeDriverCount, nullptr);
  if (zeResult != ZE_RESULT_SUCCESS) {
    msg = "ERROR: Level Zero error querying driver count";
    return true;
  }

  if (zeDriverCount == 0) {
    msg = "ERROR: Level Zero no driver found";
    return true;
  }

  ze_driver_handle_t zeDriver;
  zeResult = zeDriverGet(&zeDriverCount, &zeDriver);
  if (zeResult != ZE_RESULT_SUCCESS) {
    msg = "ERROR: Level Zero error querying driver";
    return true;
  }

  switch (deviceType) {
  case CL_DEVICE_TYPE_DEFAULT: // Fall through.
  case CL_DEVICE_TYPE_ALL:     // Fall through.
  case CL_DEVICE_TYPE_GPU: {
    uint32_t zeDeviceCount = 0;
    zeResult = zeDeviceGet(zeDriver, &zeDeviceCount, nullptr);
    if (zeResult != ZE_RESULT_SUCCESS) {
      msg = "ERROR: Level Zero error querying device count";
      return true;
    }
    deviceCount = static_cast<cl_uint>(zeDeviceCount);
    msg = "level zero ";
    msg += deviceTypeToString(deviceType);
    return true;
  } break;
  default:
    msg = "WARNING: Level Zero unsupported device type ";
    msg += deviceTypeToString(deviceType);
    return true;
  }
}

static bool queryCUDA(cl_device_type deviceType, cl_uint &deviceCount,
                      std::string &msg) {
  deviceCount = 0u;
#ifdef USE_PI_CUDA
  const unsigned int defaultFlag = 0;
  CUresult err = cuInit(defaultFlag);
  if (err != CUDA_SUCCESS) {
    msg = "ERROR: CUDA initialization error";
    return false;
  }

  const int minRuntimeVersion = 10010;
  int runtimeVersion = 0;
  err = cuDriverGetVersion(&runtimeVersion);
  if (err != CUDA_SUCCESS) {
    msg = "ERROR: CUDA error querying driver version";
    return false;
  }

  if (runtimeVersion < minRuntimeVersion) {
    std::stringstream stream;
    stream << "ERROR: CUDA version must be at least " << minRuntimeVersion
           << " but is only " << runtimeVersion;
    msg = stream.str();
    return false;
  }

  switch (deviceType) {
  case CL_DEVICE_TYPE_DEFAULT: // Fall through.
  case CL_DEVICE_TYPE_ALL:     // Fall through.
  case CL_DEVICE_TYPE_GPU: {
    int count = 0;
    CUresult err = cuDeviceGetCount(&count);
    if (err != CUDA_SUCCESS || count < 0) {
      msg = "ERROR: CUDA error querying device count";
      return false;
    }
    if (count < 1) {
      msg = "ERROR: CUDA no device found";
      return false;
    }
    deviceCount = static_cast<cl_uint>(count);
    msg = "cuda ";
    msg += deviceTypeToString(deviceType);
    return true;
  } break;
  default:
    msg = "WARNING: CUDA unsupported device type ";
    msg += deviceTypeToString(deviceType);
    return true;
  }
#else
  (void)deviceType;
  msg = "ERROR: CUDA not supported";
  deviceCount = 0u;

  return false;
#endif
}

static bool queryROCm(cl_device_type deviceType, cl_uint &deviceCount,
                      std::string &msg) {
  deviceCount = 0u;
#ifdef USE_PI_ROCM
  switch (deviceType) {
  case CL_DEVICE_TYPE_DEFAULT: // Fall through.
  case CL_DEVICE_TYPE_ALL:     // Fall through.
  case CL_DEVICE_TYPE_GPU: {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count < 0) {
      msg = "ERROR: ROCm error querying device count";
      return false;
    }
    if (count < 1) {
      msg = "ERROR: ROCm no device found";
      return false;
    }
    deviceCount = static_cast<cl_uint>(count);
#if defined(__HIP_PLATFORM_AMD__)
    msg = "rocm-amd ";
#elif defined(__HIP_PLATFORM_NVIDIA__)
    msg = "rocm-nvidia ";
#else
#error("Must define one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
    msg += deviceTypeToString(deviceType);
    return true;
  } break;
  default:
    msg = "WARNING: ROCm unsupported device type ";
    msg += deviceTypeToString(deviceType);
    return true;
  }
#else
  (void)deviceType;
  msg = "ERROR: ROCm not supported";
  deviceCount = 0u;

  return false;
#endif
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "0:ERROR: Please set a device type and backend to find"
              << std::endl
              << help << std::endl;
    return EXIT_FAILURE;
  }

  // Normalize all arguments to lower case.
  std::string type{lowerString(argv[1])};
  std::string backend{lowerString(argv[2])};

  cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
  if (type == "cpu") {
    deviceType = CL_DEVICE_TYPE_CPU;
  } else if (type == "gpu") {
    deviceType = CL_DEVICE_TYPE_GPU;
  } else if (type == "accelerator") {
    deviceType = CL_DEVICE_TYPE_ACCELERATOR;
  } else if (type == "default") {
    deviceType = CL_DEVICE_TYPE_DEFAULT;
  } else if (type == "all") {
    deviceType = CL_DEVICE_TYPE_ALL;
  } else {
    std::cout << "0:ERROR: Incorrect device type " << type << "\n"
              << help << std::endl;
    return EXIT_FAILURE;
  }

  std::string msg;
  cl_uint deviceCount = 0;

  bool querySuccess = false;

  if (backend == "opencl" || backend == "pi_opencl") {
    querySuccess = queryOpenCL(deviceType, deviceCount, msg);
  } else if (backend == "level_zero" || backend == "pi_level_zero") {
    querySuccess = queryLevelZero(deviceType, deviceCount, msg);
  } else if (backend == "cuda" || backend == "pi_cuda") {
    querySuccess = queryCUDA(deviceType, deviceCount, msg);
  } else if (backend == "rocm" || backend == "pi_rocm") {
    querySuccess = queryROCm(deviceType, deviceCount, msg);
  } else {
    msg = "ERROR: Unknown backend " + backend + "\n" + help + "\n";
  }

  std::cout << deviceCount << ":" << msg << std::endl;

  if (!querySuccess) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
