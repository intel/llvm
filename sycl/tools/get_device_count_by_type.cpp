//==-- get_device_count_by_type.cpp - Get device count by type -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/cl.h>
#include <CL/cl_ext.h>

#ifdef USE_PI_CUDA
#include <cuda_driver.h>
#endif  // USE_PI_CUDA

#include <iostream>
#include <string>
#include <vector>

static const std::string help =
"   Help\n"
"   Example: ./get_device_count_by_type cpu opencl\n"
"   Support types: cpu/gpu/accelerator/default/all\n"
"   Support backends: cuda/opencl \n"
"   Output format: <number_of_devices>:<additional_Information>";

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout  
            << "0:Please set a device type and backend to find" << std::endl
            << help << std::endl;
        return 0;
    }

    std::string type = argv[1];
    std::string backend{argv[2]};

    cl_uint deviceCount = 0;

#ifdef USE_PI_CUDA
    if (backend == "CUDA") {
      std::string msg{""};

      int runtime_version = 0;

      cudaError_t err = cuDriverGetVersion(&runtime_version);
      if (runtime_version < 9020 || err != CUDA_SUCCESS) {
        std::cout << deviceCount << " :Unsupported CUDA Runtime " << std::endl;
      }

      if (type == "gpu") {
        deviceCount = 1;
        msg = "cuda";
      } else {
        msg = "Unsupported device type for CUDA backend";
        msg += " type: ";
        msg += type;
      }
      std::cout << deviceCount << " : " << msg << std::endl;
      return 0;
    }
#endif  // USE_PI_CUDA

    cl_device_type device_type;
    if (type == "cpu") {
        device_type = CL_DEVICE_TYPE_CPU;
    } else if (type == "gpu") {
        device_type = CL_DEVICE_TYPE_GPU;
    } else if (type == "accelerator") {
        device_type = CL_DEVICE_TYPE_ACCELERATOR;
    } else if (type == "default") {
        device_type = CL_DEVICE_TYPE_DEFAULT;
    } else if (type == "all") {
        device_type = CL_DEVICE_TYPE_ALL;
    } else  {
        std::cout << "0:Incorrect device type." << std::endl
            << help << std::endl;
        return 0;
    }

    cl_int iRet = CL_SUCCESS;
    cl_uint platformCount = 0;

    iRet = clGetPlatformIDs(0, nullptr, &platformCount);
    if (iRet != CL_SUCCESS) {
        if (iRet == CL_PLATFORM_NOT_FOUND_KHR) {
            std::cout << "0:OpenCL runtime not found " << std::endl;
        } else {
            std::cout << "0:A problem at calling function clGetPlatformIDs count "
                << iRet << std::endl;
        }
        return 0;
    }

    std::vector<cl_platform_id> platforms(platformCount);

    iRet = clGetPlatformIDs(platformCount, &platforms[0], nullptr);
    if (iRet != CL_SUCCESS) {
        std::cout << "0:A problem at when calling function clGetPlatformIDs ids " << iRet << std::endl;
        return 0;
    }

    for (cl_uint i = 0; i < platformCount; i++) {
        cl_uint deviceCountPart = 0;
        iRet = clGetDeviceIDs(platforms[i], device_type, 0, nullptr, &deviceCountPart);
        if (iRet == CL_SUCCESS) {
            deviceCount += deviceCountPart;
        }
    }

    std::cout << deviceCount << ":" << backend << std::endl;

    return 0;
}
