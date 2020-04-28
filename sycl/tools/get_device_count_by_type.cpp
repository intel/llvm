//==-- get_device_count_by_type.cpp - Get device count by type -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#ifdef USE_PI_CUDA
#include <cuda_driver.h>
#endif  // USE_PI_CUDA

#include <iostream>
#include <string>
#include <vector>

using namespace cl::sycl;

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

    info::device_type device_type;
    if (type == "cpu") {
      device_type = info::device_type::cpu;
    } else if (type == "gpu") {
      device_type = info::device_type::gpu;
    } else if (type == "accelerator") {
      device_type = info::device_type::accelerator;
    } else if (type == "default") {
      device_type = info::device_type::automatic;
    } else if (type == "all") {
      device_type = info::device_type::all;
    } else  {
        std::cout << "0:Incorrect device type." << std::endl
            << help << std::endl;
        return 0;
    }

    std::vector<platform> platforms(platform::get_platforms());
    for (cl_uint i = 0; i < platforms.size(); i++) {
      std::vector<device> result = platforms[i].get_devices(device_type);
      deviceCount += result.size();
    }

    std::cout << deviceCount << ":" << backend << std::endl;
    return 0;
}
