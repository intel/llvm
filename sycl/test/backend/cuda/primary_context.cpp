// REQUIRES: cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -I%opencl_include_dir -I%cuda_toolkit_include -o %t.out -lcuda -lsycl
// RUN: env SYCL_DEVICE_TYPE=GPU %t.out
// NOTE: OpenCL is required for the runtime, even when using the CUDA BE.

//==---------- primary_context.cpp - SYCL cuda primary context test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <CL/sycl/detail/pi_cuda.hpp>
#include <cuda.h>
#include <iostream>

using namespace cl::sycl;

void check(bool condition, const char *conditionString, const char *filename,
           const long line) noexcept {
  if (!condition) {
    std::cerr << "CHECK failed in " << filename << "#" << line << " "
              << conditionString << "\n";
    std::abort();
  }
}

#define CHECK(CONDITION) check(CONDITION, #CONDITION, __FILE__, __LINE__)

bool isCudaDevice(const device &dev) {
  const platform platform = dev.get_info<info::device::platform>();
  const std::string platformVersion =
      platform.get_info<info::platform::version>();
  // If using PI_CUDA, don't accept a non-CUDA device
  return platformVersion.find("CUDA") != std::string::npos;
}

class cuda_device_selector : public device_selector {
public:
  int operator()(const device &dev) const {
    return isCudaDevice(dev) ? 1 : -1;
  }
};

class other_cuda_device_selector : public device_selector {
public:
  other_cuda_device_selector(const device &dev) : excludeDevice{dev} {}

  int operator()(const device &dev) const {
    if (!isCudaDevice(dev)) {
      return -1;
    }
    if (dev.get() == excludeDevice.get()) {
      // Return only this device if it is the only available
      return 0;
    }
    return 1;
  }

private:
  const device &excludeDevice;
};

int main() {
  try {
    context c;
  } catch (device_error &e) {
    std::cout << "Failed to create device for context" << std::endl;
  }

  device DeviceA = cuda_device_selector().select_device();
  device DeviceB = other_cuda_device_selector(DeviceA).select_device();

  CHECK(isCudaDevice(DeviceA));

  {
    std::cout << "create single context" << std::endl;
    context Context(DeviceA, async_handler{}, /*UsePrimaryContext=*/true);

    CUdevice CudaDevice = reinterpret_cast<pi_device>(DeviceA.get())->get();
    CUcontext CudaContext = reinterpret_cast<pi_context>(Context.get())->get();

    CUcontext PrimaryCudaContext;
    cuDevicePrimaryCtxRetain(&PrimaryCudaContext, CudaDevice);

    CHECK(CudaContext == PrimaryCudaContext);

    cuDevicePrimaryCtxRelease(CudaDevice);
  }
  {
    std::cout << "create multiple contexts for one device" << std::endl;
    context ContextA(DeviceA, async_handler{}, /*UsePrimaryContext=*/true);
    context ContextB(DeviceA, async_handler{}, /*UsePrimaryContext=*/true);

    CUcontext CudaContextA =
        reinterpret_cast<pi_context>(ContextA.get())->get();
    CUcontext CudaContextB =
        reinterpret_cast<pi_context>(ContextB.get())->get();

    CHECK(CudaContextA == CudaContextB);
  }
  if (isCudaDevice(DeviceB) && DeviceA.get() != DeviceB.get()) {
    std::cout << "create multiple contexts for multiple devices" << std::endl;
    context ContextA(DeviceA, async_handler{}, /*UsePrimaryContext=*/true);
    context ContextB(DeviceB, async_handler{}, /*UsePrimaryContext=*/true);

    CUcontext CudaContextA =
        reinterpret_cast<pi_context>(ContextA.get())->get();
    CUcontext CudaContextB =
        reinterpret_cast<pi_context>(ContextB.get())->get();

    CHECK(CudaContextA != CudaContextB);
  }
}
