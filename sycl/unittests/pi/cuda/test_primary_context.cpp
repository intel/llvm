//==---------- pi_primary_context.cpp - PI unit tests ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cuda.h>

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>
#include <pi_cuda.hpp>

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
  int operator()(const device &dev) const { return isCudaDevice(dev) ? 1 : -1; }
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

using namespace cl::sycl;

struct DISABLED_CudaPrimaryContextTests : public ::testing::Test {

protected:
  std::vector<detail::plugin> Plugins;

  pi_platform platform_;
  device deviceA_;
  device deviceB_;
  context context_;

  void SetUp() override {

    try {
      context context_;
    } catch (device_error &e) {
      std::cout << "Failed to create device for context" << std::endl;
    }

    deviceA_ = cuda_device_selector().select_device();
    deviceB_ = other_cuda_device_selector(deviceA_).select_device();

    ASSERT_TRUE(isCudaDevice(deviceA_));
  }

  void TearDown() override {}
};

TEST_F(DISABLED_CudaPrimaryContextTests, piSingleContext) {
  std::cout << "create single context" << std::endl;
  context Context(deviceA_, async_handler{}, /*UsePrimaryContext=*/true);

  CUdevice CudaDevice = reinterpret_cast<pi_device>(deviceA_.get())->get();
  CUcontext CudaContext = reinterpret_cast<pi_context>(Context.get())->get();

  CUcontext PrimaryCudaContext;
  cuDevicePrimaryCtxRetain(&PrimaryCudaContext, CudaDevice);

  ASSERT_EQ(CudaContext, PrimaryCudaContext);

  cuDevicePrimaryCtxRelease(CudaDevice);
}

TEST_F(DISABLED_CudaPrimaryContextTests, piMultiContextSingleDevice) {
  std::cout << "create multiple contexts for one device" << std::endl;
  context ContextA(deviceA_, async_handler{}, /*UsePrimaryContext=*/true);
  context ContextB(deviceA_, async_handler{}, /*UsePrimaryContext=*/true);

  CUcontext CudaContextA = reinterpret_cast<pi_context>(ContextA.get())->get();
  CUcontext CudaContextB = reinterpret_cast<pi_context>(ContextB.get())->get();

  ASSERT_EQ(CudaContextA, CudaContextB);
}

TEST_F(DISABLED_CudaPrimaryContextTests, piMultiContextMultiDevice) {
  if (isCudaDevice(deviceB_) && deviceA_.get() != deviceB_.get()) {
    std::cout << "create multiple contexts for multiple devices" << std::endl;
    context ContextA(deviceA_, async_handler{}, /*UsePrimaryContext=*/true);
    context ContextB(deviceB_, async_handler{}, /*UsePrimaryContext=*/true);

    CUcontext CudaContextA =
        reinterpret_cast<pi_context>(ContextA.get())->get();
    CUcontext CudaContextB =
        reinterpret_cast<pi_context>(ContextB.get())->get();

    ASSERT_NE(CudaContextA, CudaContextB);
  }
}
