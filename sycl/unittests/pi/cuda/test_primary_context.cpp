//==---------- pi_primary_context.cpp - PI unit tests ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cuda.h>

#include "TestGetPlatforms.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>
#include <pi_cuda.hpp>

#include <iostream>

using namespace cl::sycl;

struct CudaPrimaryContextTests : public ::testing::TestWithParam<platform> {

protected:
  device deviceA_;
  device deviceB_;

  void SetUp() override {
    std::vector<device> CudaDevices = GetParam().get_devices();

    deviceA_ = CudaDevices[0];
    deviceB_ = CudaDevices.size() > 1 ? CudaDevices[1] : deviceA_;
  }

  void TearDown() override {}
};

TEST_P(CudaPrimaryContextTests, piSingleContext) {
  std::cout << "create single context" << std::endl;
  context Context(deviceA_, async_handler{},
                  {sycl::property::context::cuda::use_primary_context{}});

  CUdevice CudaDevice = deviceA_.get_native<backend::cuda>();
  CUcontext CudaContext = Context.get_native<backend::cuda>();

  CUcontext PrimaryCudaContext;
  cuDevicePrimaryCtxRetain(&PrimaryCudaContext, CudaDevice);

  ASSERT_EQ(CudaContext, PrimaryCudaContext);

  cuDevicePrimaryCtxRelease(CudaDevice);
}

TEST_P(CudaPrimaryContextTests, piMultiContextSingleDevice) {
  std::cout << "create multiple contexts for one device" << std::endl;
  context ContextA(deviceA_, async_handler{},
                   {sycl::property::context::cuda::use_primary_context{}});
  context ContextB(deviceA_, async_handler{},
                   {sycl::property::context::cuda::use_primary_context{}});

  CUcontext CudaContextA = ContextA.get_native<backend::cuda>();
  CUcontext CudaContextB = ContextB.get_native<backend::cuda>();

  ASSERT_EQ(CudaContextA, CudaContextB);
}

TEST_P(CudaPrimaryContextTests, piMultiContextMultiDevice) {
  if (deviceA_ == deviceB_)
    return;

  CUdevice CudaDeviceA = deviceA_.get_native<backend::cuda>();
  CUdevice CudaDeviceB = deviceB_.get_native<backend::cuda>();

  ASSERT_NE(CudaDeviceA, CudaDeviceB);

  std::cout << "create multiple contexts for multiple devices" << std::endl;
  context ContextA(deviceA_, async_handler{},
                   {sycl::property::context::cuda::use_primary_context{}});
  context ContextB(deviceB_, async_handler{},
                   {sycl::property::context::cuda::use_primary_context{}});

  CUcontext CudaContextA = ContextA.get_native<backend::cuda>();
  CUcontext CudaContextB = ContextB.get_native<backend::cuda>();

  ASSERT_NE(CudaContextA, CudaContextB);
}

INSTANTIATE_TEST_SUITE_P(
    OnCudaPlatform, CudaPrimaryContextTests,
    ::testing::ValuesIn(pi::getPlatformsWithName("CUDA BACKEND")));
