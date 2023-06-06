//==---------- test_primary_context.cpp - PI unit tests --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <hip/hip_runtime.h>

#include "TestGetPlatforms.hpp"
#include <pi_hip.hpp>
#include <sycl/ext/oneapi/backend/hip.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

struct HipPrimaryContextTests : public ::testing::TestWithParam<platform> {

protected:
  device deviceA_;
  device deviceB_;

  void SetUp() override {
    std::vector<device> HipDevices = GetParam().get_devices();

    deviceA_ = HipDevices[0];
    deviceB_ = HipDevices.size() > 1 ? HipDevices[1] : deviceA_;
  }

  void TearDown() override {}
};

TEST_P(HipPrimaryContextTests, piSingleContext) {
  std::cout << "create single context" << std::endl;
  context Context(deviceA_, async_handler{});

  hipDevice_t HipDevice = get_native<backend::ext_oneapi_hip>(deviceA_);
  hipCtx_t HipContext = get_native<backend::ext_oneapi_hip>(Context);

  hipCtx_t PrimaryHipContext;
  hipDevicePrimaryCtxRetain(&PrimaryHipContext, HipDevice);

  ASSERT_EQ(HipContext, PrimaryHipContext);

  hipDevicePrimaryCtxRelease(HipDevice);
}

TEST_P(HipPrimaryContextTests, piMultiContextSingleDevice) {
  std::cout << "create multiple contexts for one device" << std::endl;
  context ContextA(deviceA_, async_handler{});
  context ContextB(deviceA_, async_handler{});

  hipCtx_t HipContextA = get_native<backend::ext_oneapi_hip>(ContextA);
  hipCtx_t HipContextB = get_native<backend::ext_oneapi_hip>(ContextB);

  ASSERT_EQ(HipContextA, HipContextB);
}

TEST_P(HipPrimaryContextTests, piMultiContextMultiDevice) {
  if (deviceA_ == deviceB_)
    return;

  hipDevice_t HipDeviceA = get_native<backend::ext_oneapi_hip>(deviceA_);
  hipDevice_t HipDeviceB = get_native<backend::ext_oneapi_hip>(deviceB_);

  ASSERT_NE(HipDeviceA, HipDeviceB);

  std::cout << "create multiple contexts for multiple devices" << std::endl;
  context ContextA(deviceA_, async_handler{});
  context ContextB(deviceB_, async_handler{});

  hipCtx_t HipContextA = get_native<backend::ext_oneapi_hip>(ContextA);
  hipCtx_t HipContextB = get_native<backend::ext_oneapi_hip>(ContextB);

  ASSERT_NE(HipContextA, HipContextB);
}

INSTANTIATE_TEST_SUITE_P(
    OnHipPlatform, HipPrimaryContextTests,
    ::testing::ValuesIn(pi::getPlatformsWithName("HIP BACKEND")));
