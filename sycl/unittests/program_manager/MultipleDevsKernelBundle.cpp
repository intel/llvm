//==----------------------- MultipleDevsKernelBundle.cpp -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Kernel bundle for multiple devices unit test

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include "detail/context_impl.hpp"
#include "detail/kernel_bundle_impl.hpp"
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>

using namespace sycl;

class MultipleDevsKernelBundleTestKernel;

MOCK_INTEGRATION_HEADER(MultipleDevsKernelBundleTestKernel)

static constexpr uint32_t NumDevices = 3;

static sycl::unittest::MockDeviceImage Img =
    sycl::unittest::generateDefaultImage(
        {"MultipleDevsKernelBundleTestKernel"});
static sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};

inline ur_result_t redefinedurKernelGetInfo(void *pParams) {
  auto params = *static_cast<ur_kernel_get_info_params_t *>(pParams);
  constexpr char MockKernel[] = "MultipleDevsKernelBundleTestKernel";
  if (*params.ppropName == UR_KERNEL_INFO_FUNCTION_NAME) {
    if (*params.ppPropValue) {
      assert(*params.ppropSize == sizeof(MockKernel));
      std::memcpy(*params.ppPropValue, MockKernel, sizeof(MockKernel));
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(MockKernel);
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedDeviceGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices) {
    **params.ppNumDevices = static_cast<uint32_t>(NumDevices);
    return UR_RESULT_SUCCESS;
  }

  if (*params.pNumEntries == NumDevices && *params.pphDevices) {
    for (std::uintptr_t i = 0; i < NumDevices; ++i)
      (*params.pphDevices)[i] = reinterpret_cast<ur_device_handle_t>(i + 1);
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_MULTI_DEVICE_COMPILE_SUPPORT_EXP) {
    auto *Result = reinterpret_cast<ur_bool_t *>(*params.ppPropValue);
    *Result = true;
  }
  return UR_RESULT_SUCCESS;
}

static int ProgramBuildExpCounter = 0;
static ur_result_t redefinedurProgramBuildExp(void *) {
  ++ProgramBuildExpCounter;
  return UR_RESULT_SUCCESS;
}

static int ProgramCreateWithILCounter = 0;
static ur_result_t redefinedProgramCreateWithIL(void *) {
  ++ProgramCreateWithILCounter;
  return UR_RESULT_SUCCESS;
}

class MultipleDevsKernelBundleTest
    : public testing::TestWithParam<std::array<size_t, NumDevices>> {
public:
  MultipleDevsKernelBundleTest() : Mock{}, Plt{sycl::platform()} {}

protected:
  void SetUp() override {
    mock::getCallbacks().set_after_callback("urDeviceGet", &redefinedDeviceGet);
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &redefinedDeviceGetInfo);
    mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                            &redefinedurKernelGetInfo);
    mock::getCallbacks().set_after_callback("urProgramBuildExp",
                                            &redefinedurProgramBuildExp);
    mock::getCallbacks().set_after_callback("urProgramCreateWithIL",
                                            &redefinedProgramCreateWithIL);
  }

protected:
  unittest::UrMock<> Mock;
  platform Plt;
};

// Test to check that we can create input kernel bundle and call build twice for
// overlapping set of devices and execute the kernel on each device.
TEST_P(MultipleDevsKernelBundleTest, BuildTwiceWithOverlappingDevices) {
  // Reset counters
  ProgramCreateWithILCounter = 0;
  ProgramBuildExpCounter = 0;

  // Get devices and create a context with at least 3 devices
  std::vector<sycl::device> Devices = Plt.get_devices();
  ASSERT_GE(Devices.size(), 3lu) << "Test requires at least 3 devices";

  auto Dev1 = Devices[0], Dev2 = Devices[1], Dev3 = Devices[2];

  // Create a context with the selected devices
  sycl::context Context({Dev1, Dev2, Dev3});

  // Create queues for each device
  sycl::queue Queue1(Context, Dev1);
  sycl::queue Queue2(Context, Dev2);
  sycl::queue Queue3(Context, Dev3);

  // Get kernel ID
  auto KernelID = sycl::get_kernel_id<MultipleDevsKernelBundleTestKernel>();

  // Create an input kernel bundle
  auto KernelBundleInput =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Context, {KernelID});

  // Build kernel bundles for overlapping sets of devices
  auto KernelBundleExe1 = sycl::build(KernelBundleInput, {Dev1, Dev2});
  auto KernelBundleExe2 = sycl::build(KernelBundleInput, {Dev2, Dev3});

  // Get kernel objects from the built bundles
  auto KernelObj1 = KernelBundleExe1.get_kernel(KernelID);
  auto KernelObj2 = KernelBundleExe2.get_kernel(KernelID);

  // Submit tasks to the queues using the kernel bundles
  Queue1.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(KernelBundleExe1);
    cgh.single_task<MultipleDevsKernelBundleTestKernel>([]() {});
  });

  Queue2.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(KernelBundleExe1);
    cgh.single_task(KernelObj1);
  });

  Queue2.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(KernelBundleExe2);
    cgh.single_task(KernelObj2);
  });

  Queue3.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(KernelBundleExe2);
    cgh.single_task(KernelObj2);
  });

  // Verify the number of urProgramCreateWithIL calls
  EXPECT_EQ(ProgramCreateWithILCounter, 2)
      << "Expect 2 urProgramCreateWithIL calls";

  // Verify the number of urProgramBuildExp calls
  EXPECT_EQ(ProgramBuildExpCounter, 2) << "Expect 2 urProgramBuildExp calls";
}

INSTANTIATE_TEST_SUITE_P(
    MultipleDevsKernelBundleTestInstance, MultipleDevsKernelBundleTest,
    testing::Values(std::array<size_t, NumDevices>{0, 1, 2},
                    std::array<size_t, NumDevices>{1, 0, 2},
                    std::array<size_t, NumDevices>{2, 1, 0}));
