//==--------------- DeviceArchitectureOneArchSelected.cpp ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>

// define one of __SYCL_TARGET_ macro, e.g., the one for SKL
#define __SYCL_TARGET_INTEL_GPU_SKL__ 1

#include <sycl/ext/oneapi/experimental/device_architecture.hpp>

using namespace sycl;
using namespace sycl::detail;
using namespace sycl::ext::oneapi::experimental;

TEST(DeviceArchitectureTest, DeviceArchitecture_If) {
  bool res = false;
  if_architecture_is<architecture::intel_gpu_skl>([&]() { res = true; });
  ASSERT_TRUE(res);
}

TEST(DeviceArchitectureTest, DeviceArchitecture_If_Negative) {
  bool res = false;
  if_architecture_is<architecture::intel_gpu_pvc>([&]() { res = true; });
  ASSERT_FALSE(res);
}

TEST(DeviceArchitectureTest, DeviceArchitecture_Else_If) {
  bool res = false;
  if_architecture_is<architecture::intel_gpu_dg1>([]() {
  }).else_if_architecture_is<architecture::intel_gpu_skl>([&]() {
    res = true;
  });
  ASSERT_TRUE(res);
}

TEST(DeviceArchitectureTest, DeviceArchitecture_Otherwise) {
  bool res = false;
  if_architecture_is<architecture::intel_gpu_dg1>([]() {
  }).else_if_architecture_is<architecture::intel_gpu_pvc>([&]() {
    }).otherwise([&]() { res = true; });
  ASSERT_TRUE(res);
}

namespace {
struct IPVersionCase {
  uint32_t IPVersion;
  architecture ExpectedArchitecture;
};

class DeviceArchitectureIPVersionTest
    : public ::testing::TestWithParam<IPVersionCase> {
protected:
  static ur_result_t afterDeviceGetInfo(void *Params) {
    auto &Info = *static_cast<ur_device_get_info_params_t *>(Params);
    if (*Info.ppropName == UR_DEVICE_INFO_IP_VERSION) {
      if (*Info.ppPropValue)
        *static_cast<uint32_t *>(*Info.ppPropValue) = GetParam().IPVersion;
      if (*Info.ppPropSizeRet)
        **Info.ppPropSizeRet = sizeof(uint32_t);
    }
    return UR_RESULT_SUCCESS;
  }
};
} // namespace

TEST_P(DeviceArchitectureIPVersionTest, IPVersionMapsToArchitecture) {
  unittest::UrMock<backend::ext_oneapi_level_zero> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &afterDeviceGetInfo);

  auto Devices = platform().get_devices();
  ASSERT_FALSE(Devices.empty());
  EXPECT_EQ(
      Devices[0]
          .get_info<ext::oneapi::experimental::info::device::architecture>(),
      GetParam().ExpectedArchitecture);
}

INSTANTIATE_TEST_SUITE_P(Common, DeviceArchitectureIPVersionTest,
                         ::testing::Values(IPVersionCase{
                             0x03000000, architecture::intel_gpu_tgllp}));
