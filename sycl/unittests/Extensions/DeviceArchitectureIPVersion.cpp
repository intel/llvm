//==------------ DeviceArchitectureIPVersion.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>

#include <sycl/ext/oneapi/experimental/device_architecture.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

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
