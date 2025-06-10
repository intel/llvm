//==------------------ DeviceInfo.cpp - device info query test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

namespace {

ur_result_t redefinedDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_CURRENT_CLOCK_THROTTLE_REASONS) {
    if (*params.ppPropValue) {
      ur_device_throttle_reasons_flags_t *ThrottleReasons =
          reinterpret_cast<ur_device_throttle_reasons_flags_t *>(
              *params.ppPropValue);
      *ThrottleReasons = UR_DEVICE_THROTTLE_REASONS_FLAG_POWER_CAP |
                         UR_DEVICE_THROTTLE_REASONS_FLAG_CURRENT_LIMIT |
                         UR_DEVICE_THROTTLE_REASONS_FLAG_THERMAL_LIMIT;
    }
  } else if (*params.ppropName == UR_DEVICE_INFO_FAN_SPEED) {
    if (*params.ppPropValue) {
      int32_t *FanSpeed = reinterpret_cast<int32_t *>(*params.ppPropValue);
      *FanSpeed = 75;
    }
  } else if (*params.ppropName == UR_DEVICE_INFO_MIN_POWER_LIMIT) {
    if (*params.ppPropValue) {
      int32_t *MinPowerLimit = reinterpret_cast<int32_t *>(*params.ppPropValue);
      *MinPowerLimit = 50;
    }
  } else if (*params.ppropName == UR_DEVICE_INFO_MAX_POWER_LIMIT) {
    if (*params.ppPropValue) {
      int32_t *MaxPowerLimit = reinterpret_cast<int32_t *>(*params.ppPropValue);
      *MaxPowerLimit = 150;
    }
  }
  return UR_RESULT_SUCCESS;
}

class DeviceInfoTests : public ::testing::Test {
public:
  DeviceInfoTests() : Mock{}, Dev{sycl::platform().get_devices()[0]} {}

protected:
  void SetUp() override {

    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &redefinedDeviceGetInfo);
  }

  sycl::unittest::UrMock<> Mock;
  sycl::device Dev;
};

TEST_F(DeviceInfoTests, CheckCurrentClockThrottleReasons) {
  auto ThrottleReasons =
      Dev.get_info<ext::intel::info::device::current_clock_throttle_reasons>();
  constexpr size_t expectedThrottleReasonsVecSize = 3;
  EXPECT_EQ(ThrottleReasons.size(), expectedThrottleReasonsVecSize);

  auto HasThrottleReason =
      [&](const std::vector<ext::intel::throttle_reason> &deviceThrottleReasons,
          ext::intel::throttle_reason reasonToFind) -> bool {
    return std::find(deviceThrottleReasons.begin(), deviceThrottleReasons.end(),
                     reasonToFind) != deviceThrottleReasons.end();
  };

  EXPECT_TRUE(HasThrottleReason(ThrottleReasons,
                                ext::intel::throttle_reason::power_cap));
  EXPECT_TRUE(HasThrottleReason(ThrottleReasons,
                                ext::intel::throttle_reason::current_limit));
  EXPECT_TRUE(HasThrottleReason(ThrottleReasons,
                                ext::intel::throttle_reason::thermal_limit));
  EXPECT_FALSE(
      HasThrottleReason(ThrottleReasons, ext::intel::throttle_reason::other));
}

TEST_F(DeviceInfoTests, CheckFanSpeed) {
  auto FanSpeed = Dev.get_info<ext::intel::info::device::fan_speed>();
  EXPECT_EQ(FanSpeed, 75);
}

TEST_F(DeviceInfoTests, CheckPowerLimits) {
  auto MinPowerLimit =
      Dev.get_info<ext::intel::info::device::min_power_limit>();
  EXPECT_EQ(MinPowerLimit, 50);

  auto MaxPowerLimit =
      Dev.get_info<ext::intel::info::device::max_power_limit>();
  EXPECT_EQ(MaxPowerLimit, 150);
}

} // namespace
