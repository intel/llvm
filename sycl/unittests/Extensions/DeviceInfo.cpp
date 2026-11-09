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

struct IgcaMockState {
  bool Supported;
  uint32_t Target;
  ur_device_igca_feature_flags_t Flags;
};

IgcaMockState IgcaState;

ur_result_t afterIgcaDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  const ur_device_info_t Name = *params.ppropName;
  if (Name != UR_DEVICE_INFO_IGCA_TARGET &&
      Name != UR_DEVICE_INFO_IGCA_FEATURE_SET)
    return UR_RESULT_SUCCESS;

  if (!IgcaState.Supported)
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;

  if (*params.ppPropValue) {
    if (Name == UR_DEVICE_INFO_IGCA_TARGET)
      *reinterpret_cast<uint32_t *>(*params.ppPropValue) = IgcaState.Target;
    else
      *reinterpret_cast<ur_device_igca_feature_flags_t *>(*params.ppPropValue) =
          IgcaState.Flags;
  }
  return UR_RESULT_SUCCESS;
}

// aspect::ext_intel_igca is cached per device_impl, so the mocked values must
// be in place before the device is constructed. Hence the device is obtained in
// the test body rather than in the fixture.
class IgcaDeviceInfoTests : public ::testing::Test {
protected:
  void SetUp() override {
    IgcaState = {/*Supported=*/true, /*Target=*/20,
                 UR_DEVICE_IGCA_FEATURE_FLAG_COMPUTE};
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &afterIgcaDeviceGetInfo);
  }

  sycl::device getDevice() { return sycl::platform().get_devices()[0]; }

  sycl::unittest::UrMock<> Mock;
};

// Keeping the render and compute cases separate catches a swapped mapping from
// the UR feature-set flags onto the igca_feature_set_* constants.
TEST_F(IgcaDeviceInfoTests, CheckIgcaComputeFeatureSet) {
  IgcaState.Flags = UR_DEVICE_IGCA_FEATURE_FLAG_COMPUTE;
  sycl::device Dev = getDevice();

  ASSERT_TRUE(Dev.has(aspect::ext_intel_igca));
  auto Igca = Dev.get_info<ext::intel::info::device::igca>();
  EXPECT_EQ(Igca.target, 20);
  EXPECT_EQ(Igca.feature_sets, ext::intel::info::igca_feature_set_compute);
}

TEST_F(IgcaDeviceInfoTests, CheckIgcaRenderFeatureSet) {
  IgcaState.Flags = UR_DEVICE_IGCA_FEATURE_FLAG_RENDER;
  sycl::device Dev = getDevice();

  ASSERT_TRUE(Dev.has(aspect::ext_intel_igca));
  auto Igca = Dev.get_info<ext::intel::info::device::igca>();
  EXPECT_EQ(Igca.feature_sets, ext::intel::info::igca_feature_set_render);
}

TEST_F(IgcaDeviceInfoTests, CheckIgcaBothFeatureSets) {
  IgcaState.Flags =
      UR_DEVICE_IGCA_FEATURE_FLAG_RENDER | UR_DEVICE_IGCA_FEATURE_FLAG_COMPUTE;
  IgcaState.Target = 30;
  sycl::device Dev = getDevice();

  ASSERT_TRUE(Dev.has(aspect::ext_intel_igca));
  auto Igca = Dev.get_info<ext::intel::info::device::igca>();
  EXPECT_EQ(Igca.target, 30);
  EXPECT_EQ(Igca.feature_sets, ext::intel::info::igca_feature_set_render |
                                   ext::intel::info::igca_feature_set_compute);
}

TEST_F(IgcaDeviceInfoTests, CheckIgcaUnsupported) {
  IgcaState.Supported = false;
  sycl::device Dev = getDevice();

  ASSERT_FALSE(Dev.has(aspect::ext_intel_igca));
  try {
    Dev.get_info<ext::intel::info::device::igca>();
    FAIL() << "Expected an exception for a device without the IGCA aspect";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::feature_not_supported));
  }
}

} // namespace
