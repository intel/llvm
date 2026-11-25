// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %maybe-v1 ./bindless_images_info-test
// RUN: %maybe-v2 ./bindless_images_info-test

#include "ze_api.h"
#include <uur/fixtures.h>

#include <cstring>
#include <utility>
#include <vector>

struct urLevelZeroBindlessImagesInfoTest : uur::urDeviceTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urDeviceTest::SetUp());

    ur_native_handle_t nativeDevice = 0;
    ASSERT_SUCCESS(urDeviceGetNativeHandle(device, &nativeDevice));
    auto zeDevice = reinterpret_cast<ze_device_handle_t>(nativeDevice);

    ur_platform_handle_t platform = nullptr;
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_PLATFORM,
                                   sizeof(platform), &platform, nullptr));
    ur_native_handle_t nativePlatform = 0;
    ASSERT_SUCCESS(urPlatformGetNativeHandle(platform, &nativePlatform));
    auto zeDriver = reinterpret_cast<ze_driver_handle_t>(nativePlatform);

    uint32_t count = 0;
    ASSERT_EQ(zeDriverGetExtensionProperties(zeDriver, &count, nullptr),
              ZE_RESULT_SUCCESS);
    std::vector<ze_driver_extension_properties_t> driverExts(count);
    ASSERT_EQ(
        zeDriverGetExtensionProperties(zeDriver, &count, driverExts.data()),
        ZE_RESULT_SUCCESS);
    for (const auto &ext : driverExts)
      if (std::strcmp(ext.name, ZE_BINDLESS_IMAGE_EXP_NAME) == 0 &&
          ext.version == ZE_BINDLESS_IMAGE_EXP_VERSION_1_0)
        driverHasExt = true;

    imageProps.stype = ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES;
    ASSERT_EQ(zeDeviceGetImageProperties(zeDevice, &imageProps),
              ZE_RESULT_SUCCESS);
  }

  bool getBoolInfo(ur_device_info_t info) {
    ur_bool_t value = false;
    EXPECT_SUCCESS(
        urDeviceGetInfo(device, info, sizeof(value), &value, nullptr));
    return value;
  }

  bool driverHasExt = false;
  ze_device_image_properties_t imageProps = {};
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urLevelZeroBindlessImagesInfoTest);

TEST_P(urLevelZeroBindlessImagesInfoTest, SupportMatchesImageLimits) {
  const std::pair<ur_device_info_t, uint32_t> expectations[] = {
      {UR_DEVICE_INFO_BINDLESS_IMAGES_SUPPORT_EXP, imageProps.maxImageDims1D},
      {UR_DEVICE_INFO_BINDLESS_IMAGES_1D_USM_SUPPORT_EXP,
       imageProps.maxImageDims1D},
      {UR_DEVICE_INFO_BINDLESS_IMAGES_2D_USM_SUPPORT_EXP,
       imageProps.maxImageDims2D},
      {UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_USM_SUPPORT_EXP,
       imageProps.maxImageDims1D},
      {UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_SUPPORT_EXP,
       imageProps.maxImageDims1D},
      {UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_USM_SUPPORT_EXP,
       imageProps.maxImageDims2D},
      {UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_SUPPORT_EXP,
       imageProps.maxImageDims2D},
      {UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_3D_SUPPORT_EXP,
       imageProps.maxImageDims3D},
      {UR_DEVICE_INFO_IMAGE_ARRAY_SUPPORT_EXP, imageProps.maxImageArraySlices},
  };

  for (const auto &[info, limit] : expectations) {
    SCOPED_TRACE(info);
    EXPECT_EQ(getBoolInfo(info), driverHasExt && limit > 0);
  }
}
