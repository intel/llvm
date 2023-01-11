// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: MIT

#include "fixtures.h"
#include <cstring>

struct urPlatformGetInfoTest
    : uur::platform::urPlatformTest,
      ::testing::WithParamInterface<ur_platform_info_t> {

  void SetUp() {
    UUR_RETURN_ON_FATAL_FAILURE(uur::platform::urPlatformTest::SetUp());
  }
};

INSTANTIATE_TEST_SUITE_P(
    urPlatformGetInfo, urPlatformGetInfoTest,
    ::testing::Values(UR_PLATFORM_INFO_NAME
                      /*
                        UR_PLATFORM_INFO_VENDOR_NAME,
                        UR_PLATFORM_INFO_VERSION,
                        UR_PLATFORM_INFO_EXTENSIONS,
                        UR_PLATFORM_INFO_PROFILE
                      */
                      ),
    [](const ::testing::TestParamInfo<ur_platform_info_t> &info) {
      std::stringstream ss;
      ss << info.param;
      return ss.str();
    });

TEST_P(urPlatformGetInfoTest, Success) {
  size_t size = 0;
  ur_platform_info_t info_type = GetParam();
  ASSERT_SUCCESS(urPlatformGetInfo(platform, info_type, 0, nullptr, &size));
  ASSERT_NE(size, 0);
  std::vector<char> name(size);
  ASSERT_SUCCESS(
      urPlatformGetInfo(platform, info_type, size, name.data(), nullptr));
  ASSERT_EQ(size, std::strlen(name.data()) + 1);
}

TEST_P(urPlatformGetInfoTest, InvalidNullHandlePlatform) {
  size_t size = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urPlatformGetInfo(nullptr, GetParam(), 0, nullptr, &size));
}

TEST_F(urPlatformGetInfoTest, InvalidEnumerationPlatformInfoType) {
  size_t size = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urPlatformGetInfo(platform, UR_PLATFORM_INFO_FORCE_UINT32, 0,
                                     nullptr, &size));
}
