// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/checks.h>

TEST(urInitTest, Success) {
  ur_platform_init_flags_t platform_flags = 0;
  ur_device_init_flags_t device_flags = 0;
  ASSERT_SUCCESS(urInit(platform_flags, device_flags));

  ASSERT_SUCCESS(urTearDown(nullptr));
}

TEST(urInitTest, DISABLED_ErrorInvalidEnumerationPlatformFlags) {
  const ur_platform_init_flags_t platform_flags =
      UR_PLATFORM_INIT_FLAG_FORCE_UINT32;
  const ur_device_init_flags_t device_flags = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urInit(platform_flags, device_flags));
}

TEST(urInitTest, DISABLED_ErrorInvalidEnumerationDeviceFlags) {
  const ur_platform_init_flags_t platform_flags = 0;
  const ur_device_init_flags_t device_flags =
      UR_PLATFORM_INIT_FLAG_FORCE_UINT32;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urInit(platform_flags, device_flags));
}
