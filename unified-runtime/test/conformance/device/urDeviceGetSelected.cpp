// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

#ifdef _MSC_VER
#include <Windows.h>
#endif

namespace uur {
static int set_env(const char *name, const char *value) {
#ifdef _MSC_VER
  return _putenv_s(name, value);
#else
  return setenv(name, value, 1);
#endif
}

static int unset_env(const char *name) {
#ifdef _MSC_VER
  return _putenv_s(name, "");
#else
  return unsetenv(name);
#endif
}

} // namespace uur

struct urDeviceGetSelectedTest : uur::urPlatformTest {
  void SetUp() {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urPlatformTest::SetUp());

    // These tests require at least one device in the platform
    uint32_t totalCount = 0;
    ASSERT_SUCCESS(
        urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &totalCount));
    if (!totalCount) {
      GTEST_SKIP() << "Platform has no devices";
    }
  }
};
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urDeviceGetSelectedTest);

/* adpater agnostic tests -- none assume the existence or support of any
 * specific adapter */

TEST_P(urDeviceGetSelectedTest, Success) {
  uur::unset_env("ONEAPI_DEVICE_SELECTOR");
  uint32_t count = 0;
  ASSERT_SUCCESS(
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_NE(count, 0);
  std::vector<ur_device_handle_t> devices(count);
  ASSERT_SUCCESS(urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, count,
                                     devices.data(), nullptr));
  for (auto &device : devices) {
    ASSERT_NE(nullptr, device);
  }
}

TEST_P(urDeviceGetSelectedTest, SuccessSubsetOfDevices) {
  uur::unset_env("ONEAPI_DEVICE_SELECTOR");
  uint32_t count = 0;
  ASSERT_SUCCESS(
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  if (count < 2) {
    GTEST_SKIP() << "There are fewer than two devices in total for the "
                    "platform so the subset test is impossible";
  }
  std::vector<ur_device_handle_t> devices(count - 1);
  ASSERT_SUCCESS(urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, count - 1,
                                     devices.data(), nullptr));
  for (auto *device : devices) {
    ASSERT_NE(nullptr, device);
  }
}

TEST_P(urDeviceGetSelectedTest, SuccessSelected_StarColonStar) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*:*");
  uint32_t count = 0;
  ASSERT_SUCCESS(
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_NE(count, 0);
  std::vector<ur_device_handle_t> devices(count);
  ASSERT_SUCCESS(urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, count,
                                     devices.data(), nullptr));
  for (auto &device : devices) {
    ASSERT_NE(nullptr, device);
  }

  uint32_t countAll = 0;
  ASSERT_SUCCESS(
      urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &countAll));
  ASSERT_NE(countAll, 0);
  ASSERT_EQ(countAll, count);
  std::vector<ur_device_handle_t> devicesAll(countAll);
  ASSERT_SUCCESS(urDeviceGet(platform, UR_DEVICE_TYPE_ALL, countAll,
                             devicesAll.data(), nullptr));
  for (auto &device : devicesAll) {
    ASSERT_NE(nullptr, device);
  }

  for (size_t i = 0; i < count; ++i) {
    ASSERT_EQ(devices[i], devicesAll[i]);
  }
}

TEST_P(urDeviceGetSelectedTest, SuccessSelected_StarColonZero) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*:0");
  uint32_t count = 0;
  ASSERT_SUCCESS(
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_NE(count, 0);
  std::vector<ur_device_handle_t> devices(count);
  ASSERT_SUCCESS(urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, count,
                                     devices.data(), nullptr));
  for (auto &device : devices) {
    ASSERT_NE(nullptr, device);
  }
}

TEST_P(urDeviceGetSelectedTest, SuccessSelected_StarColonZeroCommaStar) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*:0,*");
  uint32_t count = 0;
  ASSERT_SUCCESS(
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_NE(count, 0);
  std::vector<ur_device_handle_t> devices(count);
  ASSERT_SUCCESS(urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, count,
                                     devices.data(), nullptr));
  for (auto &device : devices) {
    ASSERT_NE(nullptr, device);
  }
}

TEST_P(urDeviceGetSelectedTest, SuccessSelected_DiscardStarColonStar) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "!*:*");
  uint32_t count = 0;
  ASSERT_SUCCESS(
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_EQ(count, 0);
}

TEST_P(urDeviceGetSelectedTest, SuccessSelected_SelectAndDiscard) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*:0;!*:*");
  uint32_t count = 0;
  ASSERT_SUCCESS(
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_EQ(count, 0);
}

TEST_P(urDeviceGetSelectedTest,
       SuccessSelected_SelectSomethingAndDiscardSomethingElse) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*:0;!*:1");
  uint32_t count = 0;
  ASSERT_SUCCESS(
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_NE(count, 0);
  std::vector<ur_device_handle_t> devices(count);
  ASSERT_SUCCESS(urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, count,
                                     devices.data(), nullptr));
  for (auto &device : devices) {
    ASSERT_NE(nullptr, device);
  }
}

TEST_P(urDeviceGetSelectedTest, InvalidNullHandlePlatform) {
  uur::unset_env("ONEAPI_DEVICE_SELECTOR");
  uint32_t count = 0;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urDeviceGetSelected(nullptr, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
}

TEST_P(urDeviceGetSelectedTest, InvalidEnumerationDevicesType) {
  uur::unset_env("ONEAPI_DEVICE_SELECTOR");
  uint32_t count = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urDeviceGetSelected(platform, UR_DEVICE_TYPE_FORCE_UINT32, 0,
                                       nullptr, &count));
}

TEST_P(urDeviceGetSelectedTest, InvalidValueNumEntries) {
  uur::unset_env("ONEAPI_DEVICE_SELECTOR");
  uint32_t count = 0;
  ASSERT_SUCCESS(
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_NE(count, 0);
  std::vector<ur_device_handle_t> devices(count);
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0,
                                       devices.data(), nullptr));
}

TEST_P(urDeviceGetSelectedTest, InvalidMissingBackend) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", ":garbage");
  uint32_t count = 0;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_VALUE,
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_EQ(count, 0);
}

TEST_P(urDeviceGetSelectedTest, InvalidGarbageBackendString) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "garbage:0");
  uint32_t count = 0;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_VALUE,
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_EQ(count, 0);
}

TEST_P(urDeviceGetSelectedTest, SuccessCaseSensitive) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "OpEnCl:0");
  uint32_t count = 0;
  ASSERT_SUCCESS(
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
}

TEST_P(urDeviceGetSelectedTest, InvalidMissingFilterStrings) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*");
  uint32_t count = 0;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_VALUE,
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_EQ(count, 0);
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*:");
  uint32_t count2 = 0;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_VALUE,
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count2));
  ASSERT_EQ(count2, 0);
}

TEST_P(urDeviceGetSelectedTest, InvalidMissingFilterString) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*:0,,2");
  uint32_t count = 0;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_VALUE,
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_EQ(count, 0);
}

TEST_P(urDeviceGetSelectedTest, InvalidTooManyDotsInFilterString) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*:0.1.2.3");
  uint32_t count = 0;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_VALUE,
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_EQ(count, 0);
}

TEST_P(urDeviceGetSelectedTest, InvalidBadWildardInFilterString) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*:*.");
  uint32_t count = 0;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_VALUE,
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_EQ(count, 0);
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*:*.0");
  uint32_t count2 = 0;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_VALUE,
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count2));
  ASSERT_EQ(count2, 0);
}

TEST_P(urDeviceGetSelectedTest, InvalidSelectingNonexistentDevice) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*:4321");
  uint32_t count = 0;
  ASSERT_SUCCESS(
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_EQ(count, 0);
}

TEST_P(urDeviceGetSelectedTest, InvalidSelectingNonexistentSubDevice) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*:0.4321");
  uint32_t count = 0;
  ASSERT_SUCCESS(
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_EQ(count, 0);
}

TEST_P(urDeviceGetSelectedTest, InvalidSelectingNonexistentSubSubDevice) {
  uur::set_env("ONEAPI_DEVICE_SELECTOR", "*:0.0.4321");
  uint32_t count = 0;
  ASSERT_SUCCESS(
      urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
  ASSERT_EQ(count, 0);
}
