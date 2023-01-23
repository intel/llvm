// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_CONFORMANCE_INCLUDE_FIXTURES_H_INCLUDED
#define UR_CONFORMANCE_INCLUDE_FIXTURES_H_INCLUDED

#include <uur/checks.h>
#include <uur/environment.h>
#include <uur/utils.h>

#define UUR_RETURN_ON_FATAL_FAILURE(...)                                       \
  __VA_ARGS__;                                                                 \
  if (this->HasFatalFailure() || this->IsSkipped()) {                          \
    return;                                                                    \
  }                                                                            \
  (void)0

namespace uur {

struct urPlatformTest : ::testing::Test {
  void SetUp() override {
    platform = uur::PlatformEnvironment::instance->platform;
  }

  ur_platform_handle_t platform = nullptr;
};

inline std::pair<bool, std::vector<ur_device_handle_t>>
GetDevices(ur_platform_handle_t platform) {
  uint32_t count = 0;
  if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count)) {
    return {false, {}};
  }
  if (count == 0) {
    return {false, {}};
  }
  std::vector<ur_device_handle_t> devices(count);
  if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, count, devices.data(),
                  nullptr)) {
    return {false, {}};
  }
  return {true, devices};
}

inline bool
hasDevicePartitionSupport(ur_device_handle_t device,
                          const ur_device_partition_property_flags_t property) {
  ur_device_partition_property_flags_t flags = 0;
  auto result = urDeviceGetInfo(device, UR_DEVICE_INFO_PARTITION_PROPERTIES,
                                sizeof(flags), &flags, nullptr);
  if (result != UR_RESULT_SUCCESS) {
    return false;
  }
  return (flags & property);
}

struct urAllDevicesTest : urPlatformTest {

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::SetUp());
    auto devicesPair = GetDevices(platform);
    if (!devicesPair.first) {
      FAIL() << "Failed to get devices";
    }
    devices = std::move(devicesPair.second);
  }
  std::vector<ur_device_handle_t> devices;
};

struct urDeviceTest : urPlatformTest,
                      ::testing::WithParamInterface<ur_device_handle_t> {

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::SetUp());
    device = GetParam();
  }

  ur_device_handle_t device;
};
} // namespace uur

#define UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(FIXTURE)                           \
  INSTANTIATE_TEST_SUITE_P(                                                    \
      , FIXTURE,                                                               \
      ::testing::ValuesIn(uur::DevicesEnvironment::instance->devices),         \
      [](const ::testing::TestParamInfo<ur_device_handle_t> &info) {           \
        return uur::GetPlatformAndDeviceName(info.param);                      \
      })

namespace uur {

template <class T>
struct urDeviceTestWithParam
    : urPlatformTest,
      ::testing::WithParamInterface<std::tuple<ur_device_handle_t, T>> {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::SetUp());
    device = std::get<0>(this->GetParam());
  }
  // TODO - I don't like the confusion with GetParam();
  const T &getParam() const { return std::get<1>(this->GetParam()); }
  ur_device_handle_t device;
};

struct urContextTest : urDeviceTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urDeviceTest::SetUp());
    ASSERT_SUCCESS(urContextCreate(1, &device, &context));
  }

  void TearDown() override {
    EXPECT_SUCCESS(urContextRelease(context));
    UUR_RETURN_ON_FATAL_FAILURE(urDeviceTest::TearDown());
  }

  ur_context_handle_t context;
};

} // namespace uur

#define UUR_TEST_SUITE_P(FIXTURE, VALUES, PRINTER)                             \
  INSTANTIATE_TEST_SUITE_P(                                                    \
      , FIXTURE,                                                               \
      testing::Combine(                                                        \
          ::testing::ValuesIn(uur::DevicesEnvironment::instance->devices),     \
          VALUES),                                                             \
      PRINTER)

namespace uur {

template <class T> struct urContextTestWithParam : urDeviceTestWithParam<T> {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urDeviceTestWithParam<T>::SetUp());
    ASSERT_SUCCESS(urContextCreate(1, &this->device, &context));
  }

  void TearDown() override {
    EXPECT_SUCCESS(urContextRelease(context));
    UUR_RETURN_ON_FATAL_FAILURE(urDeviceTestWithParam<T>::TearDown());
  }
  ur_context_handle_t context;
};

struct urQueueTest : urContextTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
    ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue));
  }

  void TearDown() override {
    EXPECT_SUCCESS(urQueueRelease(queue));
    urContextTest::TearDown();
  }

  ur_queue_handle_t queue;
};

template <class T> struct urQueueTestWithParam : urContextTestWithParam<T> {

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());
    ASSERT_SUCCESS(urQueueCreate(this->context, this->device, 0, &queue));
  }

  void TearDown() override {
    EXPECT_SUCCESS(urQueueRelease(queue));
    urContextTestWithParam<T>::TearDown();
  }

  ur_queue_handle_t queue;
};

struct urMultiQueueTest : urContextTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
    ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue1));
    ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue2));
  }

  void TearDown() override {
    if (queue1 != nullptr) {
      EXPECT_SUCCESS(urQueueRelease(queue1));
    }
    if (queue2 != nullptr) {
      EXPECT_SUCCESS(urQueueRelease(queue2));
    }
    urContextTest::TearDown();
  }

  ur_queue_handle_t queue1 = nullptr;
  ur_queue_handle_t queue2 = nullptr;
};

struct urMultiDeviceContextTest : urPlatformTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::SetUp());
    auto &devices = DevicesEnvironment::instance->devices;
    if (devices.size() <= 1) {
      GTEST_SKIP();
    }
    ASSERT_SUCCESS(urContextCreate(static_cast<uint32_t>(devices.size()),
                                   devices.data(), &context));
  }

  void TearDown() override {
    if (context) {
      ASSERT_SUCCESS(urContextRelease(context));
    }
    urPlatformTest::TearDown();
  }

  ur_context_handle_t context = nullptr;
};

struct urMultiDeviceMemBufferTest : urMultiDeviceContextTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceContextTest::SetUp());
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                     nullptr, &buffer));
    ASSERT_NE(nullptr, buffer);
  }

  void TearDown() override {
    if (buffer) {
      EXPECT_SUCCESS(urMemRelease(buffer));
    }
    urMultiDeviceContextTest::TearDown();
  }

  ur_mem_handle_t buffer = nullptr;
  const size_t count = 1024;
  const size_t size = count * sizeof(uint32_t);
};

struct urMultiDeviceMemBufferQueueTest : urMultiDeviceMemBufferTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceMemBufferTest::SetUp());
    queues.reserve(DevicesEnvironment::instance->devices.size());
    for (const auto &device : DevicesEnvironment::instance->devices) {
      ur_queue_handle_t queue = nullptr;
      ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue));
      queues.push_back(queue);
    }
  }

  void TearDown() override {
    for (const auto &queue : queues) {
      EXPECT_SUCCESS(urQueueRelease(queue));
    }
    urMultiDeviceMemBufferTest::TearDown();
  }

  std::vector<ur_queue_handle_t> queues;
};

struct urMemBufferQueueTest : urQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                     nullptr, &buffer));
  }

  void TearDown() override {
    if (buffer) {
      EXPECT_SUCCESS(urMemRelease(buffer));
    }
    urQueueTest::TearDown();
  }

  const size_t count = 8;
  const size_t size = sizeof(uint32_t) * count;
  ur_mem_handle_t buffer = nullptr;
};

} // namespace uur

#endif // UR_CONFORMANCE_INCLUDE_FIXTURES_H_INCLUDED
