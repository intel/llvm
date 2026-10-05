// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>
#include <uur/known_failure.h>
#include <uur/raii.h>

#include <atomic>
#include <thread>
#include <vector>

using urQueueGetInfoTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urQueueGetInfoTest);

TEST_P(urQueueGetInfoTest, SuccessContext) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_queue_info_t property_name = UR_QUEUE_INFO_CONTEXT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urQueueGetInfo(queue, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_context_handle_t), property_size);

  ur_context_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urQueueGetInfo(queue, property_name, property_size,
                                &property_value, nullptr));

  ASSERT_EQ(context, property_value);
}

TEST_P(urQueueGetInfoTest, SuccessRoundtripContext) {
  const ur_queue_info_t property_name = UR_QUEUE_INFO_CONTEXT;
  size_t property_size = sizeof(ur_context_handle_t);

  ur_native_handle_t native_queue;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urQueueGetNativeHandle(queue, nullptr, &native_queue));

  ur_queue_handle_t from_native_queue;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urQueueCreateWithNativeHandle(
      native_queue, context, device, nullptr, &from_native_queue));

  ur_context_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urQueueGetInfo(from_native_queue, property_name, property_size,
                                &property_value, nullptr));

  ASSERT_EQ(property_value, context);
}

TEST_P(urQueueGetInfoTest, SuccessDevice) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_queue_info_t property_name = UR_QUEUE_INFO_DEVICE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urQueueGetInfo(queue, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_device_handle_t), property_size);

  ur_device_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urQueueGetInfo(queue, property_name, property_size,
                                &property_value, nullptr));

  ASSERT_EQ(device, property_value);
}

TEST_P(urQueueGetInfoTest, SuccessRoundtripDevice) {
  // Segfaults
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  const ur_queue_info_t property_name = UR_QUEUE_INFO_DEVICE;
  size_t property_size = 0;

  ur_native_handle_t native_queue;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urQueueGetNativeHandle(queue, nullptr, &native_queue));

  ur_queue_handle_t from_native_queue;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urQueueCreateWithNativeHandle(
      native_queue, context, device, nullptr, &from_native_queue));

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(urQueueGetInfo(from_native_queue,
                                                  property_name, 0, nullptr,
                                                  &property_size),
                                   property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_handle_t));

  ur_device_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urQueueGetInfo(from_native_queue, property_name, property_size,
                                &property_value, nullptr));

  ASSERT_EQ(property_value, device);
}

TEST_P(urQueueGetInfoTest, SuccessRoundtripNullDevice) {
  // Segfaults
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  const ur_queue_info_t property_name = UR_QUEUE_INFO_DEVICE;
  size_t property_size = 0;

  ur_native_handle_t native_queue;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urQueueGetNativeHandle(queue, nullptr, &native_queue));

  ur_queue_handle_t from_native_queue;
  auto result = urQueueCreateWithNativeHandle(native_queue, context, nullptr,
                                              nullptr, &from_native_queue);
  if (result == UR_RESULT_ERROR_INVALID_NULL_HANDLE) {
    GTEST_SKIP() << "Implementation requires a valid device";
  }
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(result);

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(urQueueGetInfo(from_native_queue,
                                                  property_name, 0, nullptr,
                                                  &property_size),
                                   property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_handle_t));

  ur_device_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urQueueGetInfo(from_native_queue, property_name, property_size,
                                &property_value, nullptr));

  ASSERT_EQ(property_value, device);
}

TEST_P(urQueueGetInfoTest, SuccessFlags) {
  size_t property_size = 0;
  const ur_queue_info_t property_name = UR_QUEUE_INFO_FLAGS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urQueueGetInfo(queue, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_queue_flags_t), property_size);

  ur_queue_flags_t property_value = 0;
  ASSERT_SUCCESS(urQueueGetInfo(queue, property_name, property_size,
                                &property_value, nullptr));

  EXPECT_EQ(property_value, queue_properties.flags);
}

TEST_P(urQueueGetInfoTest, SuccessReferenceCount) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_queue_info_t property_name = UR_QUEUE_INFO_REFERENCE_COUNT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urQueueGetInfo(queue, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(uint32_t), property_size);

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urQueueGetInfo(queue, property_name, property_size,
                                            &property_value, nullptr),
                             property_value);

  ASSERT_GT(property_value, 0U);
}

TEST_P(urQueueGetInfoTest, SuccessEmptyQueue) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_queue_info_t property_name = UR_QUEUE_INFO_EMPTY;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urQueueGetInfo(queue, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_bool_t), property_size);
}

TEST_P(urQueueGetInfoTest, SuccessEmptyConcurrentEnqueues) {
  ur_bool_t empty = false;
  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(urQueueGetInfo(queue, UR_QUEUE_INFO_EMPTY,
                                                  sizeof(empty), &empty,
                                                  nullptr),
                                   UR_QUEUE_INFO_EMPTY);

  constexpr size_t numThreads = 8;
  constexpr size_t numEvents = 128;
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  for (size_t i = 0; i < numThreads; ++i) {
    threads.emplace_back([&] {
      std::vector<uur::raii::Event> events(numEvents);
      while (!start.load()) {
        std::this_thread::yield();
      }
      for (size_t j = 0; j < numEvents; ++j) {
        ASSERT_SUCCESS(urEnqueueEventsWait(queue, j ? 1 : 0,
                                           j ? events[j - 1].ptr() : nullptr,
                                           events[j].ptr()));

        // Also invalidate the cached last event without returning an event.
        ASSERT_SUCCESS(urEnqueueEventsWait(queue, 1, events[j].ptr(), nullptr));
        ur_bool_t emptyDuringSubmission = false;
        ASSERT_SUCCESS(urQueueGetInfo(queue, UR_QUEUE_INFO_EMPTY,
                                      sizeof(emptyDuringSubmission),
                                      &emptyDuringSubmission, nullptr));
      }

      ASSERT_SUCCESS(urEventWait(1, events.back().ptr()));
      // Updating the queue's event cache must not consume references owned
      // by the application, including earlier events in each thread's chain.
      for (auto &event : events) {
        ur_event_status_t status = UR_EVENT_STATUS_QUEUED;
        ASSERT_SUCCESS(urEventGetInfo(event,
                                      UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                      sizeof(status), &status, nullptr));
        ASSERT_EQ(status, UR_EVENT_STATUS_COMPLETE);
      }
    });
  }
  start.store(true);
  for (auto &thread : threads) {
    thread.join();
  }

  uur::raii::Event finalEvent;
  ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, finalEvent.ptr()));
  ASSERT_SUCCESS(urQueueFinish(queue));
  ASSERT_SUCCESS(urQueueGetInfo(queue, UR_QUEUE_INFO_EMPTY, sizeof(empty),
                                &empty, nullptr));
  ASSERT_TRUE(empty);
}

TEST_P(urQueueGetInfoTest, InvalidNullHandleQueue) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urQueueGetInfo(nullptr, UR_QUEUE_INFO_CONTEXT,
                                  sizeof(ur_context_handle_t), &context,
                                  nullptr));
}

TEST_P(urQueueGetInfoTest, InvalidEnumerationProperty) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urQueueGetInfo(queue, UR_QUEUE_INFO_FORCE_UINT32,
                                  sizeof(ur_context_handle_t), &context,
                                  nullptr));
}

TEST_P(urQueueGetInfoTest, InvalidSizeZero) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_SIZE,
      urQueueGetInfo(queue, UR_QUEUE_INFO_CONTEXT, 0, &context, nullptr));
}

TEST_P(urQueueGetInfoTest, InvalidSizeSmall) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  ur_context_handle_t context = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urQueueGetInfo(queue, UR_QUEUE_INFO_CONTEXT,
                                  sizeof(ur_context_handle_t) - 1, &context,
                                  nullptr));
}

TEST_P(urQueueGetInfoTest, InvalidNullPointerPropValue) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urQueueGetInfo(queue, UR_QUEUE_INFO_CONTEXT,
                                  sizeof(ur_context_handle_t), nullptr,
                                  nullptr));
}

TEST_P(urQueueGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urQueueGetInfo(queue, UR_QUEUE_INFO_CONTEXT, 0, nullptr, nullptr));
}

struct urQueueGetInfoDeviceQueueTestWithInfoParam : public uur::urQueueTest {
  void SetUp() override {
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});
    UUR_RETURN_ON_FATAL_FAILURE(urQueueGetInfoTest::SetUp());
    ur_queue_flags_t deviceQueueCapabilities = 0;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES,
        sizeof(deviceQueueCapabilities), &deviceQueueCapabilities, nullptr));
    if (!deviceQueueCapabilities) {
      GTEST_SKIP() << "Queue on device is not supported.";
    }
    ASSERT_SUCCESS(urQueueCreate(context, device, &queueProperties, &queue));
  }

  void TearDown() override {
    if (queue) {
      ASSERT_SUCCESS(urQueueRelease(queue));
    }
    urQueueGetInfoTest::TearDown();
  }

  ur_queue_handle_t queue = nullptr;
  const ur_queue_properties_t queueProperties = {
      UR_STRUCTURE_TYPE_QUEUE_PROPERTIES, nullptr,
      UR_QUEUE_FLAG_ON_DEVICE | UR_QUEUE_FLAG_ON_DEVICE_DEFAULT |
          UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urQueueGetInfoDeviceQueueTestWithInfoParam);

TEST_P(urQueueGetInfoDeviceQueueTestWithInfoParam, SuccessDeviceDefault) {
  size_t property_size = 0;
  const ur_queue_info_t property_name = UR_QUEUE_INFO_DEVICE_DEFAULT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urQueueGetInfo(queue, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_queue_handle_t), property_size);

  ur_queue_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urQueueGetInfo(queue, property_name, property_size,
                                &property_value, nullptr));

  ASSERT_EQ(queue, property_value);
}

TEST_P(urQueueGetInfoDeviceQueueTestWithInfoParam, SuccessSize) {
  size_t property_size = 0;
  const ur_queue_info_t property_name = UR_QUEUE_INFO_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urQueueGetInfo(queue, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(uint32_t), property_size);

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urQueueGetInfo(queue, property_name, property_size,
                                            &property_value, nullptr),
                             property_value);

  ASSERT_GT(property_value, 0);
}
