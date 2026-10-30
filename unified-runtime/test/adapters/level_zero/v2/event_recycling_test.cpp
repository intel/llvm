// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %with-v2 ./event_recycling-test
// REQUIRES: v2

#include "../ze_helpers.hpp"
#include "uur/checks.h"
#include "uur/fixtures.h"
#include "uur/utils.h"
#include "ze_api.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <set>
#include <vector>

struct EventRecyclingTest : uur::urQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());

    ur_bool_t reusableEventsSupported = false;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_REUSABLE_EVENTS_SUPPORT_EXP,
        sizeof(reusableEventsSupported), &reusableEventsSupported, nullptr));
    if (!reusableEventsSupported) {
      GTEST_SKIP() << "Reusable events are not supported";
    }
  }

  void createEvent(ur_exp_event_flags_t flags, ur_event_handle_t *phEvent) {
    ur_exp_event_desc_t desc{UR_STRUCTURE_TYPE_EXP_EVENT_DESC, nullptr, flags};
    ASSERT_SUCCESS(urEventCreateExp(context, device, &desc, phEvent));
  }

  ur_native_handle_t getNativeHandle(ur_event_handle_t hEvent) {
    ur_native_handle_t hNative = 0;
    EXPECT_SUCCESS(urEventGetNativeHandle(hEvent, &hNative));
    return hNative;
  }

  ur_event_status_t getStatus(ur_event_handle_t hEvent) {
    ur_event_status_t status = UR_EVENT_STATUS_QUEUED;
    EXPECT_SUCCESS(urEventGetInfo(hEvent,
                                  UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                  sizeof(status), &status, nullptr));
    return status;
  }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(EventRecyclingTest);

TEST_P(EventRecyclingTest, ForceEventCreation) {
  constexpr size_t numEvents = 8;

  std::vector<ur_event_handle_t> events(numEvents);
  for (auto &hEvent : events) {
    ASSERT_NO_FATAL_FAILURE(createEvent(0, &hEvent));
  }

  std::set<ur_event_handle_t> released;
  std::set<ur_native_handle_t> releasedNative;
  for (auto hEvent : events) {
    released.insert(hEvent);
    releasedNative.insert(getNativeHandle(hEvent));
    ASSERT_SUCCESS(urEventRelease(hEvent));
  }

  // Without the flag, a released event is reused.
  ur_event_handle_t reused = nullptr;
  ASSERT_NO_FATAL_FAILURE(createEvent(0, &reused));
  EXPECT_EQ(released.count(reused), 1u);
  ASSERT_SUCCESS(urEventRelease(reused));

  // With the flag, a new event is created although reusable ones are free.
  ur_event_handle_t forced = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      createEvent(UR_EXP_EVENT_FLAG_FORCE_EVENT_CREATION, &forced));
  EXPECT_EQ(released.count(forced), 0u);
  EXPECT_EQ(releasedNative.count(getNativeHandle(forced)), 0u);
  EXPECT_EQ(getStatus(forced), UR_EVENT_STATUS_COMPLETE);
  ASSERT_SUCCESS(urEventRelease(forced));

  // The forced event is still returned to the pool on release.
  ur_event_handle_t next = nullptr;
  ASSERT_NO_FATAL_FAILURE(createEvent(0, &next));
  EXPECT_EQ(next, forced);
  ASSERT_SUCCESS(urEventRelease(next));
}

TEST_P(EventRecyclingTest, ForceEventCreationSkipsPendingEvent) {
  auto zeBlocker = createZeEvent(context, device);
  ASSERT_NE(zeBlocker, nullptr);

  ur_event_native_properties_t nativeProps{
      UR_STRUCTURE_TYPE_EVENT_NATIVE_PROPERTIES, nullptr, false};
  ur_event_handle_t hBlocker = nullptr;
  ASSERT_SUCCESS(urEventCreateWithNativeHandle(
      reinterpret_cast<ur_native_handle_t>(zeBlocker.get()), context,
      &nativeProps, &hBlocker));

  ur_event_handle_t pending = nullptr;
  ASSERT_NO_FATAL_FAILURE(createEvent(0, &pending));

  ur_exp_enqueue_ext_properties_t enqueueProps{
      UR_STRUCTURE_TYPE_EXP_ENQUEUE_EXT_PROPERTIES, nullptr, 0};
  ASSERT_SUCCESS(urEnqueueEventsWaitWithBarrierExt(queue, &enqueueProps, 1,
                                                   &hBlocker, &pending));
  ASSERT_EQ(getStatus(pending), UR_EVENT_STATUS_SUBMITTED);
  ASSERT_SUCCESS(urEventRelease(pending));

  // The forced event is not bound to the released event's pending work.
  ur_event_handle_t forced = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      createEvent(UR_EXP_EVENT_FLAG_FORCE_EVENT_CREATION, &forced));
  EXPECT_NE(forced, pending);
  EXPECT_EQ(getStatus(forced), UR_EVENT_STATUS_COMPLETE);

  ASSERT_EQ(zeEventHostSignal(zeBlocker.get()), ZE_RESULT_SUCCESS);
  ASSERT_SUCCESS(urQueueFinish(queue));

  ur_event_handle_t recycled = nullptr;
  ASSERT_NO_FATAL_FAILURE(createEvent(0, &recycled));
  EXPECT_EQ(recycled, pending);

  ASSERT_SUCCESS(urEventRelease(recycled));
  ASSERT_SUCCESS(urEventRelease(forced));
  ASSERT_SUCCESS(urEventRelease(hBlocker));
}

struct EventRecyclingGraphTest : uur::urMultiQueueTypeTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urMultiQueueTypeTest::SetUp());

    ur_bool_t graphSupported = false;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP,
        sizeof(graphSupported), &graphSupported, nullptr));
    if (!graphSupported) {
      GTEST_SKIP() << "Graph record and replay is not supported";
    }

    // A batched queue takes captured events from a different pool.
    bool batched = false;
    ASSERT_NO_FATAL_FAILURE(uur::isQueueBatched(queue, &batched));
    if (batched) {
      GTEST_SKIP() << "Not applicable to batched queues";
    }

    ur_device_usm_access_capability_flags_t usmSupport = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, usmSupport));
    if (!usmSupport) {
      GTEST_SKIP() << "Device USM is not supported";
    }

    ASSERT_SUCCESS(
        urUSMDeviceAlloc(context, device, nullptr, nullptr, size, &src));
    ASSERT_SUCCESS(
        urUSMDeviceAlloc(context, device, nullptr, nullptr, size, &dst));
  }

  void TearDown() override {
    if (src) {
      EXPECT_SUCCESS(urUSMFree(context, src));
    }
    if (dst) {
      EXPECT_SUCCESS(urUSMFree(context, dst));
    }
    UUR_RETURN_ON_FATAL_FAILURE(urMultiQueueTypeTest::TearDown());
  }

  static constexpr size_t size = 1024;
  void *src = nullptr;
  void *dst = nullptr;
};

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    EventRecyclingGraphTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(EventRecyclingGraphTest, CapturedEventsAreKeptUntilGraphDestruction) {
  constexpr uint8_t pattern = 0xAB;
  constexpr uint8_t zero = 0;
  constexpr size_t numStealAttempts = 16;

  ur_exp_graph_handle_t graph = nullptr;
  ASSERT_SUCCESS(urGraphCreateExp(context, &graph));
  ASSERT_SUCCESS(urQueueBeginCaptureIntoGraphExp(queue, graph));

  ur_event_handle_t captured = nullptr;
  {
    ur_event_handle_t fillEvent = nullptr;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, src, sizeof(pattern), &pattern, size,
                                    0, nullptr, &fillEvent));
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, dst, src, size, 1,
                                      &fillEvent, nullptr));
    captured = fillEvent;
    // Drop the only user reference, as ~sycl::event would.
    ASSERT_SUCCESS(urEventRelease(fillEvent));
  }

  ur_exp_graph_handle_t sameGraph = nullptr;
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue, &sameGraph));
  ASSERT_EQ(sameGraph, graph);

  ur_exp_executable_graph_handle_t exGraph = nullptr;
  ASSERT_SUCCESS(urGraphInstantiateGraphExp(graph, &exGraph));

  for (int replay = 0; replay < 3; ++replay) {
    std::vector<ur_event_handle_t> events(numStealAttempts);
    for (auto &hEvent : events) {
      ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &hEvent));
      EXPECT_NE(hEvent, captured);
    }
    for (auto hEvent : events) {
      ASSERT_SUCCESS(urEventRelease(hEvent));
    }

    ASSERT_SUCCESS(urEnqueueUSMFill(queue, src, sizeof(zero), &zero, size, 0,
                                    nullptr, nullptr));
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, dst, sizeof(zero), &zero, size, 0,
                                    nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    ASSERT_SUCCESS(urEnqueueGraphExp(queue, exGraph, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    std::vector<uint8_t> host(size);
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, host.data(), dst, size, 0,
                                      nullptr, nullptr));
    ASSERT_EQ(host, std::vector<uint8_t>(size, pattern));
  }

  ASSERT_SUCCESS(urGraphExecutableGraphDestroyExp(exGraph));
  ASSERT_SUCCESS(urGraphDestroyExp(graph));
}
