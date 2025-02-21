// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "helpers.h"
#include "uur/known_failure.h"

#include <thread>

#include <uur/fixtures.h>
#include <uur/raii.h>

void checkDevicesSupportSharedUSM(
    const std::vector<ur_device_handle_t> &devices) {
  for (auto device : devices) {
    ur_device_usm_access_capability_flags_t shared_usm_single = 0;
    EXPECT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_single));
    if (!shared_usm_single) {
      GTEST_SKIP() << "Shared USM is not supported by the device.";
    }
  }
}

struct urEnqueueEventsWaitMultiDeviceTest
    : uur::urMultiQueueMultiDeviceTest<2> {
  void SetUp() override {
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

    UUR_RETURN_ON_FATAL_FAILURE(uur::urMultiQueueMultiDeviceTest<2>::SetUp());

    checkDevicesSupportSharedUSM(devices);

    ptrs.resize(devices.size());
    for (size_t i = 0; i < devices.size(); i++) {
      EXPECT_SUCCESS(urUSMSharedAlloc(context, devices[i], nullptr, nullptr,
                                      size, &ptrs[i]));
    }
  }

  void TearDown() override {
    for (auto ptr : ptrs) {
      if (ptr) {
        EXPECT_SUCCESS(urUSMFree(context, ptr));
      }
    }
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urMultiQueueMultiDeviceTest<2>::TearDown());
  }

  void initData() {
    EXPECT_SUCCESS(urEnqueueUSMFill(queues[0], ptrs[0], sizeof(pattern),
                                    &pattern, size, 0, nullptr, nullptr));
    EXPECT_SUCCESS(urQueueFinish(queues[0]));
  }

  void verifyData(void *ptr, uint32_t pattern) {
    for (size_t i = 0; i < count; i++) {
      ASSERT_EQ(reinterpret_cast<uint32_t *>(ptr)[i], pattern);
    }
  }

  uint32_t pattern = 42;
  const size_t count = 1024;
  const size_t size = sizeof(uint32_t) * count;

  std::vector<void *> ptrs;
};
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urEnqueueEventsWaitMultiDeviceTest);

TEST_P(urEnqueueEventsWaitMultiDeviceTest, EmptyWaitList) {
  initData();

  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queues[0], false, ptrs[1], ptrs[0], size, 0,
                                    nullptr, nullptr));
  ASSERT_SUCCESS(urEnqueueEventsWait(queues[0], 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queues[0]));

  verifyData(ptrs[1], pattern);
}

TEST_P(urEnqueueEventsWaitMultiDeviceTest, EmptyWaitListWithEvent) {
  initData();

  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queues[0], false, ptrs[1], ptrs[0], size, 0,
                                    nullptr, nullptr));

  uur::raii::Event event;
  ASSERT_SUCCESS(urEnqueueEventsWait(queues[0], 0, nullptr, event.ptr()));
  ASSERT_SUCCESS(urEventWait(1, event.ptr()));

  verifyData(ptrs[1], pattern);
}

TEST_P(urEnqueueEventsWaitMultiDeviceTest, EnqueueWaitOnADifferentQueue) {
  initData();

  uur::raii::Event event;
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queues[0], false, ptrs[1], ptrs[0], size, 0,
                                    nullptr, event.ptr()));
  ASSERT_SUCCESS(urEnqueueEventsWait(queues[1], 1, event.ptr(), nullptr));
  ASSERT_SUCCESS(urQueueFinish(queues[0]));

  verifyData(ptrs[1], pattern);
}

struct urEnqueueEventsWaitMultiDeviceMTTest
    : uur::urMultiQueueMultiDeviceTestWithParam<8, uur::BoolTestParam> {
  void doComputation(std::function<void(size_t)> work) {
    auto multiThread = getParam().value;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < devices.size(); i++) {
      if (multiThread) {
        threads.emplace_back(work, i);
      } else {
        work(i);
      }
    }
    for (auto &thread : threads) {
      thread.join();
    }
  }

  void SetUp() override {
    UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::NativeCPU{});

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urMultiQueueMultiDeviceTestWithParam<8,
                                                  uur::BoolTestParam>::SetUp());
    checkDevicesSupportSharedUSM(devices);

    ptrs.resize(devices.size());
    for (size_t i = 0; i < devices.size(); i++) {
      EXPECT_SUCCESS(urUSMSharedAlloc(context, devices[i], nullptr, nullptr,
                                      size, &ptrs[i]));
    }
  }

  void TearDown() override {
    for (auto ptr : ptrs) {
      if (ptr) {
        EXPECT_SUCCESS(urUSMFree(context, ptr));
      }
    }
    UUR_RETURN_ON_FATAL_FAILURE(uur::urMultiQueueMultiDeviceTestWithParam<
                                8, uur::BoolTestParam>::TearDown());
  }

  void initData() {
    EXPECT_SUCCESS(urEnqueueUSMFill(queues[0], ptrs[0], sizeof(pattern),
                                    &pattern, size, 0, nullptr, nullptr));
    EXPECT_SUCCESS(urQueueFinish(queues[0]));
  }

  void verifyData(void *ptr, uint32_t pattern) {
    for (size_t i = 0; i < count; i++) {
      ASSERT_EQ(reinterpret_cast<uint32_t *>(ptr)[i], pattern);
    }
  }

  uint32_t pattern = 42;
  const size_t count = 1024;
  const size_t size = sizeof(uint32_t) * count;

  std::vector<void *> ptrs;
};

UUR_PLATFORM_TEST_SUITE_WITH_PARAM(
    urEnqueueEventsWaitMultiDeviceMTTest,
    testing::ValuesIn(uur::BoolTestParam::makeBoolParam("MultiThread")),
    uur::platformTestWithParamPrinter<uur::BoolTestParam>);

TEST_P(urEnqueueEventsWaitMultiDeviceMTTest, EnqueueWaitSingleQueueMultiOps) {
  std::vector<uint32_t> data(count, pattern);

  auto work = [this, &data](size_t i) {
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queues[0], false, ptrs[i], data.data(),
                                      size, 0, nullptr, nullptr));
  };

  doComputation(work);

  auto verify = [this](size_t i) {
    uur::raii::Event event;
    ASSERT_SUCCESS(urEnqueueEventsWait(queues[0], 0, nullptr, event.ptr()));
    ASSERT_SUCCESS(urEventWait(1, event.ptr()));

    verifyData(ptrs[i], pattern);
  };

  doComputation(verify);
}

TEST_P(urEnqueueEventsWaitMultiDeviceMTTest, EnqueueWaitOnAllQueues) {
  // Fails when -fsanitize=cfi
#ifdef UR_USE_CFI
  if (getParam().value) {
    UUR_KNOWN_FAILURE_ON(uur::OpenCL{});
  }
#endif

  std::vector<uur::raii::Event> eventsRaii(devices.size());
  std::vector<ur_event_handle_t> events(devices.size());
  auto work = [this, &events, &eventsRaii](size_t i) {
    ASSERT_SUCCESS(urEnqueueUSMFill(queues[i], ptrs[i], sizeof(pattern),
                                    &pattern, size, 0, nullptr,
                                    eventsRaii[i].ptr()));
    events[i] = eventsRaii[i].get();
  };

  doComputation(work);

  uur::raii::Event gatherEvent;
  ASSERT_SUCCESS(urEnqueueEventsWait(queues[0], devices.size(), events.data(),
                                     gatherEvent.ptr()));
  ASSERT_SUCCESS(urEventWait(1, gatherEvent.ptr()));

  for (size_t i = 0; i < devices.size(); i++) {
    verifyData(ptrs[i], pattern);
  }
}

TEST_P(urEnqueueEventsWaitMultiDeviceMTTest,
       EnqueueWaitOnAllQueuesCommonDependency) {
  uur::raii::Event event;
  ASSERT_SUCCESS(urEnqueueUSMFill(queues[0], ptrs[0], sizeof(pattern), &pattern,
                                  size, 0, nullptr, event.ptr()));

  std::vector<uur::raii::Event> perQueueEvents(devices.size());
  std::vector<ur_event_handle_t> eventHandles(devices.size());
  auto work = [this, &event, &perQueueEvents, &eventHandles](size_t i) {
    ASSERT_SUCCESS(urEnqueueEventsWait(queues[i], 1, event.ptr(),
                                       perQueueEvents[i].ptr()));
    eventHandles[i] = perQueueEvents[i].get();
  };

  doComputation(work);

  uur::raii::Event hGatherEvent;
  ASSERT_SUCCESS(urEnqueueEventsWait(queues[0], eventHandles.size(),
                                     eventHandles.data(), hGatherEvent.ptr()));
  ASSERT_SUCCESS(urEventWait(1, hGatherEvent.ptr()));

  for (auto &event : eventHandles) {
    ur_event_status_t status;
    ASSERT_SUCCESS(urEventGetInfo(event, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                  sizeof(ur_event_status_t), &status, nullptr));
    ASSERT_EQ(status, UR_EVENT_STATUS_COMPLETE);
  }

  verifyData(ptrs[0], pattern);
}
