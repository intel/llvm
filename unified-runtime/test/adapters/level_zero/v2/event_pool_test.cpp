// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "command_list_cache.hpp"

#include "level_zero/common.hpp"
#include "level_zero/device.hpp"

#include "../ze_helpers.hpp"
#include "context.hpp"
#include "event_pool.hpp"
#include "event_pool_cache.hpp"
#include "event_provider.hpp"
#include "event_provider_counter.hpp"
#include "event_provider_normal.hpp"
#include "queue_handle.hpp"
#include "uur/fixtures.h"
#include "ze_api.h"

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <unordered_set>

using namespace v2;

static constexpr size_t MAX_DEVICES = 10;

// mock necessary functions from context, we can't pull in entire context
// implementation due to a lot of other dependencies
std::vector<ur_device_handle_t> mockVec{};
const std::vector<ur_device_handle_t> &
ur_context_handle_t_::getDevices() const {
  return mockVec;
}

enum ProviderType {
  TEST_PROVIDER_NORMAL,
  TEST_PROVIDER_COUNTER,
};

static const char *provider_to_str(ProviderType p) {
  switch (p) {
  case TEST_PROVIDER_NORMAL:
    return "provider_normal";
  case TEST_PROVIDER_COUNTER:
    return "provider_counter";
  default:
    return nullptr;
  }
}

static std::string flags_to_str(event_flags_t flags) {
  std::string str;
  if (flags & EVENT_FLAGS_COUNTER) {
    str += "provider_counter";
  } else {
    str += "provider_normal";
  }

  if (flags & EVENT_FLAGS_PROFILING_ENABLED) {
    str += "_profiling";
  } else {
    str += "_no_profiling";
  }

  return str;
}

static const char *queue_to_str(queue_type e) {
  switch (e) {
  case QUEUE_REGULAR:
    return "QUEUE_REGULAR";
  case QUEUE_IMMEDIATE:
    return "QUEUE_IMMEDIATE";
  default:
    return nullptr;
  }
}

struct ProviderParams {
  ProviderType provider;
  event_flags_t flags;
  v2::queue_type queue;
};

template <typename T>
inline std::string
printParams(const testing::TestParamInfo<typename T::ParamType> &info) {
  const auto device_handle = std::get<0>(info.param).device;
  const auto platform_device_name =
      uur::GetPlatformAndDeviceName(device_handle);
  auto params = std::get<1>(info.param);

  std::ostringstream params_stream;
  params_stream << platform_device_name << "__"
                << provider_to_str(params.provider) << "_"
                << flags_to_str(params.flags) << "_"
                << queue_to_str(params.queue);
  return params_stream.str();
}

struct EventPoolTest : public uur::urQueueTestWithParam<ProviderParams> {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTestWithParam::SetUp());

    auto params = getParam();

    // Initialize Level Zero driver is required if this test is linked
    // statically with Level Zero loader, the driver will not be init otherwise.
    zeInit(ZE_INIT_FLAG_GPU_ONLY);

    mockVec.push_back(device);

    cache = std::unique_ptr<event_pool_cache>(new event_pool_cache(
        nullptr, MAX_DEVICES,
        [this, params](DeviceId,
                       event_flags_t flags) -> std::unique_ptr<event_provider> {
          // normally id would be used to find the appropriate device to create
          // the provider
          switch (params.provider) {
          case TEST_PROVIDER_COUNTER:
            return std::make_unique<provider_counter>(platform, context,
                                                      device);
          case TEST_PROVIDER_NORMAL:
            return std::make_unique<provider_normal>(context, params.queue,
                                                     flags);
          }
          return nullptr;
        }));
  }
  void TearDown() override {
    cache.reset();
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTestWithParam::TearDown());
  }

  std::unique_ptr<event_pool_cache> cache;
};

static ProviderParams test_cases[] = {
    {TEST_PROVIDER_NORMAL, 0, QUEUE_REGULAR},
    {TEST_PROVIDER_NORMAL, EVENT_FLAGS_COUNTER, QUEUE_REGULAR},
    {TEST_PROVIDER_NORMAL, EVENT_FLAGS_COUNTER, QUEUE_IMMEDIATE},
    {TEST_PROVIDER_NORMAL, EVENT_FLAGS_COUNTER | EVENT_FLAGS_PROFILING_ENABLED,
     QUEUE_IMMEDIATE},
    // TODO: counter provided is not fully unimplemented
    // counter-based provider ignores event and queue type
    //{TEST_PROVIDER_COUNTER, EVENT_COUNTER, QUEUE_IMMEDIATE},
};

UUR_DEVICE_TEST_SUITE_WITH_PARAM(EventPoolTest, testing::ValuesIn(test_cases),
                                 printParams<EventPoolTest>);

TEST_P(EventPoolTest, InvalidDevice) {
  auto pool = cache->borrow(MAX_DEVICES, getParam().flags);
  ASSERT_EQ(pool, nullptr);
  pool = cache->borrow(MAX_DEVICES + 10, getParam().flags);
  ASSERT_EQ(pool, nullptr);
}

TEST_P(EventPoolTest, Basic) {
  {
    ur_event_handle_t first;
    ze_event_handle_t zeFirst;
    {
      auto pool = cache->borrow(device->Id.value(), getParam().flags);

      first = pool->allocate();
      first->resetQueueAndCommand(&queue->get(), UR_COMMAND_KERNEL_LAUNCH);
      zeFirst = first->getZeEvent();

      urEventRelease(first);
    }
    ur_event_handle_t second;
    ze_event_handle_t zeSecond;
    {
      auto pool = cache->borrow(device->Id.value(), getParam().flags);

      second = pool->allocate();
      first->resetQueueAndCommand(&queue->get(), UR_COMMAND_KERNEL_LAUNCH);
      zeSecond = second->getZeEvent();

      urEventRelease(second);
    }
    ASSERT_EQ(first, second);
    ASSERT_EQ(zeFirst, zeSecond);
  }
}

TEST_P(EventPoolTest, Threaded) {
  std::vector<std::thread> threads;

  for (int iters = 0; iters < 3; ++iters) {
    for (int th = 0; th < 10; ++th) {
      threads.emplace_back([&] {
        auto pool = cache->borrow(device->Id.value(), getParam().flags);
        std::vector<ur_event_handle_t> events;
        for (int i = 0; i < 100; ++i) {
          events.push_back(pool->allocate());
          events.back()->resetQueueAndCommand(&queue->get(),
                                              UR_COMMAND_KERNEL_LAUNCH);
        }
        for (int i = 0; i < 100; ++i) {
          urEventRelease(events[i]);
        }
      });
    }
    for (auto &thread : threads) {
      thread.join();
    }
    threads.clear();
  }
}

TEST_P(EventPoolTest, ProviderNormalUseMostFreePool) {
  auto pool = cache->borrow(device->Id.value(), getParam().flags);
  std::list<ur_event_handle_t> events;
  for (int i = 0; i < 128; ++i) {
    auto event = pool->allocate();
    event->resetQueueAndCommand(&queue->get(), UR_COMMAND_KERNEL_LAUNCH);
    events.push_back(event);
  }
  auto frontZeHandle = events.front()->getZeEvent();
  for (int i = 0; i < 8; ++i) {
    urEventRelease(events.front());
    events.pop_front();
  }
  for (int i = 0; i < 8; ++i) {
    auto e = pool->allocate();
    e->resetQueueAndCommand(&queue->get(), UR_COMMAND_KERNEL_LAUNCH);
    events.push_back(e);
  }

  // the ZeEvent handles from the first provider pool will be reused
  ASSERT_EQ(frontZeHandle, events.back()->getZeEvent());

  for (auto e : events) {
    urEventRelease(e);
  }
}

using EventPoolTestWithQueue = uur::urQueueTestWithParam<ProviderParams>;

UUR_DEVICE_TEST_SUITE_WITH_PARAM(EventPoolTestWithQueue,
                                 testing::ValuesIn(test_cases),
                                 printParams<EventPoolTest>);

// TODO: actual min version is unknown, retest after drivers on CI are
// updated.
std::tuple<size_t, size_t, size_t> minL0DriverVersion = {1, 6, 31294};

TEST_P(EventPoolTestWithQueue, WithTimestamp) {
  // Skip due to driver bug causing a sigbus
  SKIP_IF_DRIVER_TOO_OLD("Level-Zero", minL0DriverVersion, platform, device);

  if (!(getParam().flags & EVENT_FLAGS_PROFILING_ENABLED)) {
    GTEST_SKIP() << "Profiling needs to be enabled";
  }

  auto zeEvent = createZeEvent(context, device);

  ur_event_handle_t hEvent;
  ASSERT_SUCCESS(urEventCreateWithNativeHandle(
      reinterpret_cast<ur_native_handle_t>(zeEvent.get()), context, nullptr,
      &hEvent));

  ur_device_handle_t hDevice;
  ASSERT_SUCCESS(urQueueGetInfo(queue, UR_QUEUE_INFO_DEVICE, sizeof(device),
                                &hDevice, nullptr));

  ur_event_handle_t first;
  ze_event_handle_t zeFirst;
  {
    ASSERT_SUCCESS(
        urEnqueueTimestampRecordingExp(queue, false, 1, &hEvent, &first));
    zeFirst = first->getZeEvent();

    urEventRelease(first); // should not actually release the event until
                           // recording is completed
  }
  ur_event_handle_t second;
  ze_event_handle_t zeSecond;
  {
    ASSERT_SUCCESS(urEnqueueEventsWaitWithBarrier(queue, 0, nullptr, &second));
    zeSecond = second->getZeEvent();
    ASSERT_SUCCESS(urEventRelease(second));
  }
  ASSERT_NE(first, second);
  ASSERT_NE(zeFirst, zeSecond);

  ASSERT_EQ(zeEventHostSignal(zeEvent.get()), ZE_RESULT_SUCCESS);

  ASSERT_SUCCESS(urQueueFinish(queue));

  // Now, the first event should be avilable for reuse
  ur_event_handle_t third;
  ze_event_handle_t zeThird;
  {
    ASSERT_SUCCESS(urEnqueueEventsWaitWithBarrier(queue, 0, nullptr, &third));
    zeThird = third->getZeEvent();
    ASSERT_SUCCESS(urEventRelease(third));

    ASSERT_FALSE(third->isTimestamped());
  }
  ASSERT_EQ(first, third);
  ASSERT_EQ(zeFirst, zeThird);
}
