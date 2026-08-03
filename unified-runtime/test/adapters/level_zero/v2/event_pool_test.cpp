// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %with-v2 ./event_pool-test
// REQUIRES: v2

// Level Zero V2 Event Pool tests are disabled when using CFI sanitizer
// See https://github.com/oneapi-src/unified-runtime/issues/2324
// UNSUPPORTED: has-cfi-sanitizer

#include "command_list_cache.hpp"

#include "level_zero/common.hpp"
#include "level_zero/common/device.hpp"
#include "level_zero/ur_interface_loader.hpp"

#include "../ze_helpers.hpp"
#include "context.hpp"
#include "event_pool.hpp"
#include "event_pool_cache.hpp"
#include "event_provider.hpp"
#include "event_provider_counter.hpp"
#include "event_provider_normal.hpp"
#include "queue_handle.hpp"
#include "uur/checks.h"
#include "uur/fixtures.h"
#include "ze_api.h"

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <unordered_set>

namespace v2 = ur::level_zero::v2;

static constexpr size_t MAX_DEVICES = 10;

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

static std::string flags_to_str(v2::event_flags_t flags) {
  std::string str;
  if (flags & v2::EVENT_FLAGS_COUNTER) {
    str += "provider_counter";
  } else {
    str += "provider_normal";
  }

  if (flags & v2::EVENT_FLAGS_PROFILING_ENABLED) {
    str += "_profiling";
  } else {
    str += "_no_profiling";
  }

  return str;
}

static const char *queue_to_str(v2::queue_type e) {
  switch (e) {
  case v2::QUEUE_REGULAR:
    return "QUEUE_REGULAR";
  case v2::QUEUE_IMMEDIATE:
    return "QUEUE_IMMEDIATE";
  default:
    return nullptr;
  }
}

struct ProviderParams {
  ProviderType provider;
  v2::event_flags_t flags;
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

    cache = std::unique_ptr<v2::event_pool_cache>(new v2::event_pool_cache(
        nullptr, MAX_DEVICES,
        [this, params](ur::level_zero::DeviceId, v2::event_flags_t flags)
            -> std::unique_ptr<v2::event_provider> {
          // normally id would be used to find the appropriate device to create
          // the provider
          switch (params.provider) {
          case TEST_PROVIDER_COUNTER:
            return std::make_unique<v2::provider_counter>(
                ur::level_zero::common_cast(platform), v2::v2_cast(context),
                params.queue, ur::level_zero::common_cast(device),
                params.flags);
          case TEST_PROVIDER_NORMAL:
            return std::make_unique<v2::provider_normal>(v2::v2_cast(context),
                                                         params.queue, flags);
          }
          return nullptr;
        }));
  }
  void TearDown() override {
    cache.reset();
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTestWithParam::TearDown());
  }

  std::unique_ptr<v2::event_pool_cache> cache;
};

static ProviderParams test_cases[] = {
    {TEST_PROVIDER_NORMAL, 0, v2::QUEUE_REGULAR},
    {TEST_PROVIDER_NORMAL, v2::EVENT_FLAGS_COUNTER, v2::QUEUE_REGULAR},
    {TEST_PROVIDER_NORMAL, v2::EVENT_FLAGS_COUNTER, v2::QUEUE_IMMEDIATE},
    {TEST_PROVIDER_NORMAL,
     v2::EVENT_FLAGS_COUNTER | v2::EVENT_FLAGS_PROFILING_ENABLED,
     v2::QUEUE_IMMEDIATE},
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
  auto deviceId = ur::level_zero::common_cast(device)->Id.value();
  {
    v2::ur_event_handle_t first;
    ze_event_handle_t zeFirst;
    {
      auto pool = cache->borrow(deviceId, getParam().flags);

      first = pool->allocate();
      first->setQueue(nullptr);
      first->setCommandType(UR_COMMAND_KERNEL_LAUNCH);
      zeFirst = first->getZeEvent();

      urEventRelease(v2::v2_cast(first));
    }
    v2::ur_event_handle_t second;
    ze_event_handle_t zeSecond;
    {
      auto pool = cache->borrow(deviceId, getParam().flags);

      second = pool->allocate();
      second->setQueue(nullptr);
      second->setCommandType(UR_COMMAND_KERNEL_LAUNCH);
      zeSecond = second->getZeEvent();

      urEventRelease(v2::v2_cast(second));
    }
    ASSERT_EQ(first, second);
    ASSERT_EQ(zeFirst, zeSecond);
  }
}

TEST_P(EventPoolTest, Threaded) {
  std::vector<std::thread> threads;
  auto deviceId = ur::level_zero::common_cast(device)->Id.value();

  for (int iters = 0; iters < 3; ++iters) {
    for (int th = 0; th < 10; ++th) {
      threads.emplace_back([&] {
        auto pool = cache->borrow(deviceId, getParam().flags);
        std::vector<v2::ur_event_handle_t> events;
        for (int i = 0; i < 100; ++i) {
          events.push_back(pool->allocate());
          events.back()->setQueue(nullptr);
          events.back()->setCommandType(UR_COMMAND_KERNEL_LAUNCH);
        }
        for (int i = 0; i < 100; ++i) {
          urEventRelease(v2::v2_cast(events[i]));
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
  auto deviceId = ur::level_zero::common_cast(device)->Id.value();
  auto pool = cache->borrow(deviceId, getParam().flags);
  std::list<v2::ur_event_handle_t> events;
  for (int i = 0; i < 128; ++i) {
    auto event = pool->allocate();
    event->setQueue(nullptr);
    event->setCommandType(UR_COMMAND_KERNEL_LAUNCH);
    events.push_back(event);
  }
  auto frontZeHandle = events.front()->getZeEvent();
  for (int i = 0; i < 8; ++i) {
    urEventRelease(v2::v2_cast(events.front()));
    events.pop_front();
  }
  for (int i = 0; i < 8; ++i) {
    auto e = pool->allocate();
    e->setQueue(nullptr);
    e->setCommandType(UR_COMMAND_KERNEL_LAUNCH);
    events.push_back(e);
  }

  // the ZeEvent handles from the first provider pool will be reused
  ASSERT_EQ(frontZeHandle, events.back()->getZeEvent());

  for (auto e : events) {
    urEventRelease(v2::v2_cast(e));
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

  if (!(getParam().flags & v2::EVENT_FLAGS_PROFILING_ENABLED)) {
    GTEST_SKIP() << "Profiling needs to be enabled";
  }

  SKIP_IF_BATCHED_QUEUE(queue);
  auto zeEvent = createZeEvent(context, device);

  ur_event_handle_t hEvent;
  ASSERT_SUCCESS(urEventCreateWithNativeHandle(
      reinterpret_cast<ur_native_handle_t>(zeEvent.get()), context, nullptr,
      &hEvent));

  ur_device_handle_t hDevice;
  ASSERT_SUCCESS(urQueueGetInfo(queue, UR_QUEUE_INFO_DEVICE, sizeof(device),
                                &hDevice, nullptr));

  ur_event_handle_t first;
  {
    ASSERT_SUCCESS(
        urEnqueueTimestampRecordingExp(queue, false, 1, &hEvent, &first));
    urEventRelease(first); // should not actually release the event until
                           // recording is completed
  }
  ur_event_handle_t second;
  ASSERT_SUCCESS(urEnqueueEventsWaitWithBarrier(queue, 0, nullptr, &second));
  // even if the event is reused, it should not be timestamped anymore
  ASSERT_FALSE(v2::v2_cast(second)->isTimestamped());
  ASSERT_SUCCESS(urEventRelease(second));

  ASSERT_EQ(zeEventHostSignal(zeEvent.get()), ZE_RESULT_SUCCESS);
  ASSERT_SUCCESS(urQueueFinish(queue));
}
