// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "command_list_cache.hpp"

#include "level_zero/common.hpp"
#include "level_zero/device.hpp"

#include "context.hpp"
#include "event_pool.hpp"
#include "event_pool_cache.hpp"
#include "event_provider.hpp"
#include "event_provider_counter.hpp"
#include "event_provider_normal.hpp"
#include "uur/fixtures.h"
#include "ze_api.h"

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <unordered_set>

using namespace v2;

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
    const auto device_handle = std::get<0>(info.param);
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

struct EventPoolTest : public uur::urContextTestWithParam<ProviderParams> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam::SetUp());

        auto params = getParam();

        cache = std::unique_ptr<event_pool_cache>(new event_pool_cache(
            MAX_DEVICES,
            [this, params](DeviceId, event_flags_t flags)
                -> std::unique_ptr<event_provider> {
                // normally id would be used to find the appropriate device to create the provider
                switch (params.provider) {
                case TEST_PROVIDER_COUNTER:
                    return std::make_unique<provider_counter>(platform, context,
                                                              device);
                case TEST_PROVIDER_NORMAL:
                    return std::make_unique<provider_normal>(
                        context, device, params.queue, flags);
                }
                return nullptr;
            }));
    }
    void TearDown() override {
        cache.reset();
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam::TearDown());
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

UUR_TEST_SUITE_P(EventPoolTest, testing::ValuesIn(test_cases),
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

            first = pool->allocate(reinterpret_cast<ur_queue_handle_t>(0x1),
                                   UR_COMMAND_KERNEL_LAUNCH);
            zeFirst = first->getZeEvent();

            urEventRelease(first);
        }
        ur_event_handle_t second;
        ze_event_handle_t zeSecond;
        {
            auto pool = cache->borrow(device->Id.value(), getParam().flags);

            second = pool->allocate(reinterpret_cast<ur_queue_handle_t>(0x1),
                                    UR_COMMAND_KERNEL_LAUNCH);
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
                    events.push_back(
                        pool->allocate(reinterpret_cast<ur_queue_handle_t>(0x1),
                                       UR_COMMAND_KERNEL_LAUNCH));
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
        events.push_back(
            pool->allocate(reinterpret_cast<ur_queue_handle_t>(0x1),
                           UR_COMMAND_KERNEL_LAUNCH));
    }
    auto frontZeHandle = events.front()->getZeEvent();
    for (int i = 0; i < 8; ++i) {
        urEventRelease(events.front());
        events.pop_front();
    }
    for (int i = 0; i < 8; ++i) {
        auto e = pool->allocate(reinterpret_cast<ur_queue_handle_t>(0x1),
                                UR_COMMAND_KERNEL_LAUNCH);
        events.push_back(e);
    }

    // the ZeEvent handles from the first provider pool will be reused
    ASSERT_EQ(frontZeHandle, events.back()->getZeEvent());

    for (auto e : events) {
        urEventRelease(e);
    }
}
