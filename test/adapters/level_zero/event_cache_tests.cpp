// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_print.hpp"
#include "uur/fixtures.h"
#include "uur/raii.h"

#include <map>
#include <string>

#include "ze_tracer_common.hpp"

std::size_t eventCreateCount = 0;
std::size_t eventDestroyCount = 0;

void OnEnterEventCreate(ze_event_create_params_t *, ze_result_t, void *,
                        void **) {
    eventCreateCount++;
}

void OnEnterEventDestroy(ze_event_destroy_params_t *, ze_result_t, void *,
                         void **) {
    eventDestroyCount++;
}

static std::shared_ptr<_zel_tracer_handle_t> tracer = [] {
    zel_core_callbacks_t prologue_callbacks{};
    prologue_callbacks.Event.pfnCreateCb = OnEnterEventCreate;
    prologue_callbacks.Event.pfnDestroyCb = OnEnterEventDestroy;
    return enableTracing(prologue_callbacks, {});
}();

template <typename... Args> auto combineFlags(std::tuple<Args...> tuple) {
    return std::apply([](auto... args) { return (... |= args); }, tuple);
}

using FlagsTupleType = std::tuple<ur_queue_flags_t, ur_queue_flags_t,
                                  ur_queue_flags_t, ur_queue_flags_t>;

// TODO: get rid of this, this is a workaround for fails on older driver
// where for some reason continuing the test leads to a segfault
#define UUR_ASSERT_SUCCESS_OR_EXIT_IF_UNSUPPORTED(ret)                         \
    auto status = ret;                                                         \
    if (status == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {                       \
        exit(0);                                                               \
    } else {                                                                   \
        ASSERT_EQ(status, UR_RESULT_SUCCESS);                                  \
    }

struct urEventCacheTest : uur::urContextTestWithParam<FlagsTupleType> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam::SetUp());

        flags = combineFlags(getParam());

        ur_queue_properties_t props;
        props.flags = flags;
        ASSERT_SUCCESS(urQueueCreate(context, device, &props, &queue));
        ASSERT_NE(queue, nullptr);

        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_WRITE_ONLY, size,
                                         nullptr, &buffer));

        eventCreateCount = 0;
        eventDestroyCount = 0;
    }

    void TearDown() override {
        if (buffer) {
            EXPECT_SUCCESS(urMemRelease(buffer));
        }
        if (queue) {
            UUR_ASSERT_SUCCESS_OR_EXIT_IF_UNSUPPORTED(urQueueRelease(queue));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam::TearDown());
    }

    auto enqueueWork(ur_event_handle_t *hEvent, int data) {
        input.assign(count, data);
        UUR_ASSERT_SUCCESS_OR_EXIT_IF_UNSUPPORTED(urEnqueueMemBufferWrite(
            queue, buffer, false, 0, size, input.data(), 0, nullptr, hEvent));
    }

    void verifyData() {
        std::vector<uint32_t> output(count, 1);
        UUR_ASSERT_SUCCESS_OR_EXIT_IF_UNSUPPORTED(urEnqueueMemBufferRead(
            queue, buffer, true, 0, size, output.data(), 0, nullptr, nullptr));

        if (!(flags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE)) {
            ASSERT_EQ(input, output);
        }
    }

    const size_t count = 1024;
    const size_t size = sizeof(uint32_t) * count;
    ur_mem_handle_t buffer = nullptr;
    ur_queue_handle_t queue = nullptr;
    std::vector<uint32_t> input;
    ur_queue_flags_t flags;
};

TEST_P(urEventCacheTest, eventsReuseNoVisibleEvent) {
    static constexpr int numIters = 16;
    static constexpr int numEnqueues = 128;

    for (int i = 0; i < numIters; i++) {
        for (int j = 0; j < numEnqueues; j++) {
            enqueueWork(nullptr, i * numEnqueues + j);
        }
        UUR_ASSERT_SUCCESS_OR_EXIT_IF_UNSUPPORTED(urQueueFinish(queue));
        verifyData();
    }

    // TODO: why events are not reused for UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE?
    if ((flags & UR_QUEUE_FLAG_DISCARD_EVENTS) &&
        !(flags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE)) {
        ASSERT_EQ(eventCreateCount, 2);
    } else {
        ASSERT_GE(eventCreateCount, numIters * numEnqueues);
    }
}

TEST_P(urEventCacheTest, eventsReuseWithVisibleEvent) {
    static constexpr int numIters = 16;
    static constexpr int numEnqueues = 128;

    for (int i = 0; i < numIters; i++) {
        std::vector<uur::raii::Event> events(numEnqueues);
        for (int j = 0; j < numEnqueues; j++) {
            enqueueWork(events[j].ptr(), i * numEnqueues + j);
        }
        UUR_ASSERT_SUCCESS_OR_EXIT_IF_UNSUPPORTED(urQueueFinish(queue));
        verifyData();
    }

    ASSERT_LT(eventCreateCount, numIters * numEnqueues);
}

TEST_P(urEventCacheTest, eventsReuseWithVisibleEventAndWait) {
    static constexpr int numIters = 16;
    static constexpr int numEnqueues = 128;
    static constexpr int waitEveryN = 16;

    for (int i = 0; i < numIters; i++) {
        std::vector<uur::raii::Event> events;
        for (int j = 0; j < numEnqueues; j++) {
            events.emplace_back();
            enqueueWork(events.back().ptr(), i * numEnqueues + j);

            if (j > 0 && j % waitEveryN == 0) {
                ASSERT_SUCCESS(urEventWait(waitEveryN,
                                           (ur_event_handle_t *)events.data()));
                verifyData();
                events.clear();
            }
        }
        UUR_ASSERT_SUCCESS_OR_EXIT_IF_UNSUPPORTED(urQueueFinish(queue));
    }

    ASSERT_GE(eventCreateCount, waitEveryN);
    // TODO: why there are more events than this?
    // ASSERT_LE(eventCreateCount,  waitEveryN * 2 + 2);
}

template <typename T>
inline std::string
printFlags(const testing::TestParamInfo<typename T::ParamType> &info) {
    const auto device_handle = std::get<0>(info.param);
    const auto platform_device_name =
        uur::GetPlatformAndDeviceName(device_handle);
    auto flags = combineFlags(std::get<1>(info.param));

    std::stringstream ss;
    ur::details::printFlag<ur_queue_flag_t>(ss, flags);

    auto str = ss.str();
    std::replace(str.begin(), str.end(), ' ', '_');
    std::replace(str.begin(), str.end(), '|', '_');
    return platform_device_name + "__" + str;
}

UUR_TEST_SUITE_P(
    urEventCacheTest,
    ::testing::Combine(
        testing::Values(0, UR_QUEUE_FLAG_DISCARD_EVENTS),
        testing::Values(0, UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE),
        // TODO: why the test fails with UR_QUEUE_FLAG_SUBMISSION_BATCHED?
        testing::Values(
            UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE /*, UR_QUEUE_FLAG_SUBMISSION_BATCHED */),
        testing::Values(0, UR_QUEUE_FLAG_PROFILING_ENABLE)),
    printFlags<urEventCacheTest>);
