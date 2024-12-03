// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

enum class BarrierType {
    Normal,
    ExtLowPower,
};

std::ostream &operator<<(std::ostream &os, BarrierType barrierType) {
    switch (barrierType) {
    case BarrierType::Normal:
        os << "Normal";
        break;
    case BarrierType::ExtLowPower:
        os << "ExtLowPower";
        break;
    default:
        os << "Unknown";
        break;
    }
    return os;
}

struct urEnqueueEventsWaitWithBarrierTest
    : uur::urMultiQueueTestWithParam<BarrierType> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urMultiQueueTestWithParam::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_WRITE_ONLY, size,
                                         nullptr, &src_buffer));
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_ONLY, size,
                                         nullptr, &dst_buffer));
        input.assign(count, 42);
        ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue1, src_buffer, true, 0,
                                               size, input.data(), 0, nullptr,
                                               nullptr));
    }

    void TearDown() override {
        if (src_buffer) {
            EXPECT_SUCCESS(urMemRelease(src_buffer));
        }
        if (dst_buffer) {
            EXPECT_SUCCESS(urMemRelease(dst_buffer));
        }
        urMultiQueueTestWithParam::TearDown();
    }

    ur_result_t EnqueueBarrier(ur_queue_handle_t queue, uint32_t num_events,
                               const ur_event_handle_t *event_list,
                               ur_event_handle_t *wait_event) {
        BarrierType barrier = getParam();
        if (barrier == BarrierType::ExtLowPower) {
            struct ur_exp_enqueue_ext_properties_t props = {
                UR_STRUCTURE_TYPE_EXP_ENQUEUE_EXT_PROPERTIES, nullptr,
                UR_EXP_ENQUEUE_EXT_FLAG_LOW_POWER_EVENTS};
            return urEnqueueEventsWaitWithBarrierExt(queue, &props, num_events,
                                                     event_list, wait_event);
        }

        return urEnqueueEventsWaitWithBarrier(queue, num_events, event_list,
                                              wait_event);
    }

    const size_t count = 1024;
    const size_t size = sizeof(uint32_t) * count;
    ur_mem_handle_t src_buffer = nullptr;
    ur_mem_handle_t dst_buffer = nullptr;
    std::vector<uint32_t> input;
};

UUR_TEST_SUITE_P(urEnqueueEventsWaitWithBarrierTest,
                 ::testing::Values(BarrierType::Normal,
                                   BarrierType::ExtLowPower),
                 uur::deviceTestWithParamPrinter<BarrierType>);

struct urEnqueueEventsWaitWithBarrierOrderingTest : uur::urProgramTest {
    void SetUp() override {
        program_name = "sequence";
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));

        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                         sizeof(uint32_t), nullptr, &buffer));

        auto entry_points =
            uur::KernelsEnvironment::instance->GetEntryPointNames(program_name);
        std::cout << entry_points[0];

        ASSERT_SUCCESS(urKernelCreate(program, "_ZTS3Add", &add_kernel));
        ASSERT_SUCCESS(urKernelCreate(program, "_ZTS3Mul", &mul_kernel));
    }

    void TearDown() override { uur::urProgramTest::TearDown(); }

    ur_kernel_handle_t add_kernel;
    ur_kernel_handle_t mul_kernel;
    ur_mem_handle_t buffer = nullptr;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueEventsWaitWithBarrierOrderingTest);

TEST_P(urEnqueueEventsWaitWithBarrierTest, Success) {
    ur_event_handle_t event1 = nullptr;
    ur_event_handle_t waitEvent = nullptr;
    ASSERT_SUCCESS(urEnqueueMemBufferCopy(queue1, src_buffer, dst_buffer, 0, 0,
                                          size, 0, nullptr, &event1));
    EXPECT_SUCCESS(EnqueueBarrier(queue2, 1, &event1, &waitEvent));
    EXPECT_SUCCESS(urQueueFlush(queue2));
    EXPECT_SUCCESS(urQueueFlush(queue1));
    EXPECT_SUCCESS(urEventWait(1, &waitEvent));

    std::vector<uint32_t> output(count, 1);
    EXPECT_SUCCESS(urEnqueueMemBufferRead(queue1, dst_buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    EXPECT_EQ(input, output);
    EXPECT_SUCCESS(urEventRelease(waitEvent));
    EXPECT_SUCCESS(urEventRelease(event1));

    ur_event_handle_t event2 = nullptr;
    input.assign(count, 420);
    EXPECT_SUCCESS(urEnqueueMemBufferWrite(queue2, src_buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));
    EXPECT_SUCCESS(urEnqueueMemBufferCopy(queue2, src_buffer, dst_buffer, 0, 0,
                                          size, 0, nullptr, &event2));
    EXPECT_SUCCESS(EnqueueBarrier(queue1, 1, &event2, &waitEvent));
    EXPECT_SUCCESS(urQueueFlush(queue2));
    EXPECT_SUCCESS(urQueueFlush(queue1));
    EXPECT_SUCCESS(urEventWait(1, &waitEvent));
    EXPECT_SUCCESS(urEnqueueMemBufferRead(queue2, dst_buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    EXPECT_SUCCESS(urEventRelease(waitEvent));
    EXPECT_SUCCESS(urEventRelease(event2));
    EXPECT_EQ(input, output);
}

TEST_P(urEnqueueEventsWaitWithBarrierTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     EnqueueBarrier(nullptr, 0, nullptr, nullptr));
}

TEST_P(urEnqueueEventsWaitWithBarrierTest, InvalidNullPtrEventWaitList) {
    ASSERT_EQ_RESULT(EnqueueBarrier(queue1, 1, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue1, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(EnqueueBarrier(queue1, 0, &validEvent, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(EnqueueBarrier(queue1, 1, &inv_evt, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_SUCCESS(urEventRelease(validEvent));
}

TEST_P(urEnqueueEventsWaitWithBarrierOrderingTest,
       SuccessEventDependenciesBarrierOnly) {
    constexpr size_t offset = 0;
    constexpr size_t count = 1;
    ur_event_handle_t event;

    uur::KernelLaunchHelper addHelper(platform, context, add_kernel, queue);
    uur::KernelLaunchHelper mulHelper(platform, context, mul_kernel, queue);

    addHelper.SetBuffer1DArg(buffer, nullptr);
    mulHelper.SetBuffer1DArg(buffer, nullptr);

    for (size_t i = 0; i < 10; i++) {
        constexpr uint32_t ONE = 1;
        urEnqueueMemBufferWrite(queue, buffer, true, 0, sizeof(uint32_t), &ONE,
                                0, nullptr, &event);
        EXPECT_SUCCESS(
            urEnqueueEventsWaitWithBarrier(queue, 1, &event, nullptr));
        EXPECT_SUCCESS(urEnqueueKernelLaunch(queue, add_kernel, 1, &offset,
                                             &count, nullptr, 0, nullptr,
                                             &event));
        EXPECT_SUCCESS(
            urEnqueueEventsWaitWithBarrier(queue, 1, &event, nullptr));
        EXPECT_SUCCESS(urEnqueueKernelLaunch(queue, mul_kernel, 1, &offset,
                                             &count, nullptr, 0, nullptr,
                                             &event));
        EXPECT_SUCCESS(
            urEnqueueEventsWaitWithBarrier(queue, 1, &event, nullptr));
        addHelper.ValidateBuffer(buffer, sizeof(uint32_t), 4004);
    }
}

TEST_P(urEnqueueEventsWaitWithBarrierOrderingTest,
       SuccessEventDependenciesLaunchOnly) {
    constexpr size_t offset = 0;
    constexpr size_t count = 1;
    ur_event_handle_t event;

    uur::KernelLaunchHelper addHelper(platform, context, add_kernel, queue);
    uur::KernelLaunchHelper mulHelper(platform, context, mul_kernel, queue);

    addHelper.SetBuffer1DArg(buffer, nullptr);
    mulHelper.SetBuffer1DArg(buffer, nullptr);

    for (size_t i = 0; i < 10; i++) {
        constexpr uint32_t ONE = 1;
        urEnqueueMemBufferWrite(queue, buffer, true, 0, sizeof(uint32_t), &ONE,
                                0, nullptr, nullptr);
        EXPECT_SUCCESS(
            urEnqueueEventsWaitWithBarrier(queue, 0, nullptr, &event));
        EXPECT_SUCCESS(urEnqueueKernelLaunch(queue, add_kernel, 1, &offset,
                                             &count, nullptr, 1, &event,
                                             nullptr));
        EXPECT_SUCCESS(
            urEnqueueEventsWaitWithBarrier(queue, 0, nullptr, &event));
        EXPECT_SUCCESS(urEnqueueKernelLaunch(queue, mul_kernel, 1, &offset,
                                             &count, nullptr, 1, &event,
                                             nullptr));
        EXPECT_SUCCESS(
            urEnqueueEventsWaitWithBarrier(queue, 0, nullptr, &event));
        addHelper.ValidateBuffer(buffer, sizeof(uint32_t), 4004);
    }
}

TEST_P(urEnqueueEventsWaitWithBarrierOrderingTest, SuccessEventDependencies) {
    constexpr size_t offset = 0;
    constexpr size_t count = 1;
    ur_event_handle_t event[6];

    uur::KernelLaunchHelper addHelper(platform, context, add_kernel, queue);
    uur::KernelLaunchHelper mulHelper(platform, context, mul_kernel, queue);

    addHelper.SetBuffer1DArg(buffer, nullptr);
    mulHelper.SetBuffer1DArg(buffer, nullptr);

    for (size_t i = 0; i < 10; i++) {
        constexpr uint32_t ONE = 1;
        urEnqueueMemBufferWrite(queue, buffer, true, 0, sizeof(uint32_t), &ONE,
                                0, nullptr, &event[0]);
        EXPECT_SUCCESS(
            urEnqueueEventsWaitWithBarrier(queue, 1, &event[0], &event[1]));
        EXPECT_SUCCESS(urEnqueueKernelLaunch(queue, add_kernel, 1, &offset,
                                             &count, nullptr, 1, &event[1],
                                             &event[2]));
        EXPECT_SUCCESS(
            urEnqueueEventsWaitWithBarrier(queue, 1, &event[2], &event[3]));
        EXPECT_SUCCESS(urEnqueueKernelLaunch(queue, mul_kernel, 1, &offset,
                                             &count, nullptr, 1, &event[3],
                                             &event[4]));
        EXPECT_SUCCESS(
            urEnqueueEventsWaitWithBarrier(queue, 1, &event[4], &event[5]));
        addHelper.ValidateBuffer(buffer, sizeof(uint32_t), 4004);
    }
}

TEST_P(urEnqueueEventsWaitWithBarrierOrderingTest,
       SuccessNonEventDependencies) {
    constexpr size_t offset = 0;
    constexpr size_t count = 1;

    uur::KernelLaunchHelper addHelper(platform, context, add_kernel, queue);
    uur::KernelLaunchHelper mulHelper(platform, context, mul_kernel, queue);

    addHelper.SetBuffer1DArg(buffer, nullptr);
    mulHelper.SetBuffer1DArg(buffer, nullptr);

    for (size_t i = 0; i < 10; i++) {
        constexpr uint32_t ONE = 1;
        urEnqueueMemBufferWrite(queue, buffer, true, 0, sizeof(uint32_t), &ONE,
                                0, nullptr, nullptr);
        EXPECT_SUCCESS(
            urEnqueueEventsWaitWithBarrier(queue, 0, nullptr, nullptr));
        EXPECT_SUCCESS(urEnqueueKernelLaunch(queue, add_kernel, 1, &offset,
                                             &count, nullptr, 0, nullptr,
                                             nullptr));
        EXPECT_SUCCESS(
            urEnqueueEventsWaitWithBarrier(queue, 0, nullptr, nullptr));
        EXPECT_SUCCESS(urEnqueueKernelLaunch(queue, mul_kernel, 1, &offset,
                                             &count, nullptr, 0, nullptr,
                                             nullptr));
        EXPECT_SUCCESS(
            urEnqueueEventsWaitWithBarrier(queue, 0, nullptr, nullptr));
        addHelper.ValidateBuffer(buffer, sizeof(uint32_t), 4004);
    }
}
