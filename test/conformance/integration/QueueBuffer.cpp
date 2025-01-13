// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include <chrono>
#include <thread>
#include <uur/known_failure.h>

struct QueueBufferTestWithParam : uur::IntegrationQueueTestWithParam {
    void SetUp() override {
        UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{},
                             uur::NativeCPU{});

        program_name = "cpy_and_mult";
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::IntegrationQueueTestWithParam::SetUp());
    }

    void TearDown() override { uur::IntegrationQueueTestWithParam::TearDown(); }

    void verifyResults(ur_mem_handle_t Buffer, uint32_t ExpectedValue) {
        uint32_t HostMem[ArraySize] = {};
        ASSERT_SUCCESS(urEnqueueMemBufferRead(Queue, Buffer, true, 0,
                                              sizeof(uint32_t) * ArraySize,
                                              HostMem, 0, nullptr, nullptr));

        for (uint32_t i : HostMem) {
            ASSERT_EQ(i, ExpectedValue);
        }
    }

    ur_mem_handle_t Buffer1 = nullptr;
    ur_mem_handle_t Buffer2 = nullptr;
};

UUR_DEVICE_TEST_SUITE_P(
    QueueBufferTestWithParam,
    testing::Values(0, /* In-Order */
                    UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE),
    uur::IntegrationQueueTestWithParam::paramPrinter);

/* Submits multiple kernels that interact with each other by accessing and
 * writing to the same buffers.
 * Checks that when using an IN_ORDER queue, no synchronization is needed
 * between calls to urEnqueueKernelLaunch.
 * Checks that when using an OUT_OF_ORDER queue, synchronizing using only
 * event barriers is enough. */
TEST_P(QueueBufferTestWithParam, QueueBufferTest) {

    std::vector<ur_event_handle_t> EventsFill;
    ur_event_handle_t Event;

    size_t Buffer1Index;
    size_t Buffer2Index;
    ASSERT_NO_FATAL_FAILURE(
        AddBuffer1DArg(ArraySize * sizeof(uint32_t), &Buffer1, &Buffer1Index));
    ASSERT_NO_FATAL_FAILURE(
        AddBuffer1DArg(ArraySize * sizeof(uint32_t), &Buffer2, &Buffer2Index));

    ASSERT_SUCCESS(urEnqueueMemBufferFill(
        Queue, Buffer1, &InitialValue, sizeof(uint32_t), 0,
        ArraySize * sizeof(uint32_t), 0, nullptr, &Event));
    EventsFill.push_back(Event);

    ASSERT_SUCCESS(urEnqueueMemBufferFill(
        Queue, Buffer2, &InitialValue, sizeof(uint32_t), 0,
        ArraySize * sizeof(uint32_t), 0, nullptr, &Event));
    EventsFill.push_back(Event);

    ASSERT_NO_FATAL_FAILURE(submitBarrierIfNeeded(EventsFill));

    constexpr size_t GlobalOffset = 0;
    constexpr size_t NDimensions = 1;
    constexpr uint32_t NumIterations = 5;

    uint32_t CurValueMem1 = InitialValue;
    uint32_t CurValueMem2 = InitialValue;
    for (uint32_t i = 0; i < NumIterations; ++i) {

        /* Copy from DeviceMem1 to DeviceMem2 and multiply by 2 */
        ASSERT_SUCCESS(
            urKernelSetArgMemObj(kernel, Buffer2Index, nullptr, Buffer2));
        ASSERT_SUCCESS(
            urKernelSetArgMemObj(kernel, Buffer1Index, nullptr, Buffer1));

        ASSERT_SUCCESS(urEnqueueKernelLaunch(Queue, kernel, NDimensions,
                                             &GlobalOffset, &ArraySize, nullptr,
                                             0, nullptr, &Event));
        ASSERT_NO_FATAL_FAILURE(submitBarrierIfNeeded(Event));

        CurValueMem2 = CurValueMem1 * 2;

        /* Copy from DeviceMem1 to DeviceMem2 and multiply by 2 */
        ASSERT_SUCCESS(
            urKernelSetArgMemObj(kernel, Buffer1Index, nullptr, Buffer2));
        ASSERT_SUCCESS(
            urKernelSetArgMemObj(kernel, Buffer2Index, nullptr, Buffer1));

        ASSERT_SUCCESS(urEnqueueKernelLaunch(Queue, kernel, NDimensions,
                                             &GlobalOffset, &ArraySize, nullptr,
                                             0, nullptr, &Event));
        ASSERT_NO_FATAL_FAILURE(submitBarrierIfNeeded(Event));

        CurValueMem1 = CurValueMem2 * 2;
    }

    ASSERT_SUCCESS(urQueueFinish(Queue));

    ASSERT_NO_FATAL_FAILURE(verifyResults(Buffer1, CurValueMem1));
    ASSERT_NO_FATAL_FAILURE(verifyResults(Buffer2, CurValueMem2));
}
