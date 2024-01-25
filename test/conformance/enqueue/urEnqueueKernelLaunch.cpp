// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urEnqueueKernelLaunchTest : uur::urKernelExecutionTest {
    void SetUp() override {
        program_name = "fill";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
    }

    uint32_t val = 42;
    size_t global_size = 32;
    size_t global_offset = 0;
    size_t n_dimensions = 1;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueKernelLaunchTest);

TEST_P(urEnqueueKernelLaunchTest, Success) {
    ur_mem_handle_t buffer = nullptr;
    AddBuffer1DArg(sizeof(val) * global_size, &buffer);
    AddPodArg(val);
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         &global_offset, &global_size, nullptr,
                                         0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    ValidateBuffer(buffer, sizeof(val) * global_size, val);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(urEnqueueKernelLaunch(nullptr, kernel, n_dimensions,
                                           &global_offset, &global_size,
                                           nullptr, 0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidNullHandleKernel) {
    ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, nullptr, n_dimensions,
                                           &global_offset, &global_size,
                                           nullptr, 0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidNullPtrEventWaitList) {
    ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                           &global_offset, &global_size,
                                           nullptr, 1, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                           &global_offset, &global_size,
                                           nullptr, 0, &validEvent, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                           &global_offset, &global_size,
                                           nullptr, 1, &inv_evt, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
    ASSERT_SUCCESS(urEventRelease(validEvent));
}

TEST_P(urEnqueueKernelLaunchTest, InvalidWorkDimension) {
    uint32_t max_work_item_dimensions = 0;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
        sizeof(max_work_item_dimensions), &max_work_item_dimensions, nullptr));
    ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel,
                                           max_work_item_dimensions + 1,
                                           &global_offset, &global_size,
                                           nullptr, 0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
}

struct urEnqueueKernelLaunch2DTest : uur::urKernelExecutionTest {
    void SetUp() override {
        program_name = "fill_2d";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
    }

    uint32_t val = 42;
    size_t global_size[2] = {8, 8};
    size_t global_offset[2] = {0, 0};
    size_t buffer_size = sizeof(val) * global_size[0] * global_size[1];
    size_t n_dimensions = 2;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueKernelLaunch2DTest);

TEST_P(urEnqueueKernelLaunch2DTest, Success) {
    ur_mem_handle_t buffer = nullptr;
    AddBuffer1DArg(buffer_size, &buffer);
    AddPodArg(val);
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         global_offset, global_size, nullptr, 0,
                                         nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    ValidateBuffer(buffer, buffer_size, val);
}

struct urEnqueueKernelLaunch3DTest : uur::urKernelExecutionTest {
    void SetUp() override {
        program_name = "fill_3d";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
    }

    uint32_t val = 42;
    size_t global_size[3] = {4, 4, 4};
    size_t global_offset[3] = {0, 0, 0};
    size_t buffer_size =
        sizeof(val) * global_size[0] * global_size[1] * global_size[2];
    size_t n_dimensions = 3;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueKernelLaunch3DTest);

TEST_P(urEnqueueKernelLaunch3DTest, Success) {
    ur_mem_handle_t buffer = nullptr;
    AddBuffer1DArg(buffer_size, &buffer);
    AddPodArg(val);
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         global_offset, global_size, nullptr, 0,
                                         nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    ValidateBuffer(buffer, buffer_size, val);
}

struct urEnqueueKernelLaunchWithVirtualMemory : uur::urKernelExecutionTest {

    void SetUp() override {
        program_name = "fill_usm";
        UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTest::SetUp());

        ur_bool_t virtual_memory_support = false;
        ASSERT_SUCCESS(urDeviceGetInfo(
            device, UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT, sizeof(ur_bool_t),
            &virtual_memory_support, nullptr));
        if (!virtual_memory_support) {
            GTEST_SKIP() << "Virtual memory is not supported.";
        }

        ASSERT_SUCCESS(urVirtualMemGranularityGetInfo(
            context, device, UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
            sizeof(granularity), &granularity, nullptr));

        alloc_size = 1024;
        virtual_page_size =
            uur::RoundUpToNearestFactor(alloc_size, granularity);

        ASSERT_SUCCESS(urPhysicalMemCreate(context, device, virtual_page_size,
                                           nullptr, &physical_mem));

        ASSERT_SUCCESS(urVirtualMemReserve(context, nullptr, virtual_page_size,
                                           &virtual_ptr));

        ASSERT_SUCCESS(urVirtualMemMap(context, virtual_ptr, virtual_page_size,
                                       physical_mem, 0,
                                       UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE));

        int pattern = 0;
        ASSERT_SUCCESS(urEnqueueUSMFill(queue, virtual_ptr, sizeof(pattern),
                                        &pattern, virtual_page_size, 0, nullptr,
                                        nullptr));
        ASSERT_SUCCESS(urQueueFinish(queue));
    }

    void TearDown() override {

        if (virtual_ptr) {
            EXPECT_SUCCESS(
                urVirtualMemUnmap(context, virtual_ptr, virtual_page_size));
            EXPECT_SUCCESS(
                urVirtualMemFree(context, virtual_ptr, virtual_page_size));
        }

        if (physical_mem) {
            EXPECT_SUCCESS(urPhysicalMemRelease(physical_mem));
        }

        UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTest::TearDown());
    }

    size_t granularity = 0;
    size_t alloc_size = 0;
    size_t virtual_page_size = 0;
    ur_physical_mem_handle_t physical_mem = nullptr;
    void *virtual_ptr = nullptr;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueKernelLaunchWithVirtualMemory);

TEST_P(urEnqueueKernelLaunchWithVirtualMemory, Success) {
    size_t work_dim = 1;
    size_t global_offset = 0;
    size_t global_size = alloc_size / sizeof(uint32_t);
    uint32_t fill_val = 42;

    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, virtual_ptr));
    ASSERT_SUCCESS(
        urKernelSetArgValue(kernel, 1, sizeof(fill_val), nullptr, &fill_val));

    ur_event_handle_t kernel_evt;
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, work_dim,
                                         &global_offset, &global_size, nullptr,
                                         0, nullptr, &kernel_evt));

    std::vector<uint32_t> data(global_size);
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, data.data(), virtual_ptr,
                                      alloc_size, 1, &kernel_evt, nullptr));

    ASSERT_SUCCESS(urQueueFinish(queue));

    // verify fill worked
    for (size_t i = 0; i < data.size(); i++) {
        ASSERT_EQ(fill_val, data.at(i));
    }
}

struct urEnqueueKernelLaunchMultiDeviceTest : public urEnqueueKernelLaunchTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urEnqueueKernelLaunchTest::SetUp());
        queues.reserve(uur::DevicesEnvironment::instance->devices.size());
        for (const auto &device : uur::DevicesEnvironment::instance->devices) {
            ur_queue_handle_t queue = nullptr;
            ASSERT_SUCCESS(urQueueCreate(this->context, device, 0, &queue));
            queues.push_back(queue);
        }
    }

    void TearDown() override {
        for (const auto &queue : queues) {
            EXPECT_SUCCESS(urQueueRelease(queue));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urEnqueueKernelLaunchTest::TearDown());
    }

    std::vector<ur_queue_handle_t> queues;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueKernelLaunchMultiDeviceTest);

TEST_P(urEnqueueKernelLaunchMultiDeviceTest, KernelLaunchReadDifferentQueues) {
    ur_mem_handle_t buffer = nullptr;
    AddBuffer1DArg(sizeof(val) * global_size, &buffer);
    AddPodArg(val);
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queues[0], kernel, n_dimensions,
                                         &global_offset, &global_size, nullptr,
                                         0, nullptr, nullptr));

    // Wait for the queue to finish executing.
    EXPECT_SUCCESS(urEnqueueEventsWait(queues[0], 0, nullptr, nullptr));

    // Then the remaining queues do blocking reads from the buffer. Since the
    // queues target different devices this checks that any devices memory has
    // been synchronized.
    for (unsigned i = 1; i < queues.size(); ++i) {
        const auto queue = queues[i];
        uint32_t output = 0;
        ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0,
                                              sizeof(output), &output, 0,
                                              nullptr, nullptr));
        ASSERT_EQ(val, output) << "Result on queue " << i << " did not match!";
    }
}
