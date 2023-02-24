// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

struct urEnqueueUSMMemsetTest : uur::urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());

        bool device_usm{false};
        ASSERT_SUCCESS(
            urDeviceGetInfo(device, UR_DEVICE_INFO_USM_DEVICE_SUPPORT,
                            sizeof(bool), &device_usm, nullptr));

        if (!device_usm) {
            GTEST_SKIP() << "Device USM is not supported";
        }

        ur_usm_mem_flags_t flags{};
        ASSERT_SUCCESS(
            urUSMDeviceAlloc(context, device, &flags, allocation_size, 0,
                             reinterpret_cast<void **>(&device_mem)));
    }

    void TearDown() override {
        if (device_mem) {
            EXPECT_SUCCESS(urUSMFree(context, device_mem));
        }

        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
    }

    void verifyData() {
        ASSERT_SUCCESS(
            urEnqueueUSMMemcpy(queue, true, host_mem.data(), device_mem,
                               allocation_size, 0, nullptr, nullptr));
        bool good = std::all_of(host_mem.begin(), host_mem.end(),
                                [this](char i) {
                                    return i == static_cast<char>(memset_value);
                                });
        ASSERT_TRUE(good);
    }

    const uint32_t allocation_size = 1024;
    const int memset_value = 12;
    std::vector<char> host_mem = std::vector<char>(allocation_size);
    int *device_mem{nullptr};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueUSMMemsetTest);

TEST_P(urEnqueueUSMMemsetTest, Success) {

    ur_event_handle_t event = nullptr;
    ASSERT_SUCCESS(
        urEnqueueUSMMemset(queue, device_mem, memset_value, sizeof(int), 0, nullptr,
                           &event));

    EXPECT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(urEventWait(1, &event));
    EXPECT_SUCCESS(urEventRelease(event));

    ASSERT_NO_FATAL_FAILURE(verifyData());
}

TEST_P(urEnqueueUSMMemsetTest, InvalidNullQueueHandle) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueUSMMemset(nullptr, device_mem, memset_value, sizeof(int), 0,
                                        nullptr, nullptr));
}

TEST_P(urEnqueueUSMMemsetTest, InvalidNullPtr) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueUSMMemset(queue, nullptr, memset_value, sizeof(int), 0,
                                        nullptr, nullptr));
}

TEST_P(urEnqueueUSMMemsetTest, InvalidNullPtrEventWaitList) {

    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
        urEnqueueUSMMemset(queue, device_mem, memset_value, sizeof(int), 1, nullptr,
                           nullptr));

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueUSMMemset(queue, device_mem, memset_value, sizeof(int), 0,
                                        &validEvent, nullptr));
}

TEST_P(urEnqueueUSMMemsetTest, InvalidCount) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_SIZE,
        urEnqueueUSMMemset(queue, device_mem, memset_value, 0, 0, nullptr,
                           nullptr));

    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_SIZE,
        urEnqueueUSMMemset(queue, device_mem, memset_value, allocation_size + 1,
                           0, nullptr, nullptr));
}

TEST_P(urEnqueueUSMMemsetTest, InvalidMemObject) {
    // Random pointer which is not a usm allocation
    intptr_t address = 0xDEADBEEF;
    int *ptr = reinterpret_cast<int *>(address);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_MEM_OBJECT,
                     urEnqueueUSMMemset(queue, ptr, memset_value, sizeof(int), 0, nullptr,
                                        nullptr));
}
