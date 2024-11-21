// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <vector>

struct urEnqueueUSMMemcpyTest : uur::urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());

        ur_device_usm_access_capability_flags_t device_usm = 0;
        ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, device_usm));
        if (!device_usm) {
            GTEST_SKIP() << "Device USM is not supported";
        }

        ASSERT_SUCCESS(
            urUSMDeviceAlloc(context, device, nullptr, nullptr, allocation_size,
                             reinterpret_cast<void **>(&device_src)));
        ASSERT_SUCCESS(
            urUSMDeviceAlloc(context, device, nullptr, nullptr, allocation_size,
                             reinterpret_cast<void **>(&device_dst)));

        ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_src, sizeof(memset_value),
                                        &memset_value, allocation_size, 0,
                                        nullptr, &memset_event));
        ASSERT_SUCCESS(urQueueFlush(queue));
    }

    void TearDown() override {
        if (memset_event) {
            EXPECT_SUCCESS(urEventRelease(memset_event));
        }
        if (device_src) {
            EXPECT_SUCCESS(urUSMFree(context, device_src));
        }
        if (device_dst) {
            EXPECT_SUCCESS(urUSMFree(context, device_dst));
        }

        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
    }

    bool memsetHasFinished() {
        ur_event_status_t memset_event_status;
        EXPECT_SUCCESS(urEventGetInfo(
            memset_event, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
            sizeof(ur_event_status_t), &memset_event_status, nullptr));
        return UR_EVENT_STATUS_COMPLETE == memset_event_status;
    }

    void verifyData() {
        ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, host_mem.data(),
                                          device_dst, allocation_size, 0,
                                          nullptr, nullptr));
        bool good = std::all_of(host_mem.begin(), host_mem.end(),
                                [](uint8_t i) { return i == memset_value; });
        ASSERT_TRUE(good);
    }

    static constexpr uint32_t num_elements = 1024;
    static constexpr uint8_t memset_value = 12;
    static constexpr uint32_t allocation_size = sizeof(uint8_t) * num_elements;
    std::vector<uint8_t> host_mem = std::vector<uint8_t>(num_elements);

    ur_event_handle_t memset_event = nullptr;
    int *device_src{nullptr};
    int *device_dst{nullptr};
};

/**
 * Test that urEnqueueUSMMemcpy blocks when the blocking parameter is set to
 * true.
 */
TEST_P(urEnqueueUSMMemcpyTest, Blocking) {
    ASSERT_SUCCESS(urEventWait(1, &memset_event));
    ASSERT_TRUE(memsetHasFinished());
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, device_dst, device_src,
                                      allocation_size, 0, nullptr, nullptr));
    ASSERT_NO_FATAL_FAILURE(verifyData());
}

/**
 * Test that urEnqueueUSMMemcpy blocks and returns an event with
 * UR_EVENT_STATUS_COMPLETE when the blocking parameter is set to true.
 */
TEST_P(urEnqueueUSMMemcpyTest, BlockingWithEvent) {
    ur_event_handle_t memcpy_event = nullptr;
    ASSERT_SUCCESS(urEventWait(1, &memset_event));
    ASSERT_TRUE(memsetHasFinished());
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, device_dst, device_src,
                                      allocation_size, 0, nullptr,
                                      &memcpy_event));

    ur_event_status_t event_status;
    ASSERT_SUCCESS(
        urEventGetInfo(memcpy_event, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                       sizeof(ur_event_status_t), &event_status, nullptr));
    ASSERT_EQ(event_status, UR_EVENT_STATUS_COMPLETE);
    EXPECT_SUCCESS(urEventRelease(memcpy_event));
    ASSERT_NO_FATAL_FAILURE(verifyData());
}

/**
 * Test that the memory copy happens when the blocking flag is set to false and
 * the application waits for the returned event to complete.
 */
TEST_P(urEnqueueUSMMemcpyTest, NonBlocking) {
    ASSERT_SUCCESS(urEventWait(1, &memset_event));
    ASSERT_TRUE(memsetHasFinished());
    ur_event_handle_t memcpy_event = nullptr;
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, device_dst, device_src,
                                      allocation_size, 0, nullptr,
                                      &memcpy_event));
    ASSERT_SUCCESS(urEventWait(1, &memcpy_event));
    ASSERT_SUCCESS(urEventRelease(memcpy_event));

    ASSERT_NO_FATAL_FAILURE(verifyData());
}

/**
 * Test that urEnqueueUSMMemcpy waits for the events dependencies before copying
 * the memory.
 */
TEST_P(urEnqueueUSMMemcpyTest, WaitForDependencies) {
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, device_dst, device_src,
                                      allocation_size, 1, &memset_event,
                                      nullptr));
    ASSERT_TRUE(memsetHasFinished());
    ASSERT_NO_FATAL_FAILURE(verifyData());
}

TEST_P(urEnqueueUSMMemcpyTest, InvalidNullQueueHandle) {
    ASSERT_EQ_RESULT(urEnqueueUSMMemcpy(nullptr, true, device_dst, device_src,
                                        allocation_size, 0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueUSMMemcpyTest, InvalidNullDst) {
    ASSERT_EQ_RESULT(urEnqueueUSMMemcpy(queue, true, nullptr, device_src,
                                        allocation_size, 0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEnqueueUSMMemcpyTest, InvalidNullSrc) {
    ASSERT_EQ_RESULT(urEnqueueUSMMemcpy(queue, true, device_dst, nullptr,
                                        allocation_size, 0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEnqueueUSMMemcpyTest, InvalidNullPtrEventWaitList) {
    ASSERT_EQ_RESULT(urEnqueueUSMMemcpy(queue, true, device_dst, device_src,
                                        allocation_size, 1, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_EQ_RESULT(urEnqueueUSMMemcpy(queue, true, device_dst, device_src,
                                        allocation_size, 0, &memset_event,
                                        nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueUSMMemcpy(queue, true, device_dst, device_src,
                                        allocation_size, 1, &inv_evt, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
}

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueUSMMemcpyTest);

struct urEnqueueUSMMemcpyMultiDeviceTest : uur::urAllDevicesTest {
    void SetUp() override {
        uur::urAllDevicesTest::SetUp();
        for (auto &device : devices) {
            ur_device_usm_access_capability_flags_t device_usm = 0;
            ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, device_usm));
            if (device_usm) {
                usm_devices.push_back(device);
                if (usm_devices.size() == 2) {
                    break;
                }
            }
        }

        if (usm_devices.size() < 2) {
            GTEST_SKIP() << "Not enough devices in platform with USM support";
        }

        ASSERT_SUCCESS(urContextCreate(usm_devices.size(), usm_devices.data(),
                                       nullptr, &context));
        ASSERT_SUCCESS(
            urQueueCreate(context, usm_devices[0], nullptr, &src_queue));
        ASSERT_SUCCESS(
            urQueueCreate(context, usm_devices[1], nullptr, &dst_queue));

        ASSERT_SUCCESS(
            urUSMHostAlloc(context, nullptr, nullptr, alloc_size, &host_alloc));
        ASSERT_SUCCESS(urUSMDeviceAlloc(context, usm_devices[0], nullptr,
                                        nullptr, alloc_size, &src_alloc));
        ASSERT_SUCCESS(urUSMDeviceAlloc(context, usm_devices[1], nullptr,
                                        nullptr, alloc_size, &dst_alloc));

        ASSERT_SUCCESS(urEnqueueUSMFill(src_queue, src_alloc,
                                        sizeof(fill_pattern), &fill_pattern,
                                        alloc_size, 0, nullptr, nullptr));
        ASSERT_SUCCESS(urQueueFinish(src_queue));
    }

    void TearDown() override {
        if (src_alloc) {
            ASSERT_SUCCESS(urUSMFree(context, src_alloc));
        }
        if (dst_alloc) {
            ASSERT_SUCCESS(urUSMFree(context, dst_alloc));
        }
        if (host_alloc) {
            ASSERT_SUCCESS(urUSMFree(context, host_alloc));
        }
        if (src_queue) {
            ASSERT_SUCCESS(urQueueRelease(src_queue));
        }
        if (dst_queue) {
            ASSERT_SUCCESS(urQueueRelease(dst_queue));
        }
        if (context) {
            ASSERT_SUCCESS(urContextRelease(context));
        }
        uur::urAllDevicesTest::TearDown();
    }

    void verifyData() {
        for (size_t i = 0; i < alloc_size; i++) {
            EXPECT_EQ(static_cast<uint8_t *>(host_alloc)[i], fill_pattern);
        }
    }

    std::vector<ur_device_handle_t> usm_devices;
    ur_context_handle_t context = nullptr;
    ur_queue_handle_t src_queue = nullptr;
    ur_queue_handle_t dst_queue = nullptr;
    void *src_alloc = nullptr;
    void *dst_alloc = nullptr;
    void *host_alloc = nullptr;
    size_t alloc_size = 64;
    uint8_t fill_pattern = 42;
};
UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urEnqueueUSMMemcpyMultiDeviceTest);

TEST_P(urEnqueueUSMMemcpyMultiDeviceTest, DeviceToDeviceCopyBlocking) {
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(src_queue, true, dst_alloc, src_alloc,
                                      alloc_size, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(dst_queue, true, host_alloc, dst_alloc,
                                      alloc_size, 0, nullptr, nullptr));
    verifyData();
}

TEST_P(urEnqueueUSMMemcpyMultiDeviceTest, DeviceToDeviceCopyNonBlocking) {
    ur_event_handle_t device_copy_event = nullptr;
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(src_queue, false, dst_alloc, src_alloc,
                                      alloc_size, 0, nullptr,
                                      &device_copy_event));
    ASSERT_SUCCESS(urQueueFlush(src_queue));
    ASSERT_SUCCESS(urEventWait(1, &device_copy_event));
    ASSERT_SUCCESS(urEventRelease(device_copy_event));
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(dst_queue, true, host_alloc, dst_alloc,
                                      alloc_size, 0, nullptr, nullptr));
    verifyData();
}
