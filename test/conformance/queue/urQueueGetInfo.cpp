// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urQueueGetInfoTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueGetInfoTest);

TEST_P(urQueueGetInfoTest, Context) {
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

    size_t size = 0;
    auto infoType = UR_QUEUE_INFO_CONTEXT;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urQueueGetInfo(queue, infoType, 0, nullptr, &size), infoType);
    ASSERT_EQ(sizeof(ur_context_handle_t), size);

    ur_context_handle_t returned_context = nullptr;
    ASSERT_SUCCESS(
        urQueueGetInfo(queue, infoType, size, &returned_context, nullptr));

    ASSERT_EQ(context, returned_context);
}

TEST_P(urQueueGetInfoTest, Device) {
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

    size_t size = 0;
    auto infoType = UR_QUEUE_INFO_DEVICE;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urQueueGetInfo(queue, infoType, 0, nullptr, &size), infoType);
    ASSERT_EQ(sizeof(ur_device_handle_t), size);

    ur_device_handle_t returned_device = nullptr;
    ASSERT_SUCCESS(
        urQueueGetInfo(queue, infoType, size, &returned_device, nullptr));

    ASSERT_EQ(device, returned_device);
}

TEST_P(urQueueGetInfoTest, Flags) {
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

    size_t size = 0;
    auto infoType = UR_QUEUE_INFO_FLAGS;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urQueueGetInfo(queue, infoType, 0, nullptr, &size), infoType);
    ASSERT_EQ(sizeof(ur_queue_flags_t), size);

    ur_queue_flags_t returned_flags = 0;
    ASSERT_SUCCESS(
        urQueueGetInfo(queue, infoType, size, &returned_flags, nullptr));

    EXPECT_EQ(returned_flags, queue_properties.flags);
}

TEST_P(urQueueGetInfoTest, ReferenceCount) {
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

    size_t size = 0;
    auto infoType = UR_QUEUE_INFO_REFERENCE_COUNT;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urQueueGetInfo(queue, infoType, 0, nullptr, &size), infoType);
    ASSERT_EQ(sizeof(uint32_t), size);

    uint32_t returned_reference_count = 0;
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, size,
                                  &returned_reference_count, nullptr));

    ASSERT_GT(returned_reference_count, 0U);
}

TEST_P(urQueueGetInfoTest, EmptyQueue) {
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

    size_t size = 0;
    auto infoType = UR_QUEUE_INFO_EMPTY;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urQueueGetInfo(queue, infoType, 0, nullptr, &size), infoType);
    ASSERT_EQ(sizeof(ur_bool_t), size);
}

TEST_P(urQueueGetInfoTest, InvalidNullHandleQueue) {
    ur_context_handle_t context = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueGetInfo(nullptr, UR_QUEUE_INFO_CONTEXT,
                                    sizeof(ur_context_handle_t), &context,
                                    nullptr));
}

TEST_P(urQueueGetInfoTest, InvalidEnumerationProperty) {
    ur_context_handle_t context = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urQueueGetInfo(queue, UR_QUEUE_INFO_FORCE_UINT32,
                                    sizeof(ur_context_handle_t), &context,
                                    nullptr));
}

TEST_P(urQueueGetInfoTest, InvalidSizeZero) {
    ur_context_handle_t context = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_SIZE,
        urQueueGetInfo(queue, UR_QUEUE_INFO_CONTEXT, 0, &context, nullptr));
}

TEST_P(urQueueGetInfoTest, InvalidSizeSmall) {
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

    ur_context_handle_t context = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urQueueGetInfo(queue, UR_QUEUE_INFO_CONTEXT,
                                    sizeof(ur_context_handle_t) - 1, &context,
                                    nullptr));
}

TEST_P(urQueueGetInfoTest, InvalidNullPointerPropValue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urQueueGetInfo(queue, UR_QUEUE_INFO_CONTEXT,
                                    sizeof(ur_context_handle_t), nullptr,
                                    nullptr));
}

TEST_P(urQueueGetInfoTest, InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urQueueGetInfo(queue, UR_QUEUE_INFO_CONTEXT, 0, nullptr, nullptr));
}

struct urQueueGetInfoDeviceQueueTestWithInfoParam : public uur::urQueueTest {
    void SetUp() {
        UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});
        urQueueGetInfoTest::SetUp();
        ur_queue_flags_t deviceQueueCapabilities = 0;
        ASSERT_SUCCESS(
            urDeviceGetInfo(device, UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES,
                            sizeof(deviceQueueCapabilities),
                            &deviceQueueCapabilities, nullptr));
        if (!deviceQueueCapabilities) {
            GTEST_SKIP() << "Queue on device is not supported.";
        }
        ASSERT_SUCCESS(
            urQueueCreate(context, device, &queueProperties, &queue));
    }

    void TearDown() {
        if (queue) {
            ASSERT_SUCCESS(urQueueRelease(queue));
        }
        urQueueGetInfoTest::TearDown();
    }

    ur_queue_handle_t queue = nullptr;
    ur_queue_properties_t queueProperties = {
        UR_STRUCTURE_TYPE_QUEUE_PROPERTIES, nullptr,
        UR_QUEUE_FLAG_ON_DEVICE | UR_QUEUE_FLAG_ON_DEVICE_DEFAULT |
            UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueGetInfoDeviceQueueTestWithInfoParam);

TEST_P(urQueueGetInfoDeviceQueueTestWithInfoParam, DeviceDefault) {
    size_t size = 0;
    auto infoType = UR_QUEUE_INFO_DEVICE_DEFAULT;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urQueueGetInfo(queue, infoType, 0, nullptr, &size), infoType);
    ASSERT_EQ(sizeof(ur_queue_handle_t), size);

    ur_queue_handle_t returned_queue = nullptr;
    ASSERT_SUCCESS(
        urQueueGetInfo(queue, infoType, size, &returned_queue, nullptr));

    ASSERT_EQ(queue, returned_queue);
}

TEST_P(urQueueGetInfoDeviceQueueTestWithInfoParam, Size) {
    size_t size = 0;
    auto infoType = UR_QUEUE_INFO_SIZE;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urQueueGetInfo(queue, infoType, 0, nullptr, &size), infoType);
    ASSERT_EQ(sizeof(uint32_t), size);

    uint32_t returned_size = 0;
    ASSERT_SUCCESS(
        urQueueGetInfo(queue, infoType, size, &returned_size, nullptr));

    ASSERT_GT(returned_size, 0);
}
