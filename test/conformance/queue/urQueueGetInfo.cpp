// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <map>
#include <uur/fixtures.h>

using urQueueGetInfoTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueGetInfoTest);

TEST_P(urQueueGetInfoTest, Context) {
    size_t size = 0;
    auto infoType = UR_QUEUE_INFO_CONTEXT;
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    ASSERT_EQ(sizeof(ur_context_handle_t), size);

    std::vector<uint8_t> data(size);
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, size, data.data(), nullptr));

    auto returned_context =
        reinterpret_cast<ur_context_handle_t *>(data.data());
    ASSERT_EQ(context, *returned_context);
}

TEST_P(urQueueGetInfoTest, Device) {
    size_t size = 0;
    auto infoType = UR_QUEUE_INFO_DEVICE;
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    ASSERT_EQ(sizeof(ur_device_handle_t), size);

    std::vector<uint8_t> data(size);
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, size, data.data(), nullptr));

    auto returned_device = reinterpret_cast<ur_device_handle_t *>(data.data());
    ASSERT_EQ(device, *returned_device);
}

TEST_P(urQueueGetInfoTest, Flags) {
    size_t size = 0;
    auto infoType = UR_QUEUE_INFO_FLAGS;
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    ASSERT_EQ(sizeof(ur_queue_flags_t), size);

    std::vector<uint8_t> data(size);
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, size, data.data(), nullptr));

    auto returned_flags = reinterpret_cast<ur_queue_flags_t *>(data.data());
    EXPECT_EQ(*returned_flags, queue_properties.flags);
}

TEST_P(urQueueGetInfoTest, ReferenceCount) {
    size_t size = 0;
    auto infoType = UR_QUEUE_INFO_REFERENCE_COUNT;
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    ASSERT_EQ(sizeof(uint32_t), size);

    std::vector<uint8_t> data(size);
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, size, data.data(), nullptr));

    auto returned_reference_count = reinterpret_cast<uint32_t *>(data.data());
    ASSERT_GT(*returned_reference_count, 0U);
}

TEST_P(urQueueGetInfoTest, EmptyQueue) {
    size_t size = 0;
    auto infoType = UR_QUEUE_INFO_EMPTY;
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urQueueGetInfo(queue, infoType, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    ASSERT_EQ(sizeof(ur_bool_t), size);

    std::vector<uint8_t> data(size);
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, size, data.data(), nullptr));

    auto returned_empty_queue = reinterpret_cast<ur_bool_t *>(data.data());
    ASSERT_TRUE(returned_empty_queue);
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
        urQueueGetInfoTest::SetUp();
        ur_queue_flags_t deviceQueueCapabilities;
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
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    ASSERT_EQ(sizeof(ur_queue_handle_t), size);

    std::vector<uint8_t> data(size);
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, size, data.data(), nullptr));

    auto returned_queue = reinterpret_cast<ur_queue_handle_t *>(data.data());
    ASSERT_EQ(queue, *returned_queue);
}

TEST_P(urQueueGetInfoDeviceQueueTestWithInfoParam, Size) {

    size_t size = 0;
    auto infoType = UR_QUEUE_INFO_SIZE;
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    ASSERT_EQ(sizeof(uint32_t), size);

    std::vector<uint8_t> data(size);
    ASSERT_SUCCESS(urQueueGetInfo(queue, infoType, size, data.data(), nullptr));

    auto returned_size = reinterpret_cast<uint32_t *>(data.data());
    ASSERT_GT(*returned_size, 0);
}
