// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urQueueCreateTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueCreateTest);

TEST_P(urQueueCreateTest, Success) {
    ur_queue_handle_t queue = nullptr;
    ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, &queue));
    ASSERT_NE(nullptr, queue);
    ASSERT_SUCCESS(urQueueRelease(queue));
}

using urQueueCreateWithParamTest = uur::urContextTestWithParam<ur_queue_flag_t>;
UUR_TEST_SUITE_P(
    urQueueCreateWithParamTest,
    ::testing::Values(UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE, //
                      UR_QUEUE_FLAG_PROFILING_ENABLE               //
                      ),
    uur::deviceTestWithParamPrinter<ur_queue_flag_t>);

TEST_P(urQueueCreateWithParamTest, SuccessWithProperties) {
    ur_queue_flags_t supportedFlags{};
    ASSERT_SUCCESS(uur::GetDeviceQueueOnHostProperties(device, supportedFlags));

    ur_queue_flag_t queryFlag = getParam();
    if (!(supportedFlags & queryFlag)) {
        GTEST_SKIP() << queryFlag << " : is not supported by the device.";
    }

    ur_queue_handle_t queue = nullptr;
    const ur_queue_property_t props[] = {UR_QUEUE_PROPERTIES_FLAGS, queryFlag,
                                         0};
    ASSERT_SUCCESS(urQueueCreate(context, device, props, &queue));
    ASSERT_NE(queue, nullptr);

    // query the queue to check that it has these properties
    ur_queue_flags_t queueFlags{};
    ASSERT_SUCCESS(urQueueGetInfo(queue, UR_QUEUE_INFO_PROPERTIES,
                                  sizeof(ur_queue_flags_t), &queueFlags,
                                  nullptr));
    ASSERT_TRUE(queueFlags & queryFlag);

    ASSERT_SUCCESS(urQueueRelease(queue));
}

TEST_P(urQueueCreateTest, InvalidNullHandleContext) {
    ur_queue_handle_t queue = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueCreate(nullptr, device, nullptr, &queue));
}

TEST_P(urQueueCreateTest, InvalidNullHandleDevice) {
    ur_queue_handle_t queue = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueCreate(context, nullptr, nullptr, &queue));
}

TEST_P(urQueueCreateTest, InvalidNullPointerQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urQueueCreate(context, device, 0, nullptr));
}

TEST_P(urQueueCreateTest, InvalidValueProperties) {
    ur_queue_handle_t queue = nullptr;
    ur_queue_property_t props[] = {UR_QUEUE_PROPERTIES_FLAGS,
                                   UR_QUEUE_FLAG_FORCE_UINT32, 0};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                     urQueueCreate(context, device, props, &queue));
}

TEST_P(urQueueCreateTest, InvalidQueueProperties) {
    ur_queue_handle_t queue = nullptr;

    // It should be an error to specify both low/high priorities
    ur_queue_property_t props[] = {
        UR_QUEUE_PROPERTIES_FLAGS,
        UR_QUEUE_FLAG_PRIORITY_HIGH | UR_QUEUE_FLAG_PRIORITY_LOW, 0};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES,
                     urQueueCreate(context, device, props, &queue));
}
