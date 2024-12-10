// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ur_api.h"
#include "uur/raii.h"
#include <uur/fixtures.h>

using urQueueCreateTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueCreateTest);

TEST_P(urQueueCreateTest, Success) {
    uur::raii::Queue queue = nullptr;
    ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, queue.ptr()));
    ASSERT_NE(nullptr, queue);

    ur_queue_info_t queue_flags;
    ASSERT_SUCCESS(urQueueGetInfo(queue, UR_QUEUE_INFO_FLAGS,
                                  sizeof(ur_queue_info_t), &queue_flags,
                                  nullptr));

    /* Check that the queue was created without any flag */
    ASSERT_EQ(queue_flags, 0);
}

using urQueueCreateWithParamTest = uur::urContextTestWithParam<ur_queue_flag_t>;
UUR_TEST_SUITE_P(urQueueCreateWithParamTest,
                 testing::Values(UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                 UR_QUEUE_FLAG_PROFILING_ENABLE,
                                 UR_QUEUE_FLAG_ON_DEVICE,
                                 UR_QUEUE_FLAG_ON_DEVICE_DEFAULT,
                                 UR_QUEUE_FLAG_DISCARD_EVENTS,
                                 UR_QUEUE_FLAG_PRIORITY_LOW,
                                 UR_QUEUE_FLAG_PRIORITY_HIGH,
                                 UR_QUEUE_FLAG_SUBMISSION_BATCHED,
                                 UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE,
                                 UR_QUEUE_FLAG_USE_DEFAULT_STREAM,
                                 UR_QUEUE_FLAG_SYNC_WITH_DEFAULT_STREAM,
                                 UR_QUEUE_FLAG_LOW_POWER_EVENTS_EXP),
                 uur::deviceTestWithParamPrinter<ur_queue_flag_t>);

TEST_P(urQueueCreateWithParamTest, SuccessWithProperties) {
    ur_queue_flags_t supportedFlags{};
    ASSERT_SUCCESS(uur::GetDeviceQueueOnHostProperties(device, supportedFlags));

    ur_queue_flags_t queryFlag = getParam();
    if (!(supportedFlags & queryFlag)) {
        GTEST_SKIP() << queryFlag << " : is not supported by the device.";
    }

    ur_queue_handle_t queue = nullptr;
    ur_queue_properties_t props = {
        /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
        /*.pNext =*/nullptr,
        /*.flags =*/queryFlag,
    };
    ASSERT_SUCCESS(urQueueCreate(context, device, &props, &queue));
    ASSERT_NE(queue, nullptr);

    // query the queue to check that it has these properties
    ur_queue_flags_t queueFlags{};
    ASSERT_SUCCESS(urQueueGetInfo(queue, UR_QUEUE_INFO_FLAGS,
                                  sizeof(ur_queue_flags_t), &queueFlags,
                                  nullptr));
    ASSERT_TRUE(queueFlags & queryFlag);

    // Check that no other bit is set (i.e. is power of 2)
    ASSERT_TRUE(queueFlags != 0 && (queueFlags & (queueFlags - 1)) == 0);

    ASSERT_SUCCESS(urQueueRelease(queue));
}

/* Creates two queues with the same platform and device, and checks that the
 * queried device and platform of both queues match. */
TEST_P(urQueueCreateWithParamTest, MatchingDeviceHandles) {
    ur_queue_flags_t supportedFlags{};
    ASSERT_SUCCESS(uur::GetDeviceQueueOnHostProperties(device, supportedFlags));

    ur_queue_flags_t queryFlag = getParam();
    if (!(supportedFlags & queryFlag)) {
        GTEST_SKIP() << queryFlag << " : is not supported by the device.";
    }

    ur_queue_properties_t props = {
        /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
        /*.pNext =*/nullptr,
        /*.flags =*/queryFlag,
    };

    uur::raii::Queue queue1 = nullptr;
    ASSERT_SUCCESS(urQueueCreate(context, device, &props, queue1.ptr()));
    ASSERT_NE(queue1, nullptr);

    uur::raii::Queue queue2 = nullptr;
    ASSERT_SUCCESS(urQueueCreate(context, device, &props, queue2.ptr()));
    ASSERT_NE(queue2, nullptr);

    ur_device_handle_t deviceQueue1;
    ASSERT_SUCCESS(urQueueGetInfo(queue1, UR_QUEUE_INFO_DEVICE,
                                  sizeof(ur_device_handle_t), &deviceQueue1,
                                  nullptr));

    ur_device_handle_t deviceQueue2;
    ASSERT_SUCCESS(urQueueGetInfo(queue1, UR_QUEUE_INFO_DEVICE,
                                  sizeof(ur_device_handle_t), &deviceQueue2,
                                  nullptr));

    ASSERT_EQ(deviceQueue1, deviceQueue2);
}

/* Create a queue and check that it returns the right context*/
TEST_P(urQueueCreateTest, CheckContext) {

    uur::raii::Queue queue = nullptr;
    ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, queue.ptr()));
    ASSERT_NE(queue.ptr(), nullptr);

    ur_context_handle_t returned_context = nullptr;
    ASSERT_SUCCESS(urQueueGetInfo(queue, UR_QUEUE_INFO_CONTEXT,
                                  sizeof(ur_context_handle_t),
                                  &returned_context, nullptr));

    ASSERT_EQ(this->context, returned_context);
}

using urQueueCreateTestMultipleDevices = uur::urAllDevicesTest;

/* Create a queue using a context from a different device */
TEST_F(urQueueCreateTestMultipleDevices, ContextFromWrongDevice) {

    if (devices.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 devices in the system";
    }
    ur_device_handle_t device1 = devices[0];
    uur::raii::Context context1 = nullptr;
    urContextCreate(1, &device1, nullptr, context1.ptr());

    ur_device_handle_t device2 = devices[1];
    uur::raii::Context context2 = nullptr;
    urContextCreate(1, &device2, nullptr, context2.ptr());

    ur_queue_handle_t queue = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_DEVICE,
                     urQueueCreate(context2, device1, nullptr, &queue));
    ASSERT_EQ(queue, nullptr);
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

TEST_P(urQueueCreateTest, InvalidQueueProperties) {
    ur_queue_properties_t props = {
        /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
        /*.pNext =*/nullptr,
        /*.flags =*/UR_QUEUE_FLAG_FORCE_UINT32,
    };

    // Initial value is just not a valid enum
    {
        ur_queue_handle_t queue = nullptr;
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                         urQueueCreate(context, device, &props, &queue));
    }
    // It should be an error to specify both low/high priorities
    {
        ur_queue_handle_t queue = nullptr;
        props.flags = UR_QUEUE_FLAG_PRIORITY_HIGH | UR_QUEUE_FLAG_PRIORITY_LOW;
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES,
                         urQueueCreate(context, device, &props, &queue));
    }
    // It should be an error to specify both batched and immediate submission
    {
        ur_queue_handle_t queue = nullptr;
        props.flags = UR_QUEUE_FLAG_SUBMISSION_BATCHED |
                      UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE;
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES,
                         urQueueCreate(context, device, &props, &queue));
    }
}
