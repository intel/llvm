// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urEnqueueUSMMemAdviseWithParamTest =
    uur::urUSMDeviceAllocTestWithParam<ur_mem_advice_t>;
UUR_TEST_SUITE_P(urEnqueueUSMMemAdviseWithParamTest,
                 ::testing::Values(UR_MEM_ADVICE_DEFAULT),
                 uur::deviceTestWithParamPrinter<ur_mem_advice_t>);

TEST_P(urEnqueueUSMMemAdviseWithParamTest, Success) {
    ur_event_handle_t advise_event = nullptr;
    ASSERT_SUCCESS(urEnqueueUSMMemAdvise(queue, ptr, allocation_size,
                                         getParam(), &advise_event));

    ASSERT_NE(advise_event, nullptr);
    ASSERT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(urEventWait(1, &advise_event));

    ur_event_status_t advise_event_status{};
    ASSERT_SUCCESS(urEventGetInfo(
        advise_event, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
        sizeof(ur_event_status_t), &advise_event_status, nullptr));
    EXPECT_EQ(advise_event_status, UR_EVENT_STATUS_COMPLETE);
    ASSERT_SUCCESS(urEventRelease(advise_event));
}

using urEnqueueUSMMemAdviseTest = uur::urUSMDeviceAllocTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueUSMMemAdviseTest);

TEST_P(urEnqueueUSMMemAdviseTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueUSMMemAdvise(nullptr, ptr, allocation_size,
                                           UR_MEM_ADVICE_DEFAULT, nullptr));
}

TEST_P(urEnqueueUSMMemAdviseTest, InvalidNullPointerMem) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueUSMMemAdvise(queue, nullptr, allocation_size,
                                           UR_MEM_ADVICE_DEFAULT, nullptr));
}

TEST_P(urEnqueueUSMMemAdviseTest, InvalidEnumeration) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urEnqueueUSMMemAdvise(queue, ptr, allocation_size,
                                           UR_MEM_ADVICE_FORCE_UINT32,
                                           nullptr));
}

TEST_P(urEnqueueUSMMemAdviseTest, InvalidSizeZero) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_SIZE,
        urEnqueueUSMMemAdvise(queue, ptr, 0, UR_MEM_ADVICE_DEFAULT, nullptr));
}

TEST_P(urEnqueueUSMMemAdviseTest, InvalidSizeTooLarge) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueUSMMemAdvise(queue, ptr, allocation_size * 2,
                                           UR_MEM_ADVICE_DEFAULT, nullptr));
}
