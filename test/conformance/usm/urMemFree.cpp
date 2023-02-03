// Copyright (C) $(year) Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urMemFreeTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemFreeTest);

TEST_P(urMemFreeTest, Success) {
    void *ptr = nullptr;
    ur_usm_mem_flags_t flags;
    ASSERT_SUCCESS(
        urUSMDeviceAlloc(context, device, &flags, sizeof(int), 0, &ptr));

    ur_event_handle_t event = nullptr;
    ASSERT_SUCCESS(
        urEnqueueUSMMemset(queue, ptr, 0, sizeof(int), 0, nullptr, &event));
    ASSERT_SUCCESS(urEventWait(1, &event));

    ASSERT_NE(ptr, nullptr);
    ASSERT_SUCCESS(urMemFree(context, ptr));
}

TEST_P(urMemFreeTest, InvalidNullContext) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urMemFree(nullptr, nullptr));
}

TEST_P(urMemFreeTest, InvalidNullPtrMem) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urMemFree(context, nullptr));
}
