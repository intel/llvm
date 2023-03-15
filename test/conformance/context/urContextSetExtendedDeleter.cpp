// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urContextSetExtendedDeleterTest = uur::urDeviceTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextSetExtendedDeleterTest);

TEST_P(urContextSetExtendedDeleterTest, Success) {
    ur_context_handle_t context = nullptr;
    ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, &context));
    ASSERT_NE(context, nullptr);

    bool called = false;
    ur_context_extended_deleter_t deleter = [](void *userdata) {
        *static_cast<bool *>(userdata) = true;
    };

    ASSERT_SUCCESS(urContextSetExtendedDeleter(context, deleter, &called));
    ASSERT_SUCCESS(urContextRelease(context));
    ASSERT_TRUE(called);
}

TEST_P(urContextSetExtendedDeleterTest, InvalidNullHandleContext) {
    ur_context_extended_deleter_t deleter = [](void *) {};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urContextSetExtendedDeleter(nullptr, deleter, nullptr));
}

TEST_P(urContextSetExtendedDeleterTest, InvalidNullPointerDeleter) {
    ur_context_handle_t context = nullptr;
    ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, &context));
    ASSERT_NE(context, nullptr);

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urContextSetExtendedDeleter(context, nullptr, nullptr));
    ASSERT_SUCCESS(urContextRelease(context));
}
