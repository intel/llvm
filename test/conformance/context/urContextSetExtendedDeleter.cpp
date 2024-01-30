// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"
#include "uur/raii.h"

using urContextSetExtendedDeleterTest = uur::urDeviceTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextSetExtendedDeleterTest);

TEST_P(urContextSetExtendedDeleterTest, Success) {
    bool called = false;
    {
        uur::raii::Context context = nullptr;
        ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, context.ptr()));
        ASSERT_NE(context, nullptr);

        ur_context_extended_deleter_t deleter = [](void *userdata) {
            *static_cast<bool *>(userdata) = true;
        };

        ASSERT_SUCCESS(urContextSetExtendedDeleter(context, deleter, &called));
    }
    ASSERT_TRUE(called);
}

TEST_P(urContextSetExtendedDeleterTest, InvalidNullHandleContext) {
    ur_context_extended_deleter_t deleter = [](void *) {};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urContextSetExtendedDeleter(nullptr, deleter, nullptr));
}

TEST_P(urContextSetExtendedDeleterTest, InvalidNullPointerDeleter) {
    uur::raii::Context context = nullptr;
    ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, context.ptr()));
    ASSERT_NE(context, nullptr);

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urContextSetExtendedDeleter(context, nullptr, nullptr));
}
