// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"
#include "uur/raii.h"

using urContextCreateTest = uur::urDeviceTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextCreateTest);

TEST_P(urContextCreateTest, Success) {
    uur::raii::Context context = nullptr;
    ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, context.ptr()));
    ASSERT_NE(nullptr, context);
}

TEST_P(urContextCreateTest, SuccessWithProperties) {
    ur_context_properties_t properties{UR_STRUCTURE_TYPE_CONTEXT_PROPERTIES};
    uur::raii::Context context = nullptr;
    ASSERT_SUCCESS(urContextCreate(1, &device, &properties, context.ptr()));
    ASSERT_NE(nullptr, context);
}

TEST_P(urContextCreateTest, InvalidNullPointerDevices) {
    uur::raii::Context context = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urContextCreate(1, nullptr, nullptr, context.ptr()));
}

TEST_P(urContextCreateTest, InvalidNullPointerContext) {
    auto device = GetParam();
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urContextCreate(1, &device, nullptr, nullptr));
}

using urContextCreateMultiDeviceTest = uur::urAllDevicesTest;
TEST_F(urContextCreateMultiDeviceTest, Success) {
    if (devices.size() < 2) {
        GTEST_SKIP();
    }
    uur::raii::Context context = nullptr;
    ASSERT_SUCCESS(urContextCreate(static_cast<uint32_t>(devices.size()),
                                   devices.data(), nullptr, context.ptr()));
    ASSERT_NE(nullptr, context);
}
