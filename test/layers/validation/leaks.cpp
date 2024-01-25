// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.hpp"

TEST_F(urTest, testUrAdapterGetLeak) {
    ur_adapter_handle_t adapter = nullptr;
    ASSERT_EQ(urAdapterGet(1, &adapter, nullptr), UR_RESULT_SUCCESS);
    ASSERT_NE(nullptr, adapter);
}

TEST_F(urTest, testUrAdapterRetainLeak) {
    ur_adapter_handle_t adapter = nullptr;
    ASSERT_EQ(urAdapterGet(1, &adapter, nullptr), UR_RESULT_SUCCESS);
    ASSERT_NE(nullptr, adapter);
    ASSERT_EQ(urAdapterRetain(adapter), UR_RESULT_SUCCESS);
}

TEST_F(urTest, testUrAdapterRetainNonexistent) {
    ur_adapter_handle_t adapter = (ur_adapter_handle_t)0xBEEF;
    ASSERT_EQ(urAdapterRetain(adapter), UR_RESULT_SUCCESS);
    ASSERT_NE(nullptr, adapter);
}

TEST_F(valDeviceTest, testUrContextCreateLeak) {
    ur_context_handle_t context = nullptr;
    ASSERT_EQ(urContextCreate(1, &device, nullptr, &context),
              UR_RESULT_SUCCESS);
    ASSERT_NE(nullptr, context);
}

TEST_F(valDeviceTest, testUrContextRetainLeak) {
    ur_context_handle_t context = nullptr;
    ASSERT_EQ(urContextCreate(1, &device, nullptr, &context),
              UR_RESULT_SUCCESS);
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(urContextRetain(context), UR_RESULT_SUCCESS);
}

TEST_F(valDeviceTest, testUrContextRetainNonexistent) {
    ur_context_handle_t context = (ur_context_handle_t)0xC0FFEE;
    ASSERT_EQ(urContextRetain(context), UR_RESULT_SUCCESS);
}

TEST_F(valDeviceTest, testUrContextCreateSuccess) {
    ur_context_handle_t context = nullptr;
    ASSERT_EQ(urContextCreate(1, &device, nullptr, &context),
              UR_RESULT_SUCCESS);
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
}

TEST_F(valDeviceTest, testUrContextRetainSuccess) {
    ur_context_handle_t context = nullptr;
    ASSERT_EQ(urContextCreate(1, &device, nullptr, &context),
              UR_RESULT_SUCCESS);
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(urContextRetain(context), UR_RESULT_SUCCESS);
    ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
    ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
}

TEST_F(valDeviceTest, testUrContextReleaseLeak) {
    ur_context_handle_t context = nullptr;
    ASSERT_EQ(urContextCreate(1, &device, nullptr, &context),
              UR_RESULT_SUCCESS);
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
    ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
}

TEST_F(valDeviceTest, testUrContextReleaseNonexistent) {
    ur_context_handle_t context = (ur_context_handle_t)0xC0FFEE;
    ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
}
