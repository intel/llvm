// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "fixtures.hpp"

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
