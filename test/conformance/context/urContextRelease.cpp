// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urContextReleaseTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextReleaseTest);

TEST_P(urContextReleaseTest, Success) {
  ASSERT_SUCCESS(urContextRetain(context));
  ASSERT_SUCCESS(urContextRelease(context));
}

TEST_P(urContextReleaseTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urContextRelease(nullptr));
}
