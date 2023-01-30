// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urMemReleaseTest = uur::urMemBufferTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemReleaseTest);

TEST_P(urMemReleaseTest, Success) {
  ASSERT_SUCCESS(urMemRetain(buffer));
  ASSERT_SUCCESS(urMemRelease(buffer));
}

TEST_P(urMemReleaseTest, InvalidNullHandleMem) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE, urMemRelease(nullptr));
}
