// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/checks.h>

struct urTearDownTest : testing::Test {
  void SetUp() override {
    ur_platform_init_flags_t platform_flags = 0;
    ur_device_init_flags_t device_flags = 0;
    ASSERT_SUCCESS(urInit(platform_flags, device_flags));
  }
};

TEST_F(urTearDownTest, Success) {
  ur_tear_down_params_t tear_down_params{};
  ASSERT_SUCCESS(urTearDown(&tear_down_params));
}

TEST_F(urTearDownTest, InvalidNullPointerParams) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER, urTearDown(nullptr));
}
