// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/checks.h>

TEST(urInitTest, Success) {
    ur_device_init_flags_t device_flags = 0;
    ASSERT_SUCCESS(urInit(device_flags));

    ur_tear_down_params_t tear_down_params{};
    ASSERT_SUCCESS(urTearDown(&tear_down_params));
}

TEST(urInitTest, ErrorInvalidEnumerationDeviceFlags) {
    const ur_device_init_flags_t device_flags =
        UR_DEVICE_INIT_FLAG_FORCE_UINT32;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION, urInit(device_flags));
}
