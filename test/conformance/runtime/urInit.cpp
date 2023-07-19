// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/checks.h>

using urInitTestWithParam = ::testing::TestWithParam<ur_device_init_flags_t>;
INSTANTIATE_TEST_SUITE_P(
    , urInitTestWithParam,
    ::testing::Values(UR_DEVICE_INIT_FLAG_GPU, UR_DEVICE_INIT_FLAG_CPU,
                      UR_DEVICE_INIT_FLAG_FPGA, UR_DEVICE_INIT_FLAG_MCA,
                      UR_DEVICE_INIT_FLAG_VPU,
                      /* Combinations */
                      UR_DEVICE_INIT_FLAG_GPU | UR_DEVICE_INIT_FLAG_CPU,
                      UR_DEVICE_INIT_FLAG_FPGA | UR_DEVICE_INIT_FLAG_VPU),
    [](const ::testing::TestParamInfo<ur_device_init_flags_t> &info) {
        std::stringstream ss;
        ur_params::serializeFlag<ur_device_init_flag_t>(ss, info.param);
        return uur::GTestSanitizeString(ss.str());
    });

TEST_P(urInitTestWithParam, Success) {
    ur_loader_config_handle_t config = nullptr;
    urLoaderConfigCreate(&config);
    urLoaderConfigEnableLayer(config, "UR_LAYER_FULL_VALIDATION");

    ur_device_init_flags_t device_flags = GetParam();
    ASSERT_SUCCESS(urInit(device_flags, config));

    ur_tear_down_params_t tear_down_params{nullptr};
    ASSERT_SUCCESS(urTearDown(&tear_down_params));
}

TEST(urInitTest, ErrorInvalidEnumerationDeviceFlags) {
    const ur_device_init_flags_t device_flags =
        UR_DEVICE_INIT_FLAG_FORCE_UINT32;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urInit(device_flags, nullptr));
}
