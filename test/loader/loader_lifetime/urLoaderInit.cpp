// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception
#include "fixtures.hpp"
#include "ur_api.h"
#include <ur_print.hpp>

using urLoaderInitTestWithParam =
    ::testing::TestWithParam<ur_device_init_flags_t>;
INSTANTIATE_TEST_SUITE_P(
    , urLoaderInitTestWithParam,
    ::testing::Values(UR_DEVICE_INIT_FLAG_GPU, UR_DEVICE_INIT_FLAG_CPU,
                      UR_DEVICE_INIT_FLAG_FPGA, UR_DEVICE_INIT_FLAG_MCA,
                      UR_DEVICE_INIT_FLAG_VPU,
                      /* Combinations */
                      UR_DEVICE_INIT_FLAG_GPU | UR_DEVICE_INIT_FLAG_CPU,
                      UR_DEVICE_INIT_FLAG_FPGA | UR_DEVICE_INIT_FLAG_VPU),
    [](const ::testing::TestParamInfo<ur_device_init_flags_t> &info) {
      std::stringstream ss;
      ur::details::printFlag<ur_device_init_flag_t>(ss, info.param);
      return GTestSanitizeString(ss.str());
    });

TEST_P(urLoaderInitTestWithParam, Success) {
  ur_loader_config_handle_t config = nullptr;
  urLoaderConfigCreate(&config);
  urLoaderConfigEnableLayer(config, "UR_LAYER_FULL_VALIDATION");

  ur_device_init_flags_t device_flags = GetParam();
  ASSERT_SUCCESS(urLoaderInit(device_flags, config));

  ASSERT_SUCCESS(urLoaderTearDown());
  urLoaderConfigRelease(config);
}

TEST(urLoaderInitTest, ErrorInvalidEnumerationDeviceFlags) {
  const ur_device_init_flags_t device_flags = UR_DEVICE_INIT_FLAG_FORCE_UINT32;
  ASSERT_EQ(UR_RESULT_ERROR_INVALID_ENUMERATION,
            urLoaderInit(device_flags, nullptr));
}
