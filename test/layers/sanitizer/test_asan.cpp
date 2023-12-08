/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file codeloc.cpp
 *
 */

#include <gtest/gtest.h>
#include <ur_api.h>

TEST(sanitizerTest, asan) {
    ur_loader_config_handle_t loader_config;
    ASSERT_EQ(urLoaderConfigCreate(&loader_config), UR_RESULT_SUCCESS);

    ASSERT_EQ(urLoaderConfigEnableLayer(loader_config, "UR_LAYER_ASAN"),
              UR_RESULT_SUCCESS);

    ASSERT_EQ(urLoaderInit(0, loader_config), UR_RESULT_SUCCESS);
}
