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

struct ur_code_location_t test_callback(void *userdata) {
    (void)userdata;

    ur_code_location_t codeloc;
    codeloc.columnNumber = 1;
    codeloc.lineNumber = 2;
    codeloc.functionName = "fname";
    codeloc.sourceFile = "sfile";

    return codeloc;
}

TEST(LoaderCodeloc, NullCallback) {
    ur_loader_config_handle_t loader_config;
    ASSERT_EQ(urLoaderConfigCreate(&loader_config), UR_RESULT_SUCCESS);
    ASSERT_EQ(
        urLoaderConfigSetCodeLocationCallback(loader_config, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
    urLoaderConfigRelease(loader_config);
}

TEST(LoaderCodeloc, NullHandle) {
    ASSERT_EQ(
        urLoaderConfigSetCodeLocationCallback(nullptr, test_callback, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST(LoaderCodeloc, Success) {
    ur_loader_config_handle_t loader_config;
    ASSERT_EQ(urLoaderConfigCreate(&loader_config), UR_RESULT_SUCCESS);
    ASSERT_EQ(urLoaderConfigSetCodeLocationCallback(loader_config,
                                                    test_callback, nullptr),
              UR_RESULT_SUCCESS);
    urLoaderInit(0, loader_config);
    uint32_t nadapters;
    urAdapterGet(0, nullptr, &nadapters);
    urLoaderConfigRelease(loader_config);
}
