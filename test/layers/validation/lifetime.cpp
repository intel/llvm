// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.hpp"

TEST_F(urTest, testUrAdapterHandleLifetimeExpectFail) {
    size_t size = 0;
    ur_adapter_handle_t adapter = (ur_adapter_handle_t)0xC0FFEE;
    ur_adapter_info_t info_type = UR_ADAPTER_INFO_BACKEND;
    ASSERT_EQ(urAdapterGetInfo(adapter, info_type, 0, nullptr, &size),
              UR_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_F(valAdapterTest, testUrAdapterHandleLifetimeExpectSuccess) {
    size_t size = 0;
    ur_adapter_info_t info_type = UR_ADAPTER_INFO_BACKEND;
    ASSERT_EQ(urAdapterGetInfo(adapter, info_type, 0, nullptr, &size),
              UR_RESULT_SUCCESS);
}
