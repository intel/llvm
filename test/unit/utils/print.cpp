// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "print.h"

TEST(Print, KernelInfo) {
    constexpr size_t BUFF_SIZE = 1024;
    char buffer[BUFF_SIZE];
    size_t out_len = 0;
    urPrintKernelInfo(UR_KERNEL_INFO_FUNCTION_NAME, buffer, BUFF_SIZE,
                      &out_len);
    EXPECT_STREQ(buffer, "UR_KERNEL_INFO_FUNCTION_NAME");
    EXPECT_EQ(out_len, strlen("UR_KERNEL_INFO_FUNCTION_NAME") + 1);
}

TEST(Print, ApiVersion) {
    char buffer[] = "UR_API_VERSION_0_0";
    size_t out_len = 0;
    urPrintApiVersion(UR_API_VERSION_0_8, buffer, strlen(buffer) + 1, &out_len);
    EXPECT_STREQ(buffer, "0.8");
    EXPECT_EQ(out_len, strlen("0.8") + 1);
}

TYPED_TEST(ParamsTest, GetParams) {
    constexpr size_t BUFF_SIZE = 1024;
    char buffer[BUFF_SIZE];
    size_t out_len = 0;
    this->params.print(buffer, BUFF_SIZE, &out_len);
    EXPECT_THAT(std::string(buffer), MatchesRegex(this->params.get_expected()));
    EXPECT_EQ(out_len, strlen(buffer) + 1);
}
