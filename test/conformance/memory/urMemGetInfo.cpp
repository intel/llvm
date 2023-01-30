// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urMemGetInfoTest = uur::urMemBufferTestWithParam<ur_mem_info_t>;

UUR_TEST_SUITE_P(urMemGetInfoTest,
                 ::testing::Values(UR_MEM_INFO_SIZE, UR_MEM_INFO_CONTEXT),
                 uur::deviceTestWithParamPrinter<ur_mem_info_t>);

TEST_P(urMemGetInfoTest, Success) {
    ur_mem_info_t info = getParam();
    size_t size;
    ASSERT_SUCCESS(urMemGetInfo(buffer, info, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    std::vector<uint8_t> info_data(size);
    ASSERT_SUCCESS(urMemGetInfo(buffer, info, size, info_data.data(), nullptr));
}
