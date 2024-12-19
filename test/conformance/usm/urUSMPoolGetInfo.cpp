// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ur_api.h"
#include <map>
#include <uur/fixtures.h>

std::unordered_map<ur_usm_pool_info_t, size_t> pool_info_size_map = {
    {UR_USM_POOL_INFO_CONTEXT, sizeof(ur_context_handle_t)},
    {UR_USM_POOL_INFO_REFERENCE_COUNT, sizeof(uint32_t)},
};

using urUSMPoolGetInfoTestWithInfoParam =
    uur::urUSMPoolTestWithParam<ur_usm_pool_info_t>;

UUR_TEST_SUITE_P(urUSMPoolGetInfoTestWithInfoParam,
                 ::testing::Values(UR_USM_POOL_INFO_CONTEXT,
                                   UR_USM_POOL_INFO_REFERENCE_COUNT),
                 uur::deviceTestWithParamPrinter<ur_usm_pool_info_t>);

TEST_P(urUSMPoolGetInfoTestWithInfoParam, Success) {
    ur_usm_pool_info_t info_type = getParam();
    size_t size = 0;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urUSMPoolGetInfo(pool, info_type, 0, nullptr, &size), info_type);
    ASSERT_NE(size, 0);

    if (const auto expected_size = pool_info_size_map.find(info_type);
        expected_size != pool_info_size_map.end()) {
        ASSERT_EQ(expected_size->second, size);
    }

    std::vector<uint8_t> data(size);
    ASSERT_SUCCESS(
        urUSMPoolGetInfo(pool, info_type, size, data.data(), nullptr));
}

using urUSMPoolGetInfoTest = uur::urUSMPoolTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUSMPoolGetInfoTest);

TEST_P(urUSMPoolGetInfoTest, InvalidNullHandlePool) {
    ur_context_handle_t context = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urUSMPoolGetInfo(nullptr, UR_USM_POOL_INFO_CONTEXT,
                                      sizeof(ur_context_handle_t), &context,
                                      nullptr));
}

TEST_P(urUSMPoolGetInfoTest, InvalidEnumerationProperty) {
    ur_context_handle_t context = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urUSMPoolGetInfo(pool, UR_USM_POOL_INFO_FORCE_UINT32,
                                      sizeof(ur_context_handle_t), &context,
                                      nullptr));
}

TEST_P(urUSMPoolGetInfoTest, InvalidSizeZero) {
    ur_context_handle_t context = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_SIZE,
        urUSMPoolGetInfo(pool, UR_USM_POOL_INFO_CONTEXT, 0, &context, nullptr));
}

TEST_P(urUSMPoolGetInfoTest, InvalidSizeTooSmall) {
    ur_context_handle_t context = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urUSMPoolGetInfo(pool, UR_USM_POOL_INFO_CONTEXT,
                                      sizeof(ur_context_handle_t) - 1, &context,
                                      nullptr));
}

TEST_P(urUSMPoolGetInfoTest, InvalidNullPointerPropValue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urUSMPoolGetInfo(pool, UR_USM_POOL_INFO_CONTEXT,
                                      sizeof(ur_context_handle_t), nullptr,
                                      nullptr));
}

TEST_P(urUSMPoolGetInfoTest, InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urUSMPoolGetInfo(pool, UR_USM_POOL_INFO_CONTEXT, 0, nullptr, nullptr));
}
