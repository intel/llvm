// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

#include <cstring>

struct urAdapterGetInfoTest : uur::runtime::urAdapterTest,
                              ::testing::WithParamInterface<ur_adapter_info_t> {

    void SetUp() {
        UUR_RETURN_ON_FATAL_FAILURE(uur::runtime::urAdapterTest::SetUp());
        adapter = adapters[0];
    }

    ur_adapter_handle_t adapter;
};

std::unordered_map<ur_adapter_info_t, size_t> adapter_info_size_map = {
    {UR_ADAPTER_INFO_BACKEND, sizeof(ur_adapter_backend_t)},
    {UR_ADAPTER_INFO_VERSION, sizeof(uint32_t)},
    {UR_ADAPTER_INFO_REFERENCE_COUNT, sizeof(uint32_t)},
};

INSTANTIATE_TEST_SUITE_P(
    urAdapterGetInfo, urAdapterGetInfoTest,
    ::testing::Values(UR_ADAPTER_INFO_BACKEND, UR_ADAPTER_INFO_VERSION,
                      UR_ADAPTER_INFO_REFERENCE_COUNT),
    [](const ::testing::TestParamInfo<ur_adapter_info_t> &info) {
        std::stringstream ss;
        ss << info.param;
        return ss.str();
    });

TEST_P(urAdapterGetInfoTest, Success) {
    size_t size = 0;
    ur_adapter_info_t info_type = GetParam();
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urAdapterGetInfo(adapter, info_type, 0, nullptr, &size), info_type);
    ASSERT_NE(size, 0);

    if (const auto expected_size = adapter_info_size_map.find(info_type);
        expected_size != adapter_info_size_map.end()) {
        ASSERT_EQ(expected_size->second, size);
    }

    std::vector<char> info_data(size);
    ASSERT_SUCCESS(
        urAdapterGetInfo(adapter, info_type, size, info_data.data(), nullptr));
}

TEST_P(urAdapterGetInfoTest, InvalidNullHandleAdapter) {
    size_t size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urAdapterGetInfo(nullptr, GetParam(), 0, nullptr, &size));
}

TEST_F(urAdapterGetInfoTest, InvalidEnumerationAdapterInfoType) {
    size_t size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urAdapterGetInfo(adapter, UR_ADAPTER_INFO_FORCE_UINT32, 0,
                                      nullptr, &size));
}

TEST_F(urAdapterGetInfoTest, InvalidSizeZero) {
    ur_adapter_backend_t backend;
    ASSERT_EQ_RESULT(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND, 0,
                                      &backend, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_F(urAdapterGetInfoTest, InvalidSizeSmall) {
    ur_adapter_backend_t backend;
    ASSERT_EQ_RESULT(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND,
                                      sizeof(backend) - 1, &backend, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_F(urAdapterGetInfoTest, InvalidNullPointerPropValue) {
    ur_adapter_backend_t backend;
    ASSERT_EQ_RESULT(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND,
                                      sizeof(backend), nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_F(urAdapterGetInfoTest, InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(
        urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_F(urAdapterGetInfoTest, ReferenceCountNotZero) {
    uint32_t referenceCount = 0;

    ASSERT_SUCCESS(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_REFERENCE_COUNT,
                                    sizeof(referenceCount), &referenceCount,
                                    nullptr));
    ASSERT_GT(referenceCount, 0);
}

TEST_F(urAdapterGetInfoTest, ValidAdapterBackend) {
    ur_adapter_backend_t backend;
    ASSERT_SUCCESS(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND,
                                    sizeof(backend), &backend, nullptr));

    ASSERT_TRUE(backend >= UR_ADAPTER_BACKEND_LEVEL_ZERO &&
                backend <= UR_ADAPTER_BACKEND_NATIVE_CPU);
}
