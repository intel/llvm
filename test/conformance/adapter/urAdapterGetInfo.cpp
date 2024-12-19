// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

#include <cstring>

using urAdapterGetInfoTest = uur::urAdapterTest;

UUR_INSTANTIATE_ADAPTER_TEST_SUITE_P(urAdapterGetInfoTest);

TEST_P(urAdapterGetInfoTest, Backend) {
    auto info_type = UR_ADAPTER_INFO_BACKEND;
    size_t size = 0;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urAdapterGetInfo(adapter, info_type, 0, nullptr, &size), info_type);
    ASSERT_NE(size, 0);

    ASSERT_EQ(size, sizeof(ur_adapter_backend_t));

    std::vector<char> info_data(size);
    ASSERT_SUCCESS(
        urAdapterGetInfo(adapter, info_type, size, info_data.data(), nullptr));
}

TEST_P(urAdapterGetInfoTest, ReferenceCount) {
    auto info_type = UR_ADAPTER_INFO_REFERENCE_COUNT;
    size_t size = 0;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urAdapterGetInfo(adapter, info_type, 0, nullptr, &size), info_type);
    ASSERT_EQ(size, sizeof(uint32_t));

    uint32_t reference_count = 0;
    ASSERT_SUCCESS(
        urAdapterGetInfo(adapter, info_type, size, &reference_count, nullptr));
    ASSERT_GE(reference_count, 0);
}

TEST_P(urAdapterGetInfoTest, InvalidNullHandleAdapter) {
    size_t size = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urAdapterGetInfo(nullptr, UR_ADAPTER_INFO_BACKEND, 0, nullptr, &size));
}

TEST_P(urAdapterGetInfoTest, InvalidEnumerationAdapterInfoType) {
    size_t size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urAdapterGetInfo(adapter, UR_ADAPTER_INFO_FORCE_UINT32, 0,
                                      nullptr, &size));
}

TEST_P(urAdapterGetInfoTest, InvalidSizeZero) {
    ur_adapter_backend_t backend;
    ASSERT_EQ_RESULT(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND, 0,
                                      &backend, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urAdapterGetInfoTest, InvalidSizeSmall) {
    ur_adapter_backend_t backend;
    ASSERT_EQ_RESULT(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND,
                                      sizeof(backend) - 1, &backend, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urAdapterGetInfoTest, InvalidNullPointerPropValue) {
    ur_adapter_backend_t backend;
    ASSERT_EQ_RESULT(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND,
                                      sizeof(backend), nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urAdapterGetInfoTest, InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(
        urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urAdapterGetInfoTest, ReferenceCountNotZero) {
    uint32_t referenceCount = 0;

    ASSERT_SUCCESS(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_REFERENCE_COUNT,
                                    sizeof(referenceCount), &referenceCount,
                                    nullptr));
    ASSERT_GT(referenceCount, 0);
}

TEST_P(urAdapterGetInfoTest, ValidAdapterBackend) {
    ur_adapter_backend_t backend;
    ASSERT_SUCCESS(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND,
                                    sizeof(backend), &backend, nullptr));

    ASSERT_TRUE(backend >= UR_ADAPTER_BACKEND_LEVEL_ZERO &&
                backend <= UR_ADAPTER_BACKEND_NATIVE_CPU);
}
