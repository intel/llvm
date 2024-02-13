// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urUSMGetMemAllocInfoTest
    : uur::urUSMDeviceAllocTestWithParam<ur_usm_alloc_info_t> {
    void SetUp() override {
        use_pool = getParam() == UR_USM_ALLOC_INFO_POOL;
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urUSMDeviceAllocTestWithParam<ur_usm_alloc_info_t>::SetUp());
    }
};

UUR_TEST_SUITE_P(urUSMGetMemAllocInfoTest,
                 ::testing::Values(UR_USM_ALLOC_INFO_TYPE,
                                   UR_USM_ALLOC_INFO_BASE_PTR,
                                   UR_USM_ALLOC_INFO_SIZE,
                                   UR_USM_ALLOC_INFO_DEVICE,
                                   UR_USM_ALLOC_INFO_POOL),
                 uur::deviceTestWithParamPrinter<ur_usm_alloc_info_t>);

static std::unordered_map<ur_usm_alloc_info_t, size_t> usm_info_size_map = {
    {UR_USM_ALLOC_INFO_TYPE, sizeof(ur_usm_type_t)},
    {UR_USM_ALLOC_INFO_BASE_PTR, sizeof(void *)},
    {UR_USM_ALLOC_INFO_SIZE, sizeof(size_t)},
    {UR_USM_ALLOC_INFO_DEVICE, sizeof(ur_device_handle_t)},
    {UR_USM_ALLOC_INFO_POOL, sizeof(ur_usm_pool_handle_t)},
};

TEST_P(urUSMGetMemAllocInfoTest, Success) {
    size_t size = 0;
    auto alloc_info = getParam();
    ASSERT_SUCCESS(
        urUSMGetMemAllocInfo(context, ptr, alloc_info, 0, nullptr, &size));
    ASSERT_NE(size, 0);

    if (const auto expected_size = usm_info_size_map.find(alloc_info);
        expected_size != usm_info_size_map.end()) {
        ASSERT_EQ(expected_size->second, size);
    }

    std::vector<uint8_t> info_data(size);
    ASSERT_SUCCESS(urUSMGetMemAllocInfo(context, ptr, alloc_info, size,
                                        info_data.data(), nullptr));
}

using urUSMGetMemAllocInfoNegativeTest = uur::urUSMDeviceAllocTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUSMGetMemAllocInfoNegativeTest);

TEST_P(urUSMGetMemAllocInfoNegativeTest, InvalidNullHandleContext) {
    ur_usm_type_t USMType;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urUSMGetMemAllocInfo(nullptr, ptr, UR_USM_ALLOC_INFO_TYPE,
                                          sizeof(ur_usm_type_t), &USMType,
                                          nullptr));
}

TEST_P(urUSMGetMemAllocInfoNegativeTest, InvalidNullPointerMem) {
    ur_usm_type_t USMType;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urUSMGetMemAllocInfo(context, nullptr, UR_USM_ALLOC_INFO_TYPE,
                             sizeof(ur_usm_type_t), &USMType, nullptr));
}

TEST_P(urUSMGetMemAllocInfoNegativeTest, InvalidEnumeration) {
    ur_usm_type_t USMType;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_ENUMERATION,
        urUSMGetMemAllocInfo(context, ptr, UR_USM_ALLOC_INFO_FORCE_UINT32,
                             sizeof(ur_usm_type_t), &USMType, nullptr));
}

TEST_P(urUSMGetMemAllocInfoNegativeTest, InvalidValuePropSize) {
    ur_usm_type_t USMType;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urUSMGetMemAllocInfo(context, ptr, UR_USM_ALLOC_INFO_TYPE,
                                          sizeof(ur_usm_type_t) - 1, &USMType,
                                          nullptr));
}
