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
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urUSMGetMemAllocInfo(context, ptr, alloc_info, 0, nullptr, &size),
        alloc_info);
    ASSERT_NE(size, 0);

    if (const auto expected_size = usm_info_size_map.find(alloc_info);
        expected_size != usm_info_size_map.end()) {
        ASSERT_EQ(expected_size->second, size);
    }

    std::vector<uint8_t> info_data(size);
    ASSERT_SUCCESS(urUSMGetMemAllocInfo(context, ptr, alloc_info, size,
                                        info_data.data(), nullptr));
    switch (alloc_info) {
    case UR_USM_ALLOC_INFO_DEVICE: {
        auto returned_device =
            reinterpret_cast<ur_device_handle_t *>(info_data.data());
        ASSERT_EQ(*returned_device, device);
        break;
    }
    case UR_USM_ALLOC_INFO_SIZE: {
        auto returned_size = reinterpret_cast<size_t *>(info_data.data());
        ASSERT_GE(*returned_size, allocation_size);
        break;
    }
    case UR_USM_ALLOC_INFO_BASE_PTR: {
        auto returned_ptr = reinterpret_cast<void **>(info_data.data());
        ASSERT_EQ(*returned_ptr, ptr);
        break;
    }
    case UR_USM_ALLOC_INFO_POOL: {
        auto returned_pool =
            reinterpret_cast<ur_usm_pool_handle_t *>(info_data.data());
        ASSERT_EQ(*returned_pool, pool);
        break;
    }
    case UR_USM_ALLOC_INFO_TYPE: {
        auto returned_type =
            reinterpret_cast<ur_usm_type_t *>(info_data.data());
        ASSERT_EQ(*returned_type, UR_USM_TYPE_DEVICE);
        break;
    }
    default:
        break;
    }
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
