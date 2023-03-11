// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urUSMAllocInfoTest =
    uur::urUSMDeviceAllocTestWithParam<ur_usm_alloc_info_t>;

UUR_TEST_SUITE_P(urUSMAllocInfoTest,
                 ::testing::Values(UR_USM_ALLOC_INFO_TYPE,
                                   UR_USM_ALLOC_INFO_BASE_PTR,
                                   UR_USM_ALLOC_INFO_SIZE,
                                   UR_USM_ALLOC_INFO_DEVICE,
                                   UR_USM_ALLOC_INFO_POOL),
                 uur::deviceTestWithParamPrinter<ur_usm_alloc_info_t>);

TEST_P(urUSMAllocInfoTest, Success) {
    size_t size = 0;
    auto alloc_info = getParam();
    ASSERT_SUCCESS(
        urUSMGetMemAllocInfo(context, ptr, alloc_info, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    std::vector<uint8_t> info_data(size);
    ASSERT_SUCCESS(urUSMGetMemAllocInfo(context, ptr, alloc_info, size,
                                        info_data.data(), nullptr));
}

using urUSMGetMemAllocInfoTest = uur::urUSMDeviceAllocTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUSMGetMemAllocInfoTest);

TEST_P(urUSMGetMemAllocInfoTest, InvalidNullHandleContext) {
    ur_usm_type_t USMType;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urUSMGetMemAllocInfo(nullptr, ptr, UR_USM_ALLOC_INFO_TYPE,
                                          sizeof(ur_usm_type_t), &USMType,
                                          nullptr));
}

TEST_P(urUSMGetMemAllocInfoTest, InvalidNullPointerMem) {
    ur_usm_type_t USMType;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urUSMGetMemAllocInfo(context, nullptr, UR_USM_ALLOC_INFO_TYPE,
                             sizeof(ur_usm_type_t), &USMType, nullptr));
}

TEST_P(urUSMGetMemAllocInfoTest, InvalidEnumeration) {
    ur_usm_type_t USMType;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_ENUMERATION,
        urUSMGetMemAllocInfo(context, ptr, UR_USM_ALLOC_INFO_FORCE_UINT32,
                             sizeof(ur_usm_type_t), &USMType, nullptr));
}

TEST_P(urUSMGetMemAllocInfoTest, InvalidValuePropValueSize) {
    ur_usm_type_t USMType;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                     urUSMGetMemAllocInfo(context, ptr, UR_USM_ALLOC_INFO_TYPE,
                                          sizeof(ur_usm_type_t) - 1, &USMType,
                                          nullptr));
}
