// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urMemImageGetInfoTest = uur::urMemImageTestWithParam<ur_image_info_t>;

static std::unordered_map<ur_image_info_t, size_t> image_info_size_map = {
    {UR_IMAGE_INFO_FORMAT, sizeof(ur_image_format_t)},
    {UR_IMAGE_INFO_ELEMENT_SIZE, sizeof(size_t)},
    {UR_IMAGE_INFO_ROW_PITCH, sizeof(size_t)},
    {UR_IMAGE_INFO_SLICE_PITCH, sizeof(size_t)},
    {UR_IMAGE_INFO_WIDTH, sizeof(size_t)},
    {UR_IMAGE_INFO_HEIGHT, sizeof(size_t)},
    {UR_IMAGE_INFO_DEPTH, sizeof(size_t)},
};

UUR_TEST_SUITE_P(urMemImageGetInfoTest,
                 ::testing::Values(UR_IMAGE_INFO_FORMAT,
                                   UR_IMAGE_INFO_ELEMENT_SIZE,
                                   UR_IMAGE_INFO_ROW_PITCH,
                                   UR_IMAGE_INFO_SLICE_PITCH,
                                   UR_IMAGE_INFO_WIDTH, UR_IMAGE_INFO_HEIGHT,
                                   UR_IMAGE_INFO_DEPTH),
                 uur::deviceTestWithParamPrinter<ur_image_info_t>);

TEST_P(urMemImageGetInfoTest, Success) {
    ur_image_info_t info = getParam();
    size_t size = 0;
    ASSERT_SUCCESS(urMemImageGetInfo(image, info, 0, nullptr, &size));
    ASSERT_NE(size, 0);

    if (const auto expected_size = image_info_size_map.find(info);
        expected_size != image_info_size_map.end()) {
        ASSERT_EQ(expected_size->second, size);
    } else {
        FAIL() << "Missing info value in image info size map";
    }

    std::vector<uint8_t> info_data(size);
    ASSERT_SUCCESS(
        urMemImageGetInfo(image, info, size, info_data.data(), nullptr));
}

TEST_P(urMemImageGetInfoTest, InvalidNullHandleImage) {
    size_t info_size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urMemImageGetInfo(nullptr, UR_IMAGE_INFO_FORMAT,
                                       sizeof(size_t), &info_size, nullptr));
}

TEST_P(urMemImageGetInfoTest, InvalidEnumerationImageInfoType) {
    size_t info_size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urMemImageGetInfo(image, UR_IMAGE_INFO_FORCE_UINT32,
                                       sizeof(size_t), &info_size, nullptr));
}

TEST_P(urMemImageGetInfoTest, InvalidSizeZero) {
    size_t info_size = 0;
    ASSERT_EQ_RESULT(
        urMemImageGetInfo(image, UR_IMAGE_INFO_FORMAT, 0, &info_size, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urMemImageGetInfoTest, InvalidSizeSmall) {
    int info_size = 0;
    ASSERT_EQ_RESULT(urMemImageGetInfo(image, UR_IMAGE_INFO_FORMAT,
                                       sizeof(info_size) - 1, &info_size,
                                       nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urMemImageGetInfoTest, InvalidNullPointerParamValue) {
    size_t info_size = 0;
    ASSERT_EQ_RESULT(urMemImageGetInfo(image, UR_IMAGE_INFO_FORMAT,
                                       sizeof(info_size), nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urMemImageGetInfoTest, InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(
        urMemImageGetInfo(image, UR_IMAGE_INFO_FORMAT, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
