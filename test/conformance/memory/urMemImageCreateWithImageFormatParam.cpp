// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>
#include <vector>

static ur_image_desc_t image_desc{
    UR_STRUCTURE_TYPE_IMAGE_DESC, ///< [in] type of this structure
    nullptr, ///< [in][optional] pointer to extension-specific structure
    UR_MEM_TYPE_IMAGE3D, ///< [in] memory object type
    1,                   ///< [in] image width
    1,                   ///< [in] image height
    1,                   ///< [in] image depth
    1,                   ///< [in] image array size
    0,                   ///< [in] image row pitch
    0,                   ///< [in] image slice pitch
    0,                   ///< [in] number of MIP levels
    0                    ///< [in] number of samples
};

const std::vector<ur_image_format_t> primary_image_formats = {
    {UR_IMAGE_CHANNEL_ORDER_RGBA, UR_IMAGE_CHANNEL_TYPE_UNORM_INT8},
    {UR_IMAGE_CHANNEL_ORDER_RGBA, UR_IMAGE_CHANNEL_TYPE_UNORM_INT16},
    {UR_IMAGE_CHANNEL_ORDER_RGBA, UR_IMAGE_CHANNEL_TYPE_SNORM_INT8},
    {UR_IMAGE_CHANNEL_ORDER_RGBA, UR_IMAGE_CHANNEL_TYPE_SNORM_INT16},
    {UR_IMAGE_CHANNEL_ORDER_RGBA, UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8},
    {UR_IMAGE_CHANNEL_ORDER_RGBA, UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16},
    {UR_IMAGE_CHANNEL_ORDER_RGBA, UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32},
    {UR_IMAGE_CHANNEL_ORDER_RGBA, UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8},
    {UR_IMAGE_CHANNEL_ORDER_RGBA, UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16},
    {UR_IMAGE_CHANNEL_ORDER_RGBA, UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32},
    {UR_IMAGE_CHANNEL_ORDER_RGBA, UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT},
    {UR_IMAGE_CHANNEL_ORDER_RGBA, UR_IMAGE_CHANNEL_TYPE_FLOAT}};

const std::vector<ur_image_channel_order_t> channel_orders = {
    UR_IMAGE_CHANNEL_ORDER_A,         UR_IMAGE_CHANNEL_ORDER_R,
    UR_IMAGE_CHANNEL_ORDER_RG,        UR_IMAGE_CHANNEL_ORDER_RA,
    UR_IMAGE_CHANNEL_ORDER_RGB,       UR_IMAGE_CHANNEL_ORDER_RGBA,
    UR_IMAGE_CHANNEL_ORDER_BGRA,      UR_IMAGE_CHANNEL_ORDER_ARGB,
    UR_IMAGE_CHANNEL_ORDER_ABGR,      UR_IMAGE_CHANNEL_ORDER_INTENSITY,
    UR_IMAGE_CHANNEL_ORDER_LUMINANCE, UR_IMAGE_CHANNEL_ORDER_RX,
    UR_IMAGE_CHANNEL_ORDER_RGX,       UR_IMAGE_CHANNEL_ORDER_RGBX,
    UR_IMAGE_CHANNEL_ORDER_SRGBA};

const std::vector<ur_image_channel_type_t> channel_types = {
    UR_IMAGE_CHANNEL_TYPE_SNORM_INT8,
    UR_IMAGE_CHANNEL_TYPE_SNORM_INT16,
    UR_IMAGE_CHANNEL_TYPE_UNORM_INT8,
    UR_IMAGE_CHANNEL_TYPE_UNORM_INT16,
    UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565,
    UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555,
    UR_IMAGE_CHANNEL_TYPE_INT_101010,
    UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8,
    UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16,
    UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32,
    UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8,
    UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16,
    UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32,
    UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT,
    UR_IMAGE_CHANNEL_TYPE_FLOAT};

std::vector<ur_image_format_t> all_image_formats;

struct urMemImageCreateTestWithImageFormatParam
    : uur::urContextTestWithParam<ur_image_format_t> {
    void SetUp() {
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urContextTestWithParam<ur_image_format_t>::SetUp());
    }
    void TearDown() {
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urContextTestWithParam<ur_image_format_t>::TearDown());
    }

    static std::vector<ur_image_format_t> makeImageFormats() {
        for (auto channel_order : channel_orders) {
            for (auto channel_type : channel_types) {
                all_image_formats.push_back({channel_order, channel_type});
            }
        }
        return all_image_formats;
    }
};

UUR_TEST_SUITE_P(
    urMemImageCreateTestWithImageFormatParam,
    ::testing::ValuesIn(
        urMemImageCreateTestWithImageFormatParam::makeImageFormats()),
    uur::deviceTestWithParamPrinter<ur_image_format_t>);

TEST_P(urMemImageCreateTestWithImageFormatParam, Success) {
    ur_image_channel_order_t channel_order =
        std::get<1>(GetParam()).channelOrder;
    ur_image_channel_type_t channel_type = std::get<1>(GetParam()).channelType;

    ur_image_format_t image_format{channel_order, channel_type};

    ur_mem_handle_t image_handle = nullptr;
    ur_result_t res =
        urMemImageCreate(context, UR_MEM_FLAG_READ_WRITE, &image_format,
                         &image_desc, nullptr, &image_handle);

    bool is_primary_image_format = false;
    for (auto primary_image_format : primary_image_formats) {
        if (primary_image_format.channelOrder == image_format.channelOrder &&
            primary_image_format.channelType == image_format.channelType) {
            is_primary_image_format = true;
            break;
        }
    }

    if (res == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
        GTEST_SKIP() << "urMemImageCreate not supported";
    }

    if (!is_primary_image_format &&
        res == UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT) {
        GTEST_SKIP();
    }
    ASSERT_SUCCESS(res);
    ASSERT_NE(nullptr, image_handle);
    ASSERT_SUCCESS(urMemRelease(image_handle));
}
