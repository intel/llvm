// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <array>
#include <map>
#include <uur/fixtures.h>

using urMemGetInfoTestWithParam = uur::urMemBufferTestWithParam<ur_mem_info_t>;

static constexpr std::array<ur_mem_info_t, 3> mem_info_values{
    UR_MEM_INFO_SIZE, UR_MEM_INFO_CONTEXT, UR_MEM_INFO_REFERENCE_COUNT};
static std::unordered_map<ur_mem_info_t, size_t> mem_info_size_map = {
    {UR_MEM_INFO_SIZE, sizeof(size_t)},
    {UR_MEM_INFO_CONTEXT, sizeof(ur_context_handle_t)},
    {UR_MEM_INFO_REFERENCE_COUNT, sizeof(uint32_t)},
};

UUR_TEST_SUITE_P(urMemGetInfoTestWithParam,
                 ::testing::ValuesIn(mem_info_values),
                 uur::deviceTestWithParamPrinter<ur_mem_info_t>);

TEST_P(urMemGetInfoTestWithParam, Success) {
    ur_mem_info_t info = getParam();
    size_t size;
    ASSERT_SUCCESS(urMemGetInfo(buffer, info, 0, nullptr, &size));
    ASSERT_NE(size, 0);

    if (const auto expected_size = mem_info_size_map.find(info);
        expected_size != mem_info_size_map.end()) {
        ASSERT_EQ(expected_size->second, size);
    }

    std::vector<uint8_t> info_data(size);
    ASSERT_SUCCESS(urMemGetInfo(buffer, info, size, info_data.data(), nullptr));

    switch (info) {
    case UR_MEM_INFO_CONTEXT: {
        auto returned_context =
            reinterpret_cast<ur_context_handle_t *>(info_data.data());
        ASSERT_EQ(context, *returned_context);
        break;
    }
    case UR_MEM_INFO_SIZE: {
        auto returned_size = reinterpret_cast<size_t *>(info_data.data());
        ASSERT_GE(*returned_size, allocation_size);
        break;
    }
    case UR_MEM_INFO_REFERENCE_COUNT: {
        const size_t ReferenceCount =
            *reinterpret_cast<const uint32_t *>(info_data.data());
        ASSERT_GT(ReferenceCount, 0);
        break;
    }
    default:
        break;
    }
}

using urMemGetInfoTest = uur::urMemBufferTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemGetInfoTest);

TEST_P(urMemGetInfoTest, InvalidNullHandleMemory) {
    size_t mem_size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urMemGetInfo(nullptr, UR_MEM_INFO_SIZE, sizeof(size_t),
                                  &mem_size, nullptr));
}

TEST_P(urMemGetInfoTest, InvalidEnumerationMemInfoType) {
    size_t mem_size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urMemGetInfo(buffer, UR_MEM_INFO_FORCE_UINT32,
                                  sizeof(size_t), &mem_size, nullptr));
}

TEST_P(urMemGetInfoTest, InvalidSizeZero) {
    size_t mem_size = 0;
    ASSERT_EQ_RESULT(
        urMemGetInfo(buffer, UR_MEM_INFO_SIZE, 0, &mem_size, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urMemGetInfoTest, InvalidSizeSmall) {
    size_t mem_size = 0;
    ASSERT_EQ_RESULT(urMemGetInfo(buffer, UR_MEM_INFO_SIZE,
                                  sizeof(mem_size) - 1, &mem_size, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urMemGetInfoTest, InvalidNullPointerParamValue) {
    size_t mem_size = 0;
    ASSERT_EQ_RESULT(urMemGetInfo(buffer, UR_MEM_INFO_SIZE, sizeof(mem_size),
                                  nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urMemGetInfoTest, InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(
        urMemGetInfo(buffer, UR_MEM_INFO_SIZE, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

using urMemGetInfoImageTest = uur::urMemImageTestWithParam<ur_mem_info_t>;
UUR_TEST_SUITE_P(urMemGetInfoImageTest, ::testing::ValuesIn(mem_info_values),
                 uur::deviceTestWithParamPrinter<ur_mem_info_t>);

TEST_P(urMemGetInfoImageTest, Success) {
    ur_mem_info_t info = getParam();
    size_t size;
    ASSERT_SUCCESS(urMemGetInfo(image, info, 0, nullptr, &size));
    ASSERT_NE(size, 0);

    if (const auto expected_size = mem_info_size_map.find(info);
        expected_size != mem_info_size_map.end()) {
        ASSERT_EQ(expected_size->second, size);
    }

    std::vector<uint8_t> info_data(size);
    ASSERT_SUCCESS(urMemGetInfo(image, info, size, info_data.data(), nullptr));

    switch (info) {
    case UR_MEM_INFO_SIZE: {
        const size_t ExpectedPixelSize = sizeof(float) * 4 /*NumChannels*/;
        const size_t ExpectedImageSize = ExpectedPixelSize * desc.arraySize *
                                         desc.width * desc.height * desc.depth;
        const size_t ImageSizeBytes =
            *reinterpret_cast<const size_t *>(info_data.data());
        ASSERT_EQ(ImageSizeBytes, ExpectedImageSize);
        break;
    }
    case UR_MEM_INFO_CONTEXT: {
        ur_context_handle_t InfoContext =
            *reinterpret_cast<ur_context_handle_t *>(info_data.data());
        ASSERT_EQ(InfoContext, context);
        break;
    }
    case UR_MEM_INFO_REFERENCE_COUNT: {
        const size_t ReferenceCount =
            *reinterpret_cast<const uint32_t *>(info_data.data());
        ASSERT_GT(ReferenceCount, 0);
        break;
    }

    default:
        break;
    }
}
