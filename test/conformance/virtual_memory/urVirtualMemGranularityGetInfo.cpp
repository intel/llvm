// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

struct urVirtualMemGranularityGetInfoTest
    : uur::urContextTestWithParam<ur_virtual_mem_granularity_info_t> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            urContextTestWithParam<ur_virtual_mem_granularity_info_t>::SetUp());
        ur_bool_t virtual_memory_support = false;
        ASSERT_SUCCESS(urDeviceGetInfo(
            this->device, UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT,
            sizeof(ur_bool_t), &virtual_memory_support, nullptr));
        if (!virtual_memory_support) {
            GTEST_SKIP() << "Virtual memory is not supported.";
        }
    }
};

UUR_TEST_SUITE_P(
    urVirtualMemGranularityGetInfoTest,
    ::testing::Values(UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
                      UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED),
    uur::deviceTestWithParamPrinter<ur_virtual_mem_granularity_info_t>);

TEST_P(urVirtualMemGranularityGetInfoTest, Success) {
    size_t size = 0;
    ur_virtual_mem_granularity_info_t info = getParam();
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urVirtualMemGranularityGetInfo(context, device, info, 0, nullptr,
                                       &size),
        info);
    ASSERT_NE(size, 0);

    std::vector<uint8_t> infoData(size);
    ASSERT_SUCCESS(urVirtualMemGranularityGetInfo(
        context, device, info, infoData.size(), infoData.data(), nullptr));

    switch (info) {
    case UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM: {
        ASSERT_EQ(size, sizeof(size_t));
        size_t minimum = *reinterpret_cast<size_t *>(infoData.data());
        ASSERT_GT(minimum, 0);
    } break;
    case UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED: {
        ASSERT_EQ(size, sizeof(size_t));
        size_t recommended = *reinterpret_cast<size_t *>(infoData.data());
        ASSERT_GT(recommended, 0);
    } break;
    default:
        FAIL() << "Unhandled ur_virtual_mem_granularity_info_t enumeration: "
               << info;
        break;
    }
}

struct urVirtualMemGranularityGetInfoNegativeTest : uur::urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());

        ur_bool_t virtual_memory_support = false;
        ASSERT_SUCCESS(urDeviceGetInfo(
            device, UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT, sizeof(ur_bool_t),
            &virtual_memory_support, nullptr));
        if (!virtual_memory_support) {
            GTEST_SKIP() << "Virtual memory is not supported.";
        }
    }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urVirtualMemGranularityGetInfoNegativeTest);

TEST_P(urVirtualMemGranularityGetInfoNegativeTest, InvalidNullHandleContext) {
    size_t size = 0;
    ASSERT_EQ_RESULT(
        urVirtualMemGranularityGetInfo(nullptr, device,
                                       UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
                                       0, nullptr, &size),
        UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urVirtualMemGranularityGetInfoNegativeTest, InvalidEnumeration) {
    size_t size = 0;
    ASSERT_EQ_RESULT(urVirtualMemGranularityGetInfo(
                         context, device,
                         UR_VIRTUAL_MEM_GRANULARITY_INFO_FORCE_UINT32, 0,
                         nullptr, &size),
                     UR_RESULT_ERROR_INVALID_ENUMERATION);
}

TEST_P(urVirtualMemGranularityGetInfoNegativeTest,
       InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(
        urVirtualMemGranularityGetInfo(context, device,
                                       UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
                                       0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urVirtualMemGranularityGetInfoNegativeTest,
       InvalidNullPointerPropValue) {
    ASSERT_EQ_RESULT(
        urVirtualMemGranularityGetInfo(context, device,
                                       UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
                                       sizeof(size_t), nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urVirtualMemGranularityGetInfoNegativeTest, InvalidPropSizeZero) {
    size_t minimum = 0;
    ASSERT_EQ_RESULT(
        urVirtualMemGranularityGetInfo(context, device,
                                       UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
                                       0, &minimum, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urVirtualMemGranularityGetInfoNegativeTest, InvalidSizePropSizeSmall) {
    size_t minimum = 0;
    ASSERT_EQ_RESULT(
        urVirtualMemGranularityGetInfo(context, device,
                                       UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
                                       sizeof(size_t) - 1, &minimum, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}
