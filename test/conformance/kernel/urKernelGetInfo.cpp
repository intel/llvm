// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urKernelGetInfoTest = uur::urKernelTestWithParam<ur_kernel_info_t>;

UUR_TEST_SUITE_P(
    urKernelGetInfoTest,
    ::testing::Values(UR_KERNEL_INFO_FUNCTION_NAME, UR_KERNEL_INFO_NUM_ARGS,
                      UR_KERNEL_INFO_REFERENCE_COUNT, UR_KERNEL_INFO_CONTEXT,
                      UR_KERNEL_INFO_PROGRAM, UR_KERNEL_INFO_ATTRIBUTES,
                      UR_KERNEL_INFO_NUM_REGS),
    uur::deviceTestWithParamPrinter<ur_kernel_info_t>);

using urKernelGetInfoSingleTest = uur::urKernelExecutionTest;
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelGetInfoSingleTest);

TEST_P(urKernelGetInfoTest, Success) {
    auto property_name = getParam();
    size_t property_size = 0;
    std::vector<char> property_value;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size),
        property_name);
    property_value.resize(property_size);
    ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                   property_value.data(), nullptr));
    switch (property_name) {
    case UR_KERNEL_INFO_CONTEXT: {
        auto returned_context =
            reinterpret_cast<ur_context_handle_t *>(property_value.data());
        ASSERT_EQ(context, *returned_context);
        break;
    }
    case UR_KERNEL_INFO_PROGRAM: {
        auto returned_program =
            reinterpret_cast<ur_program_handle_t *>(property_value.data());
        ASSERT_EQ(program, *returned_program);
        break;
    }
    case UR_KERNEL_INFO_REFERENCE_COUNT: {
        auto returned_reference_count =
            reinterpret_cast<uint32_t *>(property_value.data());
        ASSERT_GT(*returned_reference_count, 0U);
        break;
    }
    case UR_KERNEL_INFO_ATTRIBUTES: {
        auto returned_attributes = std::string(property_value.data());
        ur_platform_backend_t backend;
        ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                         sizeof(backend), &backend, nullptr));
        if (backend == UR_PLATFORM_BACKEND_OPENCL ||
            backend == UR_PLATFORM_BACKEND_LEVEL_ZERO) {
            // Older intel drivers don't attach any default attributes and newer ones force walk order to X/Y/Z using special attribute.
            ASSERT_TRUE(returned_attributes.empty() ||
                        returned_attributes ==
                            "intel_reqd_workgroup_walk_order(0,1,2)");
        } else {
            ASSERT_TRUE(returned_attributes.empty());
        }
        break;
    }
    default:
        break;
    }
}

TEST_P(urKernelGetInfoTest, InvalidNullHandleKernel) {
    size_t kernel_name_length = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelGetInfo(nullptr, UR_KERNEL_INFO_FUNCTION_NAME, 0,
                                     nullptr, &kernel_name_length));
}

TEST_P(urKernelGetInfoTest, InvalidEnumeration) {
    size_t bad_enum_length = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urKernelGetInfo(kernel, UR_KERNEL_INFO_FORCE_UINT32, 0,
                                     nullptr, &bad_enum_length));
}

TEST_P(urKernelGetInfoTest, InvalidSizeZero) {
    size_t query_size = 0;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS, 0, nullptr,
                                   &query_size));
    std::vector<char> query_data(query_size);
    ASSERT_EQ_RESULT(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS, 0,
                                     query_data.data(), nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urKernelGetInfoTest, InvalidSizeSmall) {
    size_t query_size = 0;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS, 0, nullptr,
                                   &query_size));
    std::vector<char> query_data(query_size);
    ASSERT_EQ_RESULT(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                     query_data.size() - 1, query_data.data(),
                                     nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urKernelGetInfoTest, InvalidNullPointerPropValue) {
    size_t query_size = 0;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS, 0, nullptr,
                                   &query_size));
    ASSERT_EQ_RESULT(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                     query_size, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urKernelGetInfoTest, InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(
        urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urKernelGetInfoSingleTest, KernelNameCorrect) {
    size_t name_size = 0;
    std::vector<char> name_data;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_FUNCTION_NAME, 0,
                                   nullptr, &name_size));
    name_data.resize(name_size);
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_FUNCTION_NAME,
                                   name_size, name_data.data(), nullptr));
    ASSERT_EQ(name_data[name_size - 1], '\0');
    ASSERT_STREQ(kernel_name.c_str(), name_data.data());
}

TEST_P(urKernelGetInfoSingleTest, KernelContextCorrect) {
    ur_context_handle_t info_context;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_CONTEXT,
                                   sizeof(ur_context_handle_t), &info_context,
                                   nullptr));
    ASSERT_EQ(context, info_context);
}
