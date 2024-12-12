// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_api.h"
#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urKernelGetInfoTest = uur::urKernelTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelGetInfoTest);

TEST_P(urKernelGetInfoTest, FunctionName) {
    auto property_name = UR_KERNEL_INFO_FUNCTION_NAME;
    size_t property_size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size),
        property_name);

    std::vector<char> property_value(property_size);
    ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                   property_value.data(), nullptr));
}

TEST_P(urKernelGetInfoTest, NumArgs) {
    auto property_name = UR_KERNEL_INFO_NUM_ARGS;
    size_t property_size = 0;

    ASSERT_SUCCESS(
        urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size));
    ASSERT_EQ(property_size, sizeof(uint32_t));

    std::vector<char> property_value(property_size);
    ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                   property_value.data(), nullptr));
}

TEST_P(urKernelGetInfoTest, ReferenceCount) {
    auto property_name = UR_KERNEL_INFO_REFERENCE_COUNT;
    size_t property_size = 0;

    ASSERT_SUCCESS(
        urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size));
    ASSERT_EQ(property_size, sizeof(uint32_t));

    std::vector<char> property_value(property_size);
    ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                   property_value.data(), nullptr));

    auto returned_reference_count =
        reinterpret_cast<uint32_t *>(property_value.data());
    ASSERT_GT(*returned_reference_count, 0U);
}

TEST_P(urKernelGetInfoTest, Context) {
    auto property_name = UR_KERNEL_INFO_CONTEXT;
    size_t property_size = 0;

    ASSERT_SUCCESS(
        urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size));
    ASSERT_EQ(property_size, sizeof(ur_context_handle_t));

    std::vector<char> property_value(property_size);
    ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                   property_value.data(), nullptr));

    auto returned_context =
        reinterpret_cast<ur_context_handle_t *>(property_value.data());
    ASSERT_EQ(context, *returned_context);
}

TEST_P(urKernelGetInfoTest, Program) {
    auto property_name = UR_KERNEL_INFO_PROGRAM;
    size_t property_size = 0;

    ASSERT_SUCCESS(
        urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size));
    ASSERT_EQ(property_size, sizeof(ur_program_handle_t));

    std::vector<char> property_value(property_size);
    ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                   property_value.data(), nullptr));

    auto returned_program =
        reinterpret_cast<ur_program_handle_t *>(property_value.data());
    ASSERT_EQ(program, *returned_program);
}

TEST_P(urKernelGetInfoTest, Attributes) {
    auto property_name = UR_KERNEL_INFO_ATTRIBUTES;
    size_t property_size = 0;

    ASSERT_SUCCESS(
        urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size));

    std::vector<char> property_value(property_size);
    ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                   property_value.data(), nullptr));

    auto returned_attributes = std::string(property_value.data());
    ur_platform_backend_t backend;
    ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(backend), &backend, nullptr));
    if (backend == UR_PLATFORM_BACKEND_OPENCL ||
        backend == UR_PLATFORM_BACKEND_LEVEL_ZERO) {
        // Older intel drivers don't attach any default attributes and newer
        // ones force walk order to X/Y/Z using special attribute.
        ASSERT_TRUE(returned_attributes.empty() ||
                    returned_attributes ==
                        "intel_reqd_workgroup_walk_order(0,1,2)");
    } else {
        ASSERT_TRUE(returned_attributes.empty());
    }
}

TEST_P(urKernelGetInfoTest, NumRegs) {
    UUR_KNOWN_FAILURE_ON(uur::HIP{}, uur::OpenCL{});
    auto property_name = UR_KERNEL_INFO_NUM_REGS;
    size_t property_size = 0;

    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size));
    ASSERT_EQ(property_size, sizeof(uint32_t));

    std::vector<char> property_value(property_size);
    ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                   property_value.data(), nullptr));
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

TEST_P(urKernelGetInfoTest, KernelNameCorrect) {
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

TEST_P(urKernelGetInfoTest, KernelContextCorrect) {
    ur_context_handle_t info_context;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_CONTEXT,
                                   sizeof(ur_context_handle_t), &info_context,
                                   nullptr));
    ASSERT_EQ(context, info_context);
}
