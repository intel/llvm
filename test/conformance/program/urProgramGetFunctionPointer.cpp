// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/known_failure.h"
#include <uur/fixtures.h>

struct urProgramGetFunctionPointerTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
        auto kernel_names =
            uur::KernelsEnvironment::instance->GetEntryPointNames(
                this->program_name);
        function_name = kernel_names[0];
    }

    std::string function_name;
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramGetFunctionPointerTest);

TEST_P(urProgramGetFunctionPointerTest, Success) {
    UUR_KNOWN_FAILURE_ON(uur::OpenCL{"Intel(R) UHD Graphics 770"});
    void *function_pointer = nullptr;
    ASSERT_SUCCESS(urProgramGetFunctionPointer(
        device, program, function_name.data(), &function_pointer));
    ASSERT_NE(function_pointer, nullptr);
}

TEST_P(urProgramGetFunctionPointerTest, InvalidKernelName) {
    void *function_pointer = nullptr;
    std::string missing_function = "aFakeFunctionName";
    auto result = urProgramGetFunctionPointer(
        device, program, missing_function.data(), &function_pointer);
    ur_platform_backend_t backend;
    ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(backend), &backend, nullptr));
    // TODO: level zero backend incorrectly returns UR_RESULT_ERROR_UNSUPPORTED_FEATURE
    if (backend == UR_PLATFORM_BACKEND_LEVEL_ZERO) {
        ASSERT_EQ(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, result);
    } else {
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_KERNEL_NAME, result);
    }

    ASSERT_EQ(function_pointer, nullptr);
}

TEST_P(urProgramGetFunctionPointerTest, InvalidNullHandleDevice) {
    void *function_pointer = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramGetFunctionPointer(nullptr, program,
                                                 function_name.data(),
                                                 &function_pointer));
}

TEST_P(urProgramGetFunctionPointerTest, InvalidNullHandleProgram) {
    void *function_pointer = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramGetFunctionPointer(device, nullptr,
                                                 function_name.data(),
                                                 &function_pointer));
}

TEST_P(urProgramGetFunctionPointerTest, InvalidNullPointerFunctionName) {
    void *function_pointer = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramGetFunctionPointer(device, program, nullptr,
                                                 &function_pointer));
}

TEST_P(urProgramGetFunctionPointerTest, InvalidNullPointerFunctionPointer) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramGetFunctionPointer(
                         device, program, function_name.data(), nullptr));
}
