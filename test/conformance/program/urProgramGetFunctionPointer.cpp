// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

struct urProgramGetFunctionPointerTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
        std::string kernel_list;
        size_t kernel_list_size = 0;
        ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_KERNEL_NAMES,
                                        0, nullptr, &kernel_list_size));
        kernel_list.resize(kernel_list_size);
        ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_KERNEL_NAMES,
                                        kernel_list_size, kernel_list.data(),
                                        nullptr));
        function_name = kernel_list.substr(0, kernel_list.find(";"));
    }

    std::string function_name;
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramGetFunctionPointerTest);

TEST_P(urProgramGetFunctionPointerTest, Success) {
    void *function_pointer = nullptr;
    ASSERT_SUCCESS(urProgramGetFunctionPointer(
        device, program, function_name.data(), &function_pointer));
    ASSERT_NE(function_pointer, nullptr);
}

TEST_P(urProgramGetFunctionPointerTest, SuccessFunctionNotFound) {
    void *function_pointer = nullptr;
    std::string missing_function = "aFakeFunctionName";
    ASSERT_SUCCESS(urProgramGetFunctionPointer(
        device, program, missing_function.data(), &function_pointer));
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
