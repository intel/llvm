// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

struct urKernelSetArgMemObjTest : uur::urKernelTest {
    void SetUp() {
        program_name = "fill";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                         16 * sizeof(uint32_t), nullptr,
                                         &buffer));
    }

    void TearDown() {
        if (buffer) {
            ASSERT_SUCCESS(urMemRelease(buffer));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::TearDown());
    }

    ur_mem_handle_t buffer;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelSetArgMemObjTest);

TEST_P(urKernelSetArgMemObjTest, Success) {
    ASSERT_SUCCESS(urKernelSetArgMemObj(kernel, 0, buffer));
}

TEST_P(urKernelSetArgMemObjTest, InvalidNullHandleKernel) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelSetArgMemObj(nullptr, 0, buffer));
}

TEST_P(urKernelSetArgMemObjTest, InvalidKernelArgumentIndex) {
    size_t num_kernel_args = 0;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                   sizeof(num_kernel_args), &num_kernel_args,
                                   nullptr));
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX,
                     urKernelSetArgMemObj(kernel, num_kernel_args + 1, buffer));
}

TEST_P(urKernelSetArgMemObjTest, InvalidNullPointer) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urKernelSetArgMemObj(kernel, 0, nullptr));
}
