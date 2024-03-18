// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urKernelGetSuggestedLocalWorkSizeTest : uur::urKernelExecutionTest {
    void SetUp() override {
        program_name = "fill";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
    }

    uint32_t val = 42;
    size_t global_size = 32;
    size_t global_offset = 0;
    size_t n_dimensions = 1;

    size_t suggested_local_work_size;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelGetSuggestedLocalWorkSizeTest);

TEST_P(urKernelGetSuggestedLocalWorkSizeTest, Success) {
    ur_mem_handle_t buffer = nullptr;
    AddBuffer1DArg(sizeof(val) * global_size, &buffer);
    AddPodArg(val);

    suggested_local_work_size = SIZE_MAX;
    ASSERT_SUCCESS(urKernelGetSuggestedLocalWorkSize(
        kernel, queue, n_dimensions, &global_offset, &global_size,
        &suggested_local_work_size));
    ASSERT_LE(suggested_local_work_size, global_size);
}

TEST_P(urKernelGetSuggestedLocalWorkSizeTest, InvalidNullHandleKernel) {
    ASSERT_EQ_RESULT(urKernelGetSuggestedLocalWorkSize(
                         nullptr, queue, n_dimensions, &global_offset,
                         &global_size, &suggested_local_work_size),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidNullHandleQueue) {
    // LATER: some adapter's implementation of the API does not require queue, 
    // do we still need to test for it?
    ASSERT_EQ_RESULT(urKernelGetSuggestedLocalWorkSize(
                         kernel, nullptr, n_dimensions, &global_offset,
                         &global_size, &suggested_local_work_size),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidWorkDimension) {
    uint32_t max_work_item_dimensions = 0;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
        sizeof(max_work_item_dimensions), &max_work_item_dimensions, nullptr));
    ASSERT_EQ_RESULT(urKernelGetSuggestedLocalWorkSize(
                         kernel, queue, n_dimensions, &global_offset,
                         &global_size, &suggested_local_work_size),
                     UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
}
