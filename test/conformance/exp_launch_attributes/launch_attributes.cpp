// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urEnqueueKernelLaunchCustomTest : uur::urKernelExecutionTest {
    void SetUp() override {
        program_name = "fill";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
    }

    uint32_t val = 42;
    size_t global_size = 32;
    size_t global_offset = 0;
    size_t n_dimensions = 1;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueKernelLaunchCustomTest);

TEST_P(urEnqueueKernelLaunchCustomTest, Success) {
    ur_mem_handle_t buffer = nullptr;
    AddBuffer1DArg(sizeof(val) * global_size, &buffer);
    AddPodArg(val);
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         &global_offset, &global_size, nullptr,
                                         0, nullptr, nullptr));

ur_exp_launch_attribute_t attr[1];
attr[0].id = UR_EXP_LAUNCH_ATTRIBUTE_ID_CLUSTER_DIMENSION;
uint32_t dims[3] = {1024, 1, 1};
attr[0].value.clusterDim = dims;
size_t LocalWorkSize = 5;
ASSERT_SUCCESS(urEnqueueKernelLaunchCustomExp(queue, kernel, n_dimensions,
                                          &global_size, &LocalWorkSize, 1, attr,
                                         0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    ValidateBuffer(buffer, sizeof(val) * global_size, val);
}

