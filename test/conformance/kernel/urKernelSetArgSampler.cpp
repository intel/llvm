// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urKernelSetArgSamplerTest : uur::urBaseKernelTest {
    void SetUp() {
        program_name = "image_copy";
        UUR_RETURN_ON_FATAL_FAILURE(urBaseKernelTest::SetUp());
        // Images and samplers are not available on AMD
        ur_platform_backend_t backend;
        ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                         sizeof(backend), &backend, nullptr));
        if (backend == UR_PLATFORM_BACKEND_HIP) {
            GTEST_SKIP() << "Sampler are not supported on hip.";
        }
        Build();
        ur_sampler_desc_t sampler_desc = {
            UR_STRUCTURE_TYPE_SAMPLER_DESC,   /* sType */
            nullptr,                          /* pNext */
            false,                            /* normalizedCoords */
            UR_SAMPLER_ADDRESSING_MODE_CLAMP, /* addressingMode */
            UR_SAMPLER_FILTER_MODE_NEAREST    /* filterMode */
        };
        ASSERT_SUCCESS(urSamplerCreate(context, &sampler_desc, &sampler));
    }

    void TearDown() {
        if (sampler) {
            ASSERT_SUCCESS(urSamplerRelease(sampler));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urBaseKernelTest::TearDown());
    }

    ur_sampler_handle_t sampler = nullptr;
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelSetArgSamplerTest);

TEST_P(urKernelSetArgSamplerTest, Success) {
    ASSERT_SUCCESS(urKernelSetArgSampler(kernel, 2, nullptr, sampler));
}

TEST_P(urKernelSetArgSamplerTest, InvalidNullHandleKernel) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelSetArgSampler(nullptr, 2, nullptr, sampler));
}

TEST_P(urKernelSetArgSamplerTest, InvalidNullHandleArgValue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelSetArgSampler(kernel, 2, nullptr, nullptr));
}

TEST_P(urKernelSetArgSamplerTest, InvalidKernelArgumentIndex) {
    size_t num_kernel_args = 0;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                   sizeof(num_kernel_args), &num_kernel_args,
                                   nullptr));
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX,
        urKernelSetArgSampler(kernel, num_kernel_args + 1, nullptr, sampler));
}
