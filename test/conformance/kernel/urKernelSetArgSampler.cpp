// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

struct urKernelSetArgSamplerTest : uur::urKernelTest {
    void SetUp() {
        program_name = "image_copy";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
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
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::TearDown());
    }

    ur_sampler_handle_t sampler = nullptr;
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelSetArgSamplerTest);

TEST_P(urKernelSetArgSamplerTest, Success) {
    ASSERT_SUCCESS(urKernelSetArgSampler(kernel, 2, sampler));
}

TEST_P(urKernelSetArgSamplerTest, InvalidNullHandleKernel) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelSetArgSampler(nullptr, 2, sampler));
}

TEST_P(urKernelSetArgSamplerTest, InvalidNullHandleArgValue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelSetArgSampler(kernel, 2, nullptr));
}

TEST_P(urKernelSetArgSamplerTest, InvalidKernelArgumentIndex) {
    size_t num_kernel_args = 0;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                   sizeof(num_kernel_args), &num_kernel_args,
                                   nullptr));
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX,
        urKernelSetArgSampler(kernel, num_kernel_args + 1, sampler));
}
