// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urKernelSetArgSamplerTestWithParam
    : uur::urBaseKernelTestWithParam<uur::SamplerCreateParamT> {
    void SetUp() {
        const auto param = getParam();
        const auto normalized = std::get<0>(param);
        const auto addr_mode = std::get<1>(param);
        const auto filter_mode = std::get<2>(param);

        ur_sampler_desc_t sampler_desc = {
            UR_STRUCTURE_TYPE_SAMPLER_DESC, /* sType */
            nullptr,                        /* pNext */
            normalized,                     /* normalizedCoords */
            addr_mode,                      /* addressingMode */
            filter_mode                     /* filterMode */
        };

        program_name = "image_copy";
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urBaseKernelTestWithParam<uur::SamplerCreateParamT>::SetUp());

        auto ret = urSamplerCreate(context, &sampler_desc, &sampler);
        if (ret == UR_RESULT_ERROR_UNSUPPORTED_FEATURE ||
            ret == UR_RESULT_ERROR_UNINITIALIZED) {
            GTEST_SKIP() << "urSamplerCreate not supported";
        } else {
            ASSERT_SUCCESS(ret);
        }

        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urBaseKernelTestWithParam<uur::SamplerCreateParamT>::Build());
    }

    void TearDown() {
        if (sampler) {
            ASSERT_SUCCESS(urSamplerRelease(sampler));
        }
        UUR_RETURN_ON_FATAL_FAILURE(uur::urBaseKernelTestWithParam<
                                    uur::SamplerCreateParamT>::TearDown());
    }

    ur_sampler_handle_t sampler = nullptr;
};

UUR_DEVICE_TEST_SUITE_P(
    urKernelSetArgSamplerTestWithParam,
    ::testing::Combine(
        ::testing::Values(true, false),
        ::testing::Values(UR_SAMPLER_ADDRESSING_MODE_NONE,
                          UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE,
                          UR_SAMPLER_ADDRESSING_MODE_CLAMP,
                          UR_SAMPLER_ADDRESSING_MODE_REPEAT,
                          UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT),
        ::testing::Values(UR_SAMPLER_FILTER_MODE_NEAREST,
                          UR_SAMPLER_FILTER_MODE_LINEAR)),
    uur::deviceTestWithParamPrinter<uur::SamplerCreateParamT>);

TEST_P(urKernelSetArgSamplerTestWithParam, Success) {
    uint32_t arg_index = 2;
    ASSERT_SUCCESS(urKernelSetArgSampler(kernel, arg_index, nullptr, sampler));
}

struct urKernelSetArgSamplerTest : uur::urBaseKernelTest {
    void SetUp() {
        program_name = "image_copy";
        UUR_RETURN_ON_FATAL_FAILURE(urBaseKernelTest::SetUp());

        ur_sampler_desc_t sampler_desc = {
            UR_STRUCTURE_TYPE_SAMPLER_DESC,   /* sType */
            nullptr,                          /* pNext */
            false,                            /* normalizedCoords */
            UR_SAMPLER_ADDRESSING_MODE_CLAMP, /* addressingMode */
            UR_SAMPLER_FILTER_MODE_NEAREST    /* filterMode */
        };

        auto ret = urSamplerCreate(context, &sampler_desc, &sampler);
        if (ret == UR_RESULT_ERROR_UNSUPPORTED_FEATURE ||
            ret == UR_RESULT_ERROR_UNINITIALIZED) {
            GTEST_SKIP() << "urSamplerCreate not supported";
        } else {
            ASSERT_SUCCESS(ret);
        }

        Build();
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

TEST_P(urKernelSetArgSamplerTest, SuccessWithProps) {
    ur_kernel_arg_sampler_properties_t props{
        UR_STRUCTURE_TYPE_KERNEL_ARG_SAMPLER_PROPERTIES, nullptr};
    size_t arg_index = 2;
    ASSERT_SUCCESS(urKernelSetArgSampler(kernel, arg_index, &props, sampler));
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
    uint32_t num_kernel_args = 0;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                   sizeof(num_kernel_args), &num_kernel_args,
                                   nullptr));
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX,
        urKernelSetArgSampler(kernel, num_kernel_args + 1, nullptr, sampler));
}
