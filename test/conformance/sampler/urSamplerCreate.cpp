// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"
#include "uur/raii.h"

using SamplerCreateParamTy =
    std::tuple<bool, ur_sampler_addressing_mode_t, ur_sampler_filter_mode_t>;

struct urSamplerCreateTestWithParam
    : uur::urContextTestWithParam<SamplerCreateParamTy> {};

namespace uur {
template <>
std::string deviceTestWithParamPrinter(
    const ::testing::TestParamInfo<
        std::tuple<ur_device_handle_t, SamplerCreateParamTy>> &info) {
    auto device = std::get<0>(info.param);
    auto param = std::get<1>(info.param);

    const auto normalized = std::get<0>(param);
    const auto addr_mode = std::get<1>(param);
    const auto filter_mode = std::get<2>(param);

    std::stringstream ss;
    ss << normalized << "_" << addr_mode << "_" << filter_mode;
    return uur::GetPlatformAndDeviceName(device) + "__" + ss.str();
}
} // namespace uur
UUR_TEST_SUITE_P(
    urSamplerCreateTestWithParam,
    ::testing::Combine(
        ::testing::Values(true, false),
        ::testing::Values(UR_SAMPLER_ADDRESSING_MODE_NONE,
                          UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE,
                          UR_SAMPLER_ADDRESSING_MODE_CLAMP,
                          UR_SAMPLER_ADDRESSING_MODE_REPEAT,
                          UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT),
        ::testing::Values(UR_SAMPLER_FILTER_MODE_NEAREST,
                          UR_SAMPLER_FILTER_MODE_LINEAR)),
    uur::deviceTestWithParamPrinter<SamplerCreateParamTy>);

TEST_P(urSamplerCreateTestWithParam, Success) {

    const auto param = getParam();
    const auto normalized = std::get<0>(param);
    const auto addr_mode = std::get<1>(param);
    const auto filter_mode = std::get<2>(param);

    ur_sampler_desc_t sampler_desc{
        UR_STRUCTURE_TYPE_SAMPLER_DESC, /* stype */
        nullptr,                        /* pNext */
        normalized,                     /* normalizedCoords */
        addr_mode,                      /* addressing mode */
        filter_mode,                    /* filterMode */
    };

    uur::raii::Sampler hSampler = nullptr;
    ASSERT_SUCCESS(urSamplerCreate(context, &sampler_desc, hSampler.ptr()));
    ASSERT_NE(hSampler, nullptr);
}

using urSamplerCreateTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urSamplerCreateTest);

TEST_P(urSamplerCreateTest, InvalidNullHandleContext) {
    ur_sampler_desc_t sampler_desc{
        UR_STRUCTURE_TYPE_SAMPLER_DESC,   /* stype */
        nullptr,                          /* pNext */
        true,                             /* normalizedCoords */
        UR_SAMPLER_ADDRESSING_MODE_CLAMP, /* addressing mode */
        UR_SAMPLER_FILTER_MODE_LINEAR,    /* filterMode */
    };
    uur::raii::Sampler hSampler = nullptr;
    ASSERT_EQ_RESULT(urSamplerCreate(nullptr, &sampler_desc, hSampler.ptr()),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urSamplerCreateTest, InvalidEnumerationAddrMode) {
    ur_sampler_desc_t sampler_desc{
        UR_STRUCTURE_TYPE_SAMPLER_DESC,          /* stype */
        nullptr,                                 /* pNext */
        true,                                    /* normalizedCoords */
        UR_SAMPLER_ADDRESSING_MODE_FORCE_UINT32, /* addressing mode */
        UR_SAMPLER_FILTER_MODE_LINEAR,           /* filterMode */
    };
    uur::raii::Sampler hSampler = nullptr;
    ASSERT_EQ_RESULT(urSamplerCreate(context, &sampler_desc, hSampler.ptr()),
                     UR_RESULT_ERROR_INVALID_ENUMERATION);
}
TEST_P(urSamplerCreateTest, InvalidEnumerationFilterMode) {
    ur_sampler_desc_t sampler_desc{
        UR_STRUCTURE_TYPE_SAMPLER_DESC,      /* stype */
        nullptr,                             /* pNext */
        true,                                /* normalizedCoords */
        UR_SAMPLER_ADDRESSING_MODE_CLAMP,    /* addressing mode */
        UR_SAMPLER_FILTER_MODE_FORCE_UINT32, /* filterMode */
    };
    uur::raii::Sampler hSampler = nullptr;
    ASSERT_EQ_RESULT(urSamplerCreate(context, &sampler_desc, hSampler.ptr()),
                     UR_RESULT_ERROR_INVALID_ENUMERATION);
}
