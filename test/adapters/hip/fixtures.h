// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_TEST_CONFORMANCE_ADAPTERS_HIP_FIXTURES_H_INCLUDED
#define UR_TEST_CONFORMANCE_ADAPTERS_HIP_FIXTURES_H_INCLUDED
#include <hip/hip_runtime.h>
#include <uur/fixtures.h>

namespace uur {

struct ResultHip {
    constexpr ResultHip(hipError_t result) noexcept : value(result) {}

    inline bool operator==(const ResultHip &rhs) const noexcept {
        return rhs.value == value;
    }

    hipError_t value;
};

} // namespace uur

#ifndef ASSERT_EQ_RESULT_HIP
#define ASSERT_EQ_RESULT_HIP(EXPECTED, ACTUAL)                                 \
    ASSERT_EQ(uur::ResultHip(EXPECTED), uur::ResultHip(ACTUAL))
#endif // ASSERT_EQ_RESULT_HIP

#ifndef ASSERT_SUCCESS_HIP
#define ASSERT_SUCCESS_HIP(ACTUAL) ASSERT_EQ_RESULT_HIP(hipSuccess, ACTUAL)
#endif // ASSERT_SUCCESS_HIP

#ifndef EXPECT_EQ_RESULT_HIP
#define EXPECT_EQ_RESULT_HIP(EXPECTED, ACTUAL)                                 \
    EXPECT_EQ_RESULT_HIP(uur::ResultHip(EXPECTED), uur::ResultHip(ACTUAL))
#endif // EXPECT_EQ_RESULT_HIP

#ifndef EXPECT_SUCCESS_HIP
#define EXPECT_SUCCESS_HIP(ACTUAL) EXPECT_EQ_RESULT_HIP(hipSuccess, ACTUAL)
#endif // EXPECT_EQ_RESULT_HIP

#endif // UR_TEST_CONFORMANCE_ADAPTERS_HIP_FIXTURES_H_INCLUDED
