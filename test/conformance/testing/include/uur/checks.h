// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_INCLUDE_CHECKS_H_INCLUDED
#define UR_CONFORMANCE_INCLUDE_CHECKS_H_INCLUDED

#include <gtest/gtest.h>
#include <ur_api.h>
#include <ur_print.hpp>
#include <uur/optional_queries.h>
#include <uur/utils.h>

namespace uur {

struct Result {
    Result(ur_result_t result) noexcept : value(result) {}

    constexpr bool operator==(const Result &rhs) const noexcept {
        return value == rhs.value;
    }

    ur_result_t value;
};

inline std::ostream &operator<<(std::ostream &out, const Result &result) {
    out << result.value;
    return out;
}

} // namespace uur

#ifndef ASSERT_EQ_RESULT
#define ASSERT_EQ_RESULT(EXPECTED, ACTUAL)                                     \
    ASSERT_EQ(uur::Result(EXPECTED), uur::Result(ACTUAL))
#endif
#ifndef ASSERT_SUCCESS
#define ASSERT_SUCCESS(ACTUAL) ASSERT_EQ_RESULT(UR_RESULT_SUCCESS, ACTUAL)
#endif

#ifndef EXPECT_EQ_RESULT
#define EXPECT_EQ_RESULT(EXPECTED, ACTUAL)                                     \
    EXPECT_EQ(uur::Result(EXPECTED), uur::Result(ACTUAL))
#endif
#ifndef EXPECT_SUCCESS
#define EXPECT_SUCCESS(ACTUAL) EXPECT_EQ_RESULT(UR_RESULT_SUCCESS, ACTUAL)
#endif

// This macro is intended to be used for the first call to a GetInfo query, it
// gracefully handles cases where the adapter doesn't support a query marked
// [optional-query] in the spec by returning early.
#ifndef ASSERT_SUCCESS_OR_OPTIONAL_QUERY
#define ASSERT_SUCCESS_OR_OPTIONAL_QUERY(CALL, QUERY)                          \
    do {                                                                       \
        auto result = CALL;                                                    \
        if (result != UR_RESULT_SUCCESS) {                                     \
            ASSERT_EQ_RESULT(result, UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION); \
            ASSERT_TRUE(uur::isQueryOptional(QUERY));                          \
            return;                                                            \
        }                                                                      \
    } while (0)
#endif

inline std::ostream &operator<<(std::ostream &out,
                                const ur_device_handle_t &device) {
    out << uur::GetDeviceName(device);
    return out;
}

#endif // UR_CONFORMANCE_INCLUDE_CHECKS_H_INCLUDED
