// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
// This file contains tests for UMA provider API

#include "pool.h"
#include "provider.h"
#include "provider.hpp"

#include <string>
#include <unordered_map>

using uma_test::test;

TEST_F(test, memoryProviderTrace) {
    static std::unordered_map<std::string, size_t> calls;
    auto trace = [](const char *name) { calls[name]++; };

    auto nullProvider = uma_test::wrapProviderUnique(nullProviderCreate());
    auto tracingProvider = uma_test::wrapProviderUnique(
        traceProviderCreate(nullProvider.get(), trace));

    size_t call_count = 0;

    auto ret = umaMemoryProviderAlloc(tracingProvider.get(), 0, 0, nullptr);
    ASSERT_EQ(ret, UMA_RESULT_SUCCESS);
    ASSERT_EQ(calls["alloc"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    ret = umaMemoryProviderFree(tracingProvider.get(), nullptr, 0);
    ASSERT_EQ(ret, UMA_RESULT_SUCCESS);
    ASSERT_EQ(calls["free"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    ret = umaMemoryProviderGetLastResult(tracingProvider.get(), nullptr);
    ASSERT_EQ(ret, UMA_RESULT_SUCCESS);
    ASSERT_EQ(calls["get_last_result"], 1);
    ASSERT_EQ(calls.size(), ++call_count);
}

//////////////////////////// Negative test cases
///////////////////////////////////

struct providerInitializeTest : uma_test::test,
                                ::testing::WithParamInterface<uma_result_t> {};

INSTANTIATE_TEST_SUITE_P(
    providerInitializeTest, providerInitializeTest,
    ::testing::Values(UMA_RESULT_ERROR_OUT_OF_HOST_MEMORY,
                      UMA_RESULT_ERROR_POOL_SPECIFIC,
                      UMA_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC,
                      UMA_RESULT_ERROR_INVALID_ARGUMENT,
                      UMA_RESULT_ERROR_UNKNOWN));

TEST_P(providerInitializeTest, errorPropagation) {
    struct provider : public uma_test::provider_base {
        uma_result_t initialize(uma_result_t errorToReturn) noexcept {
            return errorToReturn;
        }
    };
    auto ret = uma::memoryProviderMakeUnique<provider>(this->GetParam());
    ASSERT_EQ(ret.first, this->GetParam());
    ASSERT_EQ(ret.second, nullptr);
}
