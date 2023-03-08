// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
// This file contains tests for UMA pool API

#include "pool.h"
#include "pool.hpp"

#include "memoryPool.hpp"

#include <string>
#include <unordered_map>

using uma_test::test;

TEST_F(test, memoryPoolTrace) {
    static std::unordered_map<std::string, size_t> calls;
    auto trace = [](const char *name) { calls[name]++; };

    auto nullPool = uma_test::wrapPoolUnique(nullPoolCreate());
    auto tracingPool =
        uma_test::wrapPoolUnique(tracePoolCreate(nullPool.get(), trace));

    size_t call_count = 0;

    umaPoolMalloc(tracingPool.get(), 0);
    ASSERT_EQ(calls["malloc"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    umaPoolFree(tracingPool.get(), nullptr);
    ASSERT_EQ(calls["free"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    umaPoolCalloc(tracingPool.get(), 0, 0);
    ASSERT_EQ(calls["calloc"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    umaPoolRealloc(tracingPool.get(), nullptr, 0);
    ASSERT_EQ(calls["realloc"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    umaPoolAlignedMalloc(tracingPool.get(), 0, 0);
    ASSERT_EQ(calls["aligned_malloc"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    umaPoolMallocUsableSize(tracingPool.get(), nullptr);
    ASSERT_EQ(calls["malloc_usable_size"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    enum uma_result_t ret = umaPoolGetLastResult(tracingPool.get(), nullptr);
    ASSERT_EQ(ret, UMA_RESULT_SUCCESS);
    ASSERT_EQ(calls["get_last_result"], 1);
    ASSERT_EQ(calls.size(), ++call_count);
}

INSTANTIATE_TEST_SUITE_P(
    mallocPoolTest, umaPoolTest, ::testing::Values([] {
        return uma::poolMakeUnique<uma_test::malloc_pool>();
    }));

//////////////////////////// Negative test cases
///////////////////////////////////

struct poolInitializeTest : uma_test::test,
                            ::testing::WithParamInterface<uma_result_t> {};

INSTANTIATE_TEST_SUITE_P(
    poolInitializeTest, poolInitializeTest,
    ::testing::Values(UMA_RESULT_ERROR_OUT_OF_HOST_MEMORY,
                      UMA_RESULT_ERROR_POOL_SPECIFIC,
                      UMA_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC,
                      UMA_RESULT_ERROR_INVALID_ARGUMENT,
                      UMA_RESULT_ERROR_UNKNOWN));

TEST_P(poolInitializeTest, errorPropagation) {
    struct pool : public uma_test::pool_base {
        uma_result_t initialize(uma_result_t errorToReturn) noexcept {
            return errorToReturn;
        }
    };
    auto ret = uma::poolMakeUnique<pool>(this->GetParam());
    ASSERT_EQ(ret.first, this->GetParam());
    ASSERT_EQ(ret.second, nullptr);
}
