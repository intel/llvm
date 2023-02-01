// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "helpers.h"
#include "pool.h"

#include <string>
#include <unordered_map>

TEST_F(umaTest, memoryPoolTrace) {
    static std::unordered_map<std::string, size_t> calls;
    auto trace = [](const char *name) { calls[name]++; };

    auto nullPool = uma::wrapPoolUnique(nullPoolCreate());
    auto tracingPool =
        uma::wrapPoolUnique(tracePoolCreate(nullPool.get(), trace));

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
}
