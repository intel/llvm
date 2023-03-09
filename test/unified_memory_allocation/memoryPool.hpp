// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "pool.hpp"

#include <cstring>
#include <functional>

#ifndef UMA_TEST_MEMORY_POOL_OPS_HPP
#define UMA_TEST_MEMORY_POOL_OPS_HPP

struct umaPoolTest : uma_test::test, ::testing::WithParamInterface<std::function<std::pair<uma_result_t, uma::pool_unique_handle_t>()>> {
    umaPoolTest() : pool(nullptr, nullptr) {}
    void SetUp() {
        test::SetUp();
        auto [res, pool] = this->GetParam()();
        EXPECT_EQ(res, UMA_RESULT_SUCCESS);
        EXPECT_NE(pool, nullptr);
        this->pool = std::move(pool);
    }
    void TearDown() override {
        test::TearDown();
    }
    uma::pool_unique_handle_t pool;
};

TEST_P(umaPoolTest, allocFree) {
    static constexpr size_t allocSize = 64;
    auto *ptr = umaPoolMalloc(pool.get(), allocSize);
    ASSERT_NE(ptr, nullptr);
    std::memset(ptr, 0, allocSize);
    umaPoolFree(pool.get(), ptr);
}

TEST_P(umaPoolTest, pow2AlignedAlloc) {
#ifdef _WIN32
    // TODO: implement support for windows
    GTEST_SKIP();
#endif

    static constexpr size_t allocSize = 64;
    static constexpr size_t maxAlignment = (1u << 22);

    for (size_t alignment = 1; alignment <= maxAlignment; alignment <<= 1) {
        std::cout << alignment << std::endl;
        auto *ptr = umaPoolAlignedMalloc(pool.get(), allocSize, alignment);
        ASSERT_NE(ptr, nullptr);
        ASSERT_TRUE(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
        std::memset(ptr, 0, allocSize);
        umaPoolFree(pool.get(), ptr);
    }
}

#endif /* UMA_TEST_MEMORY_POOL_OPS_HPP */
