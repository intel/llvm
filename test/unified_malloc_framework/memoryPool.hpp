// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "disjoint_pool.hpp"
#include "pool.hpp"
#include "provider.hpp"

#include <array>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>

#ifndef UMF_TEST_MEMORY_POOL_OPS_HPP
#define UMF_TEST_MEMORY_POOL_OPS_HPP

struct umfPoolTest : umf_test::test,
                     ::testing::WithParamInterface<
                         std::function<umf::pool_unique_handle_t(void)>> {
    umfPoolTest() : pool(nullptr, nullptr) {}
    void SetUp() override {
        test::SetUp();
        this->pool = makePool();
    }

    void TearDown() override { test::TearDown(); }

    umf::pool_unique_handle_t makePool() {
        auto pool = this->GetParam()();
        EXPECT_NE(pool, nullptr);
        return pool;
    }

    umf::pool_unique_handle_t pool;
    static constexpr int NTHREADS = 5;
};

struct umfMultiPoolTest : umfPoolTest {
    static constexpr auto numPools = 16;

    void SetUp() override {
        umfPoolTest::SetUp();

        pools.emplace_back(std::move(pool));
        for (size_t i = 1; i < numPools; i++) {
            pools.emplace_back(makePool());
        }
    }

    void TearDown() override { umfPoolTest::TearDown(); }

    std::vector<umf::pool_unique_handle_t> pools;
};

struct umfMemTest
    : umf_test::test,
      ::testing::WithParamInterface<
          std::tuple<std::function<umf::pool_unique_handle_t(void)>, int>> {
    umfMemTest() : pool(nullptr, nullptr), expectedRecycledPoolAllocs(0) {}
    void SetUp() override {
        test::SetUp();
        initialize();
    }

    void TearDown() override { test::TearDown(); }

    void initialize() {
        auto [pool_fun, expectedRecycledPoolAllocs] = this->GetParam();
        EXPECT_NE(pool_fun(), nullptr);
        this->pool = pool_fun();
        this->expectedRecycledPoolAllocs = expectedRecycledPoolAllocs;
    }

    umf::pool_unique_handle_t pool;
    int expectedRecycledPoolAllocs;
};

TEST_P(umfPoolTest, allocFree) {
    static constexpr size_t allocSize = 64;
    auto *ptr = umfPoolMalloc(pool.get(), allocSize);
    ASSERT_NE(ptr, nullptr);
    std::memset(ptr, 0, allocSize);
    umfPoolFree(pool.get(), ptr);
}

TEST_P(umfPoolTest, reallocFree) {
    if (!umf_test::isReallocSupported(pool.get())) {
        GTEST_SKIP();
    }
    static constexpr size_t allocSize = 64;
    static constexpr size_t multiplier = 3;
    auto *ptr = umfPoolMalloc(pool.get(), allocSize);
    ASSERT_NE(ptr, nullptr);
    auto *new_ptr = umfPoolRealloc(pool.get(), ptr, allocSize * multiplier);
    ASSERT_NE(new_ptr, nullptr);
    std::memset(new_ptr, 0, allocSize * multiplier);
    umfPoolFree(pool.get(), new_ptr);
}

TEST_P(umfPoolTest, callocFree) {
    if (!umf_test::isCallocSupported(pool.get())) {
        GTEST_SKIP();
    }
    static constexpr size_t num = 10;
    static constexpr size_t size = sizeof(int);
    auto *ptr = umfPoolCalloc(pool.get(), num, size);
    ASSERT_NE(ptr, nullptr);
    for (size_t i = 0; i < num; ++i) {
        ASSERT_EQ(((int *)ptr)[i], 0);
    }
    umfPoolFree(pool.get(), ptr);
}

void pow2AlignedAllocHelper(umf_memory_pool_handle_t pool) {
    static constexpr size_t maxAlignment = (1u << 22);
    static constexpr size_t numAllocs = 4;
    for (size_t alignment = 1; alignment <= maxAlignment; alignment <<= 1) {
        std::cout << alignment << std::endl;
        std::vector<void *> allocs;

        for (size_t alloc = 0; alloc < numAllocs; alloc++) {
            auto *ptr = umfPoolAlignedMalloc(pool, alignment, alignment);
            ASSERT_NE(ptr, nullptr);
            ASSERT_TRUE(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
            std::memset(ptr, 0, alignment);
            allocs.push_back(ptr);
        }

        for (auto &ptr : allocs) {
            umfPoolFree(pool, ptr);
        }
    }
}

TEST_P(umfPoolTest, pow2AlignedAlloc) {
#ifdef _WIN32
    // TODO: implement support for windows
    GTEST_SKIP();
#endif
    pow2AlignedAllocHelper(pool.get());
}

TEST_P(umfPoolTest, freeNullptr) {
    void *ptr = nullptr;
    auto ret = umfPoolFree(pool.get(), ptr);
    ASSERT_EQ(ret, UMF_RESULT_SUCCESS);
}

TEST_P(umfPoolTest, multiThreadedMallocFree) {
    static constexpr size_t allocSize = 64;
    auto poolMalloc = [](size_t allocSize, umf_memory_pool_handle_t pool) {
        std::vector<void *> allocations;
        for (size_t i = 0; i <= 10; ++i) {
            allocations.emplace_back(umfPoolMalloc(pool, allocSize));
            ASSERT_NE(allocations.back(), nullptr);
        }

        for (auto allocation : allocations) {
            umfPoolFree(pool, allocation);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NTHREADS; i++) {
        threads.emplace_back(poolMalloc, allocSize, pool.get());
        ;
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

TEST_P(umfPoolTest, multiThreadedpow2AlignedAlloc) {
#ifdef _WIN32
    // TODO: implement support for windows
    GTEST_SKIP();
#endif

    auto poolpow2AlignedAlloc = [](umf_memory_pool_handle_t pool) {
        pow2AlignedAllocHelper(pool);
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NTHREADS; i++) {
        threads.emplace_back(poolpow2AlignedAlloc, pool.get());
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

TEST_P(umfPoolTest, multiThreadedReallocFree) {
    if (!umf_test::isReallocSupported(pool.get())) {
        GTEST_SKIP();
    }
    static constexpr size_t allocSize = 64;
    static constexpr size_t multiplier = 3;
    auto poolRealloc = [](size_t allocSize, size_t multiplier,
                          umf_memory_pool_handle_t pool) {
        std::vector<void *> allocations;
        for (size_t i = 0; i <= 10; ++i) {
            allocations.emplace_back(umfPoolMalloc(pool, allocSize));
            ASSERT_NE(allocations.back(), nullptr);
        }

        for (auto allocation : allocations) {
            auto *ptr =
                umfPoolRealloc(pool, allocation, allocSize * multiplier);
            umfPoolFree(pool, ptr);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NTHREADS; i++) {
        threads.emplace_back(poolRealloc, allocSize, multiplier, pool.get());
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

TEST_P(umfPoolTest, multiThreadedCallocFree) {
    if (!umf_test::isCallocSupported(pool.get())) {
        GTEST_SKIP();
    }
    static constexpr size_t num = 10;
    auto poolCalloc = [](size_t num, size_t size,
                         umf_memory_pool_handle_t pool) {
        std::vector<void *> allocations;
        for (size_t i = 0; i <= 10; ++i) {
            allocations.emplace_back(umfPoolCalloc(pool, num, size));
            ASSERT_NE(allocations.back(), nullptr);
        }

        for (auto allocation : allocations) {
            umfPoolFree(pool, allocation);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NTHREADS; i++) {
        threads.emplace_back(poolCalloc, num, sizeof(int), pool.get());
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

TEST_P(umfPoolTest, multiThreadedMallocFreeRandomSizes) {
    auto poolMalloc = [](size_t allocSize, umf_memory_pool_handle_t pool) {
        std::vector<void *> allocations;
        for (size_t i = 0; i <= 10; ++i) {
            allocations.emplace_back(umfPoolMalloc(pool, allocSize));
            ASSERT_NE(allocations.back(), nullptr);
        }

        for (auto allocation : allocations) {
            umfPoolFree(pool, allocation);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NTHREADS; i++) {
        threads.emplace_back(poolMalloc, (rand() % 16) * 8, pool.get());
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

TEST_P(umfMemTest, outOfMem) {
    static constexpr size_t allocSize = 4096;
    auto hPool = pool.get();

    std::vector<void *> allocations;

    while (true) {
        allocations.emplace_back(umfPoolMalloc(hPool, allocSize));
        if (allocations.back() == nullptr &&
            umfPoolGetLastAllocationError(hPool) ==
                UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY) {
            break;
        }
        ASSERT_NE(allocations.back(), nullptr);
    }

    // next part of the test- freeing some memory to allocate it again (as the memory
    // should be acquired from the pool itself now, not from the provider),
    // is done only for the disjoint pool for now

    // remove last nullptr from the allocations vector
    ASSERT_EQ(allocations.back(), nullptr);
    allocations.pop_back();

    ASSERT_NE(allocations.back(), nullptr);
    for (int i = 0; i < expectedRecycledPoolAllocs; i++) {
        umfPoolFree(hPool, allocations.back());
        allocations.pop_back();
    }

    for (int i = 0; i < expectedRecycledPoolAllocs; i++) {
        allocations.emplace_back(umfPoolMalloc(hPool, allocSize));
        ASSERT_NE(allocations.back(), nullptr);
    }

    for (auto allocation : allocations) {
        umfPoolFree(hPool, allocation);
    }
}

#ifdef UMF_ENABLE_POOL_TRACKING_TESTS
// TODO: add similar tests for realloc/aligned_alloc, etc.
// TODO: add multithreaded tests
TEST_P(umfMultiPoolTest, memoryTracking) {
    static constexpr int allocSizes[] = {8, 16, 32, 40, 64, 128, 1024, 4096};
    static constexpr auto nAllocs = 256;

    std::mt19937_64 g(0);
    std::uniform_int_distribution allocSizesDist(
        0, static_cast<int>(std::size(allocSizes) - 1));
    std::uniform_int_distribution poolsDist(0,
                                            static_cast<int>(pools.size() - 1));

    std::vector<std::tuple<void *, size_t, umf_memory_pool_handle_t>> ptrs;
    for (size_t i = 0; i < nAllocs; i++) {
        auto &pool = pools[poolsDist(g)];
        auto size = allocSizes[allocSizesDist(g)];

        auto *ptr = umfPoolMalloc(pool.get(), size);
        ASSERT_NE(ptr, nullptr);

        ptrs.emplace_back(ptr, size, pool.get());
    }

    for (auto [ptr, size, expectedPool] : ptrs) {
        auto pool = umfPoolByPtr(ptr);
        ASSERT_EQ(pool, expectedPool);

        pool = umfPoolByPtr(reinterpret_cast<void *>(
            reinterpret_cast<intptr_t>(ptr) + size - 1));
        ASSERT_EQ(pool, expectedPool);
    }

    for (auto &p : ptrs) {
        umfFree(std::get<0>(p));
    }
}
#endif /* UMF_ENABLE_POOL_TRACKING_TESTS */

#endif /* UMF_TEST_MEMORY_POOL_OPS_HPP */
