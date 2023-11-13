/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "disjoint_pool.hpp"

#include "memoryPool.hpp"
#include "pool.hpp"
#include "provider.h"
#include "provider.hpp"

static usm::DisjointPool::Config poolConfig() {
    usm::DisjointPool::Config config{};
    config.SlabMinSize = 4096;
    config.MaxPoolableSize = 4096;
    config.Capacity = 4;
    config.MinBucketSize = 64;
    return config;
}

static auto makePool() {
    auto [ret, provider] =
        umf::memoryProviderMakeUnique<umf_test::provider_malloc>();
    EXPECT_EQ(ret, UMF_RESULT_SUCCESS);
    auto [retp, pool] = umf::poolMakeUnique<usm::DisjointPool, 1>(
        {std::move(provider)}, poolConfig());
    EXPECT_EQ(retp, UMF_RESULT_SUCCESS);
    return std::move(pool);
}

using umf_test::test;

TEST_F(test, freeErrorPropagation) {
    static enum umf_result_t freeReturn = UMF_RESULT_SUCCESS;
    struct memory_provider : public umf_test::provider_base {
        enum umf_result_t alloc(size_t size, size_t, void **ptr) noexcept {
            *ptr = malloc(size);
            return UMF_RESULT_SUCCESS;
        }
        enum umf_result_t free(void *ptr,
                               [[maybe_unused]] size_t size) noexcept {
            ::free(ptr);
            return freeReturn;
        }
    };

    auto [ret, providerUnique] =
        umf::memoryProviderMakeUnique<memory_provider>();
    ASSERT_EQ(ret, UMF_RESULT_SUCCESS);

    auto config = poolConfig();
    config.MaxPoolableSize =
        0; // force all allocations to go to memory provider

    auto [retp, pool] = umf::poolMakeUnique<usm::DisjointPool, 1>(
        {std::move(providerUnique)}, config);
    EXPECT_EQ(retp, UMF_RESULT_SUCCESS);

    static constexpr size_t size = 1024;
    void *ptr = umfPoolMalloc(pool.get(), size);

    freeReturn = UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
    auto freeRet = umfPoolFree(pool.get(), ptr);

    EXPECT_EQ(freeRet, freeReturn);
}

INSTANTIATE_TEST_SUITE_P(disjointPoolTests, umfPoolTest,
                         ::testing::Values(makePool));

INSTANTIATE_TEST_SUITE_P(
    disjointPoolTests, umfMemTest,
    ::testing::Values(std::make_tuple(
        [] {
            return umf_test::makePoolWithOOMProvider<usm::DisjointPool>(
                static_cast<int>(poolConfig().Capacity), poolConfig());
        },
        static_cast<int>(poolConfig().Capacity) / 2)));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(umfMultiPoolTest);
INSTANTIATE_TEST_SUITE_P(disjointMultiPoolTests, umfMultiPoolTest,
                         ::testing::Values(makePool));
