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
    auto [ret, providerUnique] =
        uma::memoryProviderMakeUnique<uma_test::provider_malloc>();
    EXPECT_EQ(ret, UMA_RESULT_SUCCESS);
    auto provider = providerUnique.release();
    auto [retp, pool] =
        uma::poolMakeUnique<usm::DisjointPool>(&provider, 1, poolConfig());
    EXPECT_EQ(retp, UMA_RESULT_SUCCESS);
    auto dtor = [provider = provider](uma_memory_pool_handle_t hPool) {
        umaPoolDestroy(hPool);
        umaMemoryProviderDestroy(provider);
    };
    return uma::pool_unique_handle_t(pool.release(), std::move(dtor));
}

INSTANTIATE_TEST_SUITE_P(disjointPoolTests, umaPoolTest,
                         ::testing::Values(makePool));

INSTANTIATE_TEST_SUITE_P(disjointMultiPoolTests, umaMultiPoolTest,
                         ::testing::Values(makePool));
