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
        umf::memoryProviderMakeUnique<umf_test::provider_malloc>();
    EXPECT_EQ(ret, UMF_RESULT_SUCCESS);
    auto provider = providerUnique.release();
    auto [retp, pool] =
        umf::poolMakeUnique<usm::DisjointPool>(&provider, 1, poolConfig());
    EXPECT_EQ(retp, UMF_RESULT_SUCCESS);
    auto dtor = [provider = provider](umf_memory_pool_handle_t hPool) {
        umfPoolDestroy(hPool);
        umfMemoryProviderDestroy(provider);
    };
    return umf::pool_unique_handle_t(pool.release(), std::move(dtor));
}

INSTANTIATE_TEST_SUITE_P(disjointPoolTests, umfPoolTest,
                         ::testing::Values(makePool));

INSTANTIATE_TEST_SUITE_P(disjointMultiPoolTests, umfMultiPoolTest,
                         ::testing::Values(makePool));
