// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
// This file contains tests for UMA pool API

#include "pool.h"
#include "pool.hpp"
#include "provider.h"
#include "provider.hpp"

#include "memoryPool.hpp"
#include "uma/memory_provider.h"

#include <array>
#include <string>
#include <unordered_map>

using uma_test::test;

TEST_F(test, memoryPoolTrace) {
    static std::unordered_map<std::string, size_t> poolCalls;
    static std::unordered_map<std::string, size_t> providerCalls;
    auto tracePool = [](const char *name) { poolCalls[name]++; };
    auto traceProvider = [](const char *name) { providerCalls[name]++; };

    auto nullProvider = uma_test::wrapProviderUnique(nullProviderCreate());
    auto tracingProvider = uma_test::wrapProviderUnique(
        traceProviderCreate(nullProvider.get(), traceProvider));
    auto provider = tracingProvider.get();

    auto [ret, proxyPool] =
        uma::poolMakeUnique<uma_test::proxy_pool>(&provider, 1);
    ASSERT_EQ(ret, UMA_RESULT_SUCCESS);

    uma_memory_provider_handle_t providerDesc = nullProviderCreate();
    auto tracingPool = uma_test::wrapPoolUnique(
        tracePoolCreate(proxyPool.get(), providerDesc, tracePool));

    size_t pool_call_count = 0;
    size_t provider_call_count = 0;

    umaPoolMalloc(tracingPool.get(), 0);
    ASSERT_EQ(poolCalls["malloc"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    ASSERT_EQ(providerCalls["alloc"], 1);
    ASSERT_EQ(providerCalls.size(), ++provider_call_count);

    umaPoolFree(tracingPool.get(), nullptr);
    ASSERT_EQ(poolCalls["free"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    ASSERT_EQ(providerCalls["free"], 1);
    ASSERT_EQ(providerCalls.size(), ++provider_call_count);

    umaPoolCalloc(tracingPool.get(), 0, 0);
    ASSERT_EQ(poolCalls["calloc"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    ASSERT_EQ(providerCalls["alloc"], 2);
    ASSERT_EQ(providerCalls.size(), provider_call_count);

    umaPoolRealloc(tracingPool.get(), nullptr, 0);
    ASSERT_EQ(poolCalls["realloc"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    ASSERT_EQ(providerCalls.size(), provider_call_count);

    umaPoolAlignedMalloc(tracingPool.get(), 0, 0);
    ASSERT_EQ(poolCalls["aligned_malloc"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    ASSERT_EQ(providerCalls["alloc"], 3);
    ASSERT_EQ(providerCalls.size(), provider_call_count);

    umaPoolMallocUsableSize(tracingPool.get(), nullptr);
    ASSERT_EQ(poolCalls["malloc_usable_size"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    ASSERT_EQ(providerCalls.size(), provider_call_count);

    ret = umaPoolGetLastResult(tracingPool.get(), nullptr);
    ASSERT_EQ(ret, UMA_RESULT_SUCCESS);

    ASSERT_EQ(poolCalls["get_last_result"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    ASSERT_EQ(providerCalls["get_last_result"], 1);
    ASSERT_EQ(providerCalls.size(), ++provider_call_count);

    umaMemoryProviderDestroy(providerDesc);
}

TEST_F(test, memoryPoolWithCustomProviders) {
    uma_memory_provider_handle_t providers[] = {nullProviderCreate(),
                                                nullProviderCreate()};

    struct pool : public uma_test::pool_base {
        uma_result_t initialize(uma_memory_provider_handle_t *providers,
                                size_t numProviders) noexcept {
            EXPECT_NE(providers, nullptr);
            EXPECT_EQ(numProviders, 2);
            return UMA_RESULT_SUCCESS;
        }
    };

    auto ret = uma::poolMakeUnique<pool>(providers, 2);
    ASSERT_EQ(ret.first, UMA_RESULT_SUCCESS);
    ASSERT_NE(ret.second, nullptr);

    for (auto &provider : providers) {
        umaMemoryProviderDestroy(provider);
    }
}

TEST_F(test, retrieveMemoryProviders) {
    static constexpr size_t numProviders = 4;
    std::array<uma_memory_provider_handle_t, numProviders> providers = {
        (uma_memory_provider_handle_t)0x1, (uma_memory_provider_handle_t)0x2,
        (uma_memory_provider_handle_t)0x3, (uma_memory_provider_handle_t)0x4};

    auto [ret, pool] = uma::poolMakeUnique<uma_test::proxy_pool>(
        providers.data(), numProviders);

    std::array<uma_memory_provider_handle_t, numProviders> retProviders;
    size_t numProvidersRet = 0;

    ret = umaPoolGetMemoryProviders(pool.get(), 0, nullptr, &numProvidersRet);
    ASSERT_EQ(ret, UMA_RESULT_SUCCESS);
    ASSERT_EQ(numProvidersRet, numProviders);

    ret = umaPoolGetMemoryProviders(pool.get(), numProviders,
                                    retProviders.data(), nullptr);
    ASSERT_EQ(ret, UMA_RESULT_SUCCESS);
    ASSERT_EQ(retProviders, providers);
}

template <typename Pool>
static auto
makePool(std::function<uma::provider_unique_handle_t()> makeProvider) {
    auto providerUnique = makeProvider();
    uma_memory_provider_handle_t provider = providerUnique.get();
    auto pool = uma::poolMakeUnique<Pool>(&provider, 1).second;
    auto dtor = [provider =
                     providerUnique.release()](uma_memory_pool_handle_t hPool) {
        umaPoolDestroy(hPool);
        umaMemoryProviderDestroy(provider);
    };
    return uma::pool_unique_handle_t(pool.release(), std::move(dtor));
}

INSTANTIATE_TEST_SUITE_P(mallocPoolTest, umaPoolTest, ::testing::Values([] {
                             return makePool<uma_test::malloc_pool>([] {
                                 return uma_test::wrapProviderUnique(
                                     nullProviderCreate());
                             });
                         }));

INSTANTIATE_TEST_SUITE_P(
    mallocProviderPoolTest, umaPoolTest, ::testing::Values([] {
        return makePool<uma_test::proxy_pool>([] {
            return uma::memoryProviderMakeUnique<uma_test::provider_malloc>()
                .second;
        });
    }));

INSTANTIATE_TEST_SUITE_P(
    mallocMultiPoolTest, umaMultiPoolTest, ::testing::Values([] {
        return makePool<uma_test::proxy_pool>([] {
            return uma::memoryProviderMakeUnique<uma_test::provider_malloc>()
                .second;
        });
    }));

////////////////// Negative test cases /////////////////

TEST_F(test, memoryPoolInvalidProvidersNullptr) {
    auto ret = uma::poolMakeUnique<uma_test::pool_base>(nullptr, 1);
    ASSERT_EQ(ret.first, UMA_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_F(test, memoryPoolInvalidProvidersNum) {
    auto nullProvider = uma_test::wrapProviderUnique(nullProviderCreate());
    uma_memory_provider_handle_t providers[] = {nullProvider.get()};

    auto ret = uma::poolMakeUnique<uma_test::pool_base>(providers, 0);
    ASSERT_EQ(ret.first, UMA_RESULT_ERROR_INVALID_ARGUMENT);
}

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
    auto nullProvider = uma_test::wrapProviderUnique(nullProviderCreate());
    uma_memory_provider_handle_t providers[] = {nullProvider.get()};

    struct pool : public uma_test::pool_base {
        uma_result_t initialize(uma_memory_provider_handle_t *providers,
                                size_t numProviders,
                                uma_result_t errorToReturn) noexcept {
            return errorToReturn;
        }
    };
    auto ret = uma::poolMakeUnique<pool>(providers, 1, this->GetParam());
    ASSERT_EQ(ret.first, this->GetParam());
    ASSERT_EQ(ret.second, nullptr);
}

TEST_F(test, retrieveMemoryProvidersError) {
    static constexpr size_t numProviders = 4;
    std::array<uma_memory_provider_handle_t, numProviders> providers = {
        (uma_memory_provider_handle_t)0x1, (uma_memory_provider_handle_t)0x2,
        (uma_memory_provider_handle_t)0x3, (uma_memory_provider_handle_t)0x4};

    auto [ret, pool] = uma::poolMakeUnique<uma_test::proxy_pool>(
        providers.data(), numProviders);

    ret = umaPoolGetMemoryProviders(pool.get(), 1, providers.data(), nullptr);
    ASSERT_EQ(ret, UMA_RESULT_ERROR_INVALID_ARGUMENT);
}
