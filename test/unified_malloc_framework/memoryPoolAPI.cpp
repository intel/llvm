// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// This file contains tests for UMF pool API

#include "pool.h"
#include "pool.hpp"
#include "provider.h"
#include "provider.hpp"

#include "memoryPool.hpp"
#include "umf/memory_provider.h"

#include <array>
#include <string>
#include <thread>
#include <unordered_map>

using umf_test::test;

TEST_F(test, memoryPoolTrace) {
    static std::unordered_map<std::string, size_t> poolCalls;
    static std::unordered_map<std::string, size_t> providerCalls;
    auto tracePool = [](const char *name) { poolCalls[name]++; };
    auto traceProvider = [](const char *name) { providerCalls[name]++; };

    auto nullProvider = umf_test::wrapProviderUnique(nullProviderCreate());
    auto tracingProvider = umf_test::wrapProviderUnique(
        traceProviderCreate(nullProvider.get(), traceProvider));
    auto provider = tracingProvider.get();

    auto [ret, proxyPool] =
        umf::poolMakeUnique<umf_test::proxy_pool>(&provider, 1);
    ASSERT_EQ(ret, UMF_RESULT_SUCCESS);

    umf_memory_provider_handle_t providerDesc = nullProviderCreate();
    auto tracingPool = umf_test::wrapPoolUnique(
        tracePoolCreate(proxyPool.get(), providerDesc, tracePool));

    size_t pool_call_count = 0;
    size_t provider_call_count = 0;

    umfPoolMalloc(tracingPool.get(), 0);
    ASSERT_EQ(poolCalls["malloc"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    ASSERT_EQ(providerCalls["alloc"], 1);
    ASSERT_EQ(providerCalls.size(), ++provider_call_count);

    umfPoolFree(tracingPool.get(), nullptr);
    ASSERT_EQ(poolCalls["free"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    ASSERT_EQ(providerCalls["free"], 1);
    ASSERT_EQ(providerCalls.size(), ++provider_call_count);

    umfPoolCalloc(tracingPool.get(), 0, 0);
    ASSERT_EQ(poolCalls["calloc"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    ASSERT_EQ(providerCalls["alloc"], 2);
    ASSERT_EQ(providerCalls.size(), provider_call_count);

    umfPoolRealloc(tracingPool.get(), nullptr, 0);
    ASSERT_EQ(poolCalls["realloc"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    ASSERT_EQ(providerCalls.size(), provider_call_count);

    umfPoolAlignedMalloc(tracingPool.get(), 0, 0);
    ASSERT_EQ(poolCalls["aligned_malloc"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    ASSERT_EQ(providerCalls["alloc"], 3);
    ASSERT_EQ(providerCalls.size(), provider_call_count);

    umfPoolMallocUsableSize(tracingPool.get(), nullptr);
    ASSERT_EQ(poolCalls["malloc_usable_size"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    ASSERT_EQ(providerCalls.size(), provider_call_count);

    ret = umfPoolGetLastAllocationError(tracingPool.get());
    ASSERT_EQ(ret, UMF_RESULT_SUCCESS);
    ASSERT_EQ(poolCalls["get_last_native_error"], 1);
    ASSERT_EQ(poolCalls.size(), ++pool_call_count);

    umfMemoryProviderDestroy(providerDesc);
}

TEST_F(test, memoryPoolWithCustomProviders) {
    umf_memory_provider_handle_t providers[] = {nullProviderCreate(),
                                                nullProviderCreate()};

    struct pool : public umf_test::pool_base {
        umf_result_t initialize(umf_memory_provider_handle_t *providers,
                                size_t numProviders) noexcept {
            EXPECT_NE_NOEXCEPT(providers, nullptr);
            EXPECT_EQ_NOEXCEPT(numProviders, 2);
            return UMF_RESULT_SUCCESS;
        }
    };

    auto ret = umf::poolMakeUnique<pool>(providers, 2);
    ASSERT_EQ(ret.first, UMF_RESULT_SUCCESS);
    ASSERT_NE(ret.second, nullptr);

    for (auto &provider : providers) {
        umfMemoryProviderDestroy(provider);
    }
}

TEST_F(test, retrieveMemoryProviders) {
    static constexpr size_t numProviders = 4;
    std::array<umf_memory_provider_handle_t, numProviders> providers = {
        (umf_memory_provider_handle_t)0x1, (umf_memory_provider_handle_t)0x2,
        (umf_memory_provider_handle_t)0x3, (umf_memory_provider_handle_t)0x4};

    auto [ret, pool] = umf::poolMakeUnique<umf_test::proxy_pool>(
        providers.data(), numProviders);

    std::array<umf_memory_provider_handle_t, numProviders> retProviders;
    size_t numProvidersRet = 0;

    ret = umfPoolGetMemoryProviders(pool.get(), 0, nullptr, &numProvidersRet);
    ASSERT_EQ(ret, UMF_RESULT_SUCCESS);
    ASSERT_EQ(numProvidersRet, numProviders);

    ret = umfPoolGetMemoryProviders(pool.get(), numProviders,
                                    retProviders.data(), nullptr);
    ASSERT_EQ(ret, UMF_RESULT_SUCCESS);
    ASSERT_EQ(retProviders, providers);
}

INSTANTIATE_TEST_SUITE_P(
    mallocPoolTest, umfPoolTest, ::testing::Values([] {
        return umf::poolMakeUnique<umf_test::malloc_pool, 1>(
                   {umf_test::wrapProviderUnique(nullProviderCreate())})
            .second;
    }));

INSTANTIATE_TEST_SUITE_P(
    mallocProviderPoolTest, umfPoolTest, ::testing::Values([] {
        return umf::poolMakeUnique<umf_test::proxy_pool, 1>(
                   {umf::memoryProviderMakeUnique<umf_test::provider_malloc>()
                        .second})
            .second;
    }));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(umfMultiPoolTest);
INSTANTIATE_TEST_SUITE_P(
    mallocMultiPoolTest, umfMultiPoolTest, ::testing::Values([] {
        return umf::poolMakeUnique<umf_test::proxy_pool, 1>(
                   {umf::memoryProviderMakeUnique<umf_test::provider_malloc>()
                        .second})
            .second;
    }));

INSTANTIATE_TEST_SUITE_P(
    proxyPoolOOMTest, umfMemTest,
    ::testing::Values(std::tuple(
        [] {
            return umf_test::makePoolWithOOMProvider<umf_test::proxy_pool>(10);
        },
        0)));

////////////////// Negative test cases /////////////////

TEST_F(test, memoryPoolInvalidProvidersNullptr) {
    auto ret = umf::poolMakeUnique<umf_test::pool_base>(nullptr, 1);
    ASSERT_EQ(ret.first, UMF_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_F(test, memoryPoolInvalidProvidersNum) {
    auto nullProvider = umf_test::wrapProviderUnique(nullProviderCreate());
    umf_memory_provider_handle_t providers[] = {nullProvider.get()};

    auto ret = umf::poolMakeUnique<umf_test::pool_base>(providers, 0);
    ASSERT_EQ(ret.first, UMF_RESULT_ERROR_INVALID_ARGUMENT);
}

struct poolInitializeTest : umf_test::test,
                            ::testing::WithParamInterface<umf_result_t> {};

INSTANTIATE_TEST_SUITE_P(
    poolInitializeTest, poolInitializeTest,
    ::testing::Values(UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY,
                      UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC,
                      UMF_RESULT_ERROR_INVALID_ARGUMENT,
                      UMF_RESULT_ERROR_UNKNOWN));

TEST_P(poolInitializeTest, errorPropagation) {
    auto nullProvider = umf_test::wrapProviderUnique(nullProviderCreate());
    umf_memory_provider_handle_t providers[] = {nullProvider.get()};

    struct pool : public umf_test::pool_base {
        umf_result_t
        initialize([[maybe_unused]] umf_memory_provider_handle_t *providers,
                   [[maybe_unused]] size_t numProviders,
                   umf_result_t errorToReturn) noexcept {
            return errorToReturn;
        }
    };
    auto ret = umf::poolMakeUnique<pool>(providers, 1, this->GetParam());
    ASSERT_EQ(ret.first, this->GetParam());
    ASSERT_EQ(ret.second, nullptr);
}

TEST_F(test, retrieveMemoryProvidersError) {
    static constexpr size_t numProviders = 4;
    std::array<umf_memory_provider_handle_t, numProviders> providers = {
        (umf_memory_provider_handle_t)0x1, (umf_memory_provider_handle_t)0x2,
        (umf_memory_provider_handle_t)0x3, (umf_memory_provider_handle_t)0x4};

    auto [ret, pool] = umf::poolMakeUnique<umf_test::proxy_pool>(
        providers.data(), numProviders);

    ret = umfPoolGetMemoryProviders(pool.get(), 1, providers.data(), nullptr);
    ASSERT_EQ(ret, UMF_RESULT_ERROR_INVALID_ARGUMENT);
}

// TODO: extend test for different functions (not only alloc)
TEST_F(test, getLastFailedMemoryProvider) {
    static constexpr size_t allocSize = 8;
    static umf_result_t allocResult = UMF_RESULT_SUCCESS;

    struct memory_provider : public umf_test::provider_base {
        umf_result_t initialize(const char *name) {
            this->name = name;
            return UMF_RESULT_SUCCESS;
        }

        enum umf_result_t alloc(size_t size, size_t, void **ptr) noexcept {
            if (allocResult == UMF_RESULT_SUCCESS) {
                *ptr = malloc(size);
            } else {
                *ptr = nullptr;
            }

            return allocResult;
        }

        enum umf_result_t free(void *ptr,
                               [[maybe_unused]] size_t size) noexcept {
            ::free(ptr);
            return UMF_RESULT_SUCCESS;
        }

        const char *get_name() noexcept { return this->name; }

        const char *name;
    };

    auto [ret1, providerUnique1] =
        umf::memoryProviderMakeUnique<memory_provider>("provider1");
    ASSERT_EQ(ret1, UMF_RESULT_SUCCESS);
    auto [ret2, providerUnique2] =
        umf::memoryProviderMakeUnique<memory_provider>("provider2");
    ASSERT_EQ(ret2, UMF_RESULT_SUCCESS);

    auto hProvider = providerUnique1.get();

    auto [ret, pool] = umf::poolMakeUnique<umf_test::proxy_pool>(&hProvider, 1);
    ASSERT_EQ(ret, UMF_RESULT_SUCCESS);

    auto ptr = umfPoolMalloc(pool.get(), allocSize);
    ASSERT_NE(ptr, nullptr);
    umfPoolFree(pool.get(), ptr);

    // make provider return an error during allocation
    allocResult = UMF_RESULT_ERROR_UNKNOWN;
    ptr = umfPoolMalloc(pool.get(), allocSize);
    ASSERT_EQ(ptr, nullptr);
    ASSERT_EQ(std::string_view(
                  umfMemoryProviderGetName(umfGetLastFailedMemoryProvider())),
              "provider1");

    ret = umfMemoryProviderAlloc(providerUnique2.get(), allocSize, 0, &ptr);
    ASSERT_EQ(ptr, nullptr);
    ASSERT_EQ(std::string_view(
                  umfMemoryProviderGetName(umfGetLastFailedMemoryProvider())),
              "provider2");

    // successful provider should not be returned by umfGetLastFailedMemoryProvider
    allocResult = UMF_RESULT_SUCCESS;
    ptr = umfPoolMalloc(pool.get(), allocSize);
    ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(std::string_view(
                  umfMemoryProviderGetName(umfGetLastFailedMemoryProvider())),
              "provider2");
    umfPoolFree(pool.get(), ptr);

    // error in another thread should not impact umfGetLastFailedMemoryProvider on this thread
    allocResult = UMF_RESULT_ERROR_UNKNOWN;
    std::thread t([&, hPool = pool.get()] {
        ptr = umfPoolMalloc(hPool, allocSize);
        ASSERT_EQ(ptr, nullptr);
        ASSERT_EQ(std::string_view(umfMemoryProviderGetName(
                      umfGetLastFailedMemoryProvider())),
                  "provider1");
    });
    t.join();

    ASSERT_EQ(std::string_view(
                  umfMemoryProviderGetName(umfGetLastFailedMemoryProvider())),
              "provider2");
}
