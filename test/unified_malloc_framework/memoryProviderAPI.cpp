// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// This file contains tests for UMF provider API

#include "pool.h"
#include "provider.h"
#include "provider.hpp"

#include <string>
#include <unordered_map>

using umf_test::test;

TEST_F(test, memoryProviderTrace) {
    static std::unordered_map<std::string, size_t> calls;
    auto trace = [](const char *name) { calls[name]++; };

    auto nullProvider = umf_test::wrapProviderUnique(nullProviderCreate());
    auto tracingProvider = umf_test::wrapProviderUnique(
        traceProviderCreate(nullProvider.get(), trace));

    size_t call_count = 0;

    void *ptr;
    auto ret = umfMemoryProviderAlloc(tracingProvider.get(), 0, 0, &ptr);
    ASSERT_EQ(ret, UMF_RESULT_SUCCESS);
    ASSERT_EQ(calls["alloc"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    ret = umfMemoryProviderFree(tracingProvider.get(), nullptr, 0);
    ASSERT_EQ(ret, UMF_RESULT_SUCCESS);
    ASSERT_EQ(calls["free"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    umfMemoryProviderGetLastNativeError(tracingProvider.get(), nullptr,
                                        nullptr);
    ASSERT_EQ(calls["get_last_native_error"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    ret = umfMemoryProviderGetRecommendedPageSize(tracingProvider.get(), 0,
                                                  nullptr);
    ASSERT_EQ(ret, UMF_RESULT_SUCCESS);
    ASSERT_EQ(calls["get_recommended_page_size"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    ret = umfMemoryProviderGetMinPageSize(tracingProvider.get(), nullptr,
                                          nullptr);
    ASSERT_EQ(ret, UMF_RESULT_SUCCESS);
    ASSERT_EQ(calls["get_min_page_size"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    ret = umfMemoryProviderPurgeLazy(tracingProvider.get(), nullptr, 0);
    ASSERT_EQ(ret, UMF_RESULT_SUCCESS);
    ASSERT_EQ(calls["purge_lazy"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    ret = umfMemoryProviderPurgeForce(tracingProvider.get(), nullptr, 0);
    ASSERT_EQ(ret, UMF_RESULT_SUCCESS);
    ASSERT_EQ(calls["purge_force"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    const char *pName = umfMemoryProviderGetName(tracingProvider.get());
    ASSERT_EQ(calls["name"], 1);
    ASSERT_EQ(calls.size(), ++call_count);
    ASSERT_EQ(std::string(pName), std::string("null"));
}

//////////////////////////// Negative test cases
///////////////////////////////////

struct providerInitializeTest : umf_test::test,
                                ::testing::WithParamInterface<umf_result_t> {};

INSTANTIATE_TEST_SUITE_P(
    providerInitializeTest, providerInitializeTest,
    ::testing::Values(UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY,
                      UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC,
                      UMF_RESULT_ERROR_INVALID_ARGUMENT,
                      UMF_RESULT_ERROR_UNKNOWN));

TEST_P(providerInitializeTest, errorPropagation) {
    struct provider : public umf_test::provider_base {
        umf_result_t initialize(umf_result_t errorToReturn) noexcept {
            return errorToReturn;
        }
    };
    auto ret = umf::memoryProviderMakeUnique<provider>(this->GetParam());
    ASSERT_EQ(ret.first, this->GetParam());
    ASSERT_EQ(ret.second, nullptr);
}
