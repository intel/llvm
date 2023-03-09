// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "provider.hpp"

#include <cstring>
#include <functional>

#ifndef UMA_TEST_MEMORY_PROVIDER_OPS_HPP
#define UMA_TEST_MEMORY_PROVIDER_OPS_HPP

struct umaProviderTest : uma_test::test, ::testing::WithParamInterface<std::function<std::pair<uma_result_t, uma::provider_unique_handle_t>()>> {
    umaProviderTest() : provider(nullptr, nullptr) {}
    void SetUp() {
        test::SetUp();
        auto [res, provider] = this->GetParam()();
        EXPECT_EQ(res, UMA_RESULT_SUCCESS);
        EXPECT_NE(provider, nullptr);
        this->provider = std::move(provider);
    }
    void TearDown() override {
        test::TearDown();
    }
    uma::provider_unique_handle_t provider;
};

#endif /* UMA_TEST_MEMORY_PROVIDER_OPS_HPP */
