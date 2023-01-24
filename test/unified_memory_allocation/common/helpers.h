// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_UMA_TEST_HELPERS_H
#define UR_UMA_TEST_HELPERS_H

#include <gtest/gtest.h>

#include <uma/memory_pool.h>
#include <uma/memory_provider.h>

#include <memory>

struct umaTest : ::testing::Test {
    void SetUp() override {}
    void TearDown() override {}
};

namespace uma {

auto wrapPoolUnique(uma_memory_pool_handle_t hPool) {
    return std::unique_ptr<uma_memory_pool_t, decltype(&umaPoolDestroy)>(
        hPool, &umaPoolDestroy);
}

auto wrapProviderUnique(uma_memory_provider_handle_t hProvider) {
    return std::unique_ptr<uma_memory_provider_t,
                           decltype(&umaMemoryProviderDestroy)>(
        hProvider, &umaMemoryProviderDestroy);
}

} // namespace uma

#endif // UR_UMA_TEST_HELPERS_H
