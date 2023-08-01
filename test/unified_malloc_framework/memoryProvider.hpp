// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider.hpp"

#include <cstring>
#include <functional>

#ifndef UMF_TEST_MEMORY_PROVIDER_OPS_HPP
#define UMF_TEST_MEMORY_PROVIDER_OPS_HPP

struct umfProviderTest
    : umf_test::test,
      ::testing::WithParamInterface<std::function<
          std::pair<umf_result_t, umf::provider_unique_handle_t>()>> {
    umfProviderTest() : provider(nullptr, nullptr) {}
    void SetUp() {
        test::SetUp();
        auto [res, provider] = this->GetParam()();
        EXPECT_EQ(res, UMF_RESULT_SUCCESS);
        EXPECT_NE(provider, nullptr);
        this->provider = std::move(provider);
    }
    void TearDown() override { test::TearDown(); }
    umf::provider_unique_handle_t provider;
};

#endif /* UMF_TEST_MEMORY_PROVIDER_OPS_HPP */
