// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "helpers.h"
#include "provider.h"

#include <string>
#include <unordered_map>

TEST_F(umaTest, memoryProviderTrace) {
    static std::unordered_map<std::string, size_t> calls;
    auto trace = [](const char *name) { calls[name]++; };

    auto nullProvider = uma::wrapProviderUnique(nullProviderCreate());
    auto tracingProvider =
        uma::wrapProviderUnique(traceProviderCreate(nullProvider.get(), trace));

    size_t call_count = 0;

    auto ret = umaMemoryProviderAlloc(tracingProvider.get(), 0, 0, nullptr);
    ASSERT_EQ(ret, UMA_RESULT_SUCCESS);
    ASSERT_EQ(calls["alloc"], 1);
    ASSERT_EQ(calls.size(), ++call_count);

    ret = umaMemoryProviderFree(tracingProvider.get(), nullptr, 0);
    ASSERT_EQ(ret, UMA_RESULT_SUCCESS);
    ASSERT_EQ(calls["free"], 1);
    ASSERT_EQ(calls.size(), ++call_count);
}
