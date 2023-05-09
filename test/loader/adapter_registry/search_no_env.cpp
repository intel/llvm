// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "fixtures.hpp"

TEST_F(adapterRegSearchTest, testSearchNoEnv) {
    // Check if there's any path that's just a library name.
    auto testLibNameExists =
        std::any_of(registry.cbegin(), registry.cend(), hasTestLibName);
    ASSERT_TRUE(testLibNameExists);

    // Check for path obtained from 'UR_ADAPTERS_SEARCH_PATH'
    auto testEnvPathExists =
        std::any_of(registry.cbegin(), registry.cend(), hasTestEnvPath);
    ASSERT_FALSE(testEnvPathExists);

    // Check for current directory path
    auto testCurPathExists =
        std::any_of(registry.cbegin(), registry.cend(), hasCurPath);
    ASSERT_TRUE(testCurPathExists);
}
