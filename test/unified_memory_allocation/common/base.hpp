/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef UMA_TEST_BASE_HPP
#define UMA_TEST_BASE_HPP 1

#include <gtest/gtest.h>

namespace uma_test {
struct test : ::testing::Test {
    void SetUp() { ::testing::Test::SetUp(); }
    void TearDown() override { ::testing::Test::TearDown(); }
};
} // namespace uma_test

#endif /* UMA_TEST_BASE_HPP */
