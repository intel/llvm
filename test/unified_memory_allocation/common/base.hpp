/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMA_TEST_BASE_HPP
#define UMA_TEST_BASE_HPP 1

#include <gtest/gtest.h>

namespace uma_test {
struct test : ::testing::Test {
    void SetUp() override { ::testing::Test::SetUp(); }
    void TearDown() override { ::testing::Test::TearDown(); }
};
} // namespace uma_test

#endif /* UMA_TEST_BASE_HPP */
