// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CI Validation Tests
// These tests intentionally generate different test outcomes
// to validate CI logging and categorization

#include <gtest/gtest.h>

// Test that always passes
TEST(CIValidation, test_pass) { EXPECT_EQ(42, 42); }

// Test that always fails with detailed error information
TEST(CIValidation, test_fail) {
  int expected = 42;
  int actual = 0;  // Changed to 0 to make the mismatch more obvious

  EXPECT_EQ(expected, actual)
    << "Memory allocation test failed!" << std::endl
    << "Expected value: " << expected << std::endl
    << "Actual value: " << actual << std::endl
    << "This simulates a real test failure with detailed diagnostics";
}

// Test expected to fail (XFAIL equivalent in Google Test)
// Disabled tests are skipped, which is close to XFAIL behavior
TEST(CIValidation, DISABLED_test_xfail) {
  EXPECT_EQ(42, 41) << "Test failed: expected 42, got 41";
}

// Test expected to fail but actually passes (XPASS)
// This will show as disabled/skipped, but if someone runs it manually it passes
TEST(CIValidation, DISABLED_test_unexpected_pass) { EXPECT_EQ(42, 42); }

// Test unsupported on common platforms
TEST(CIValidation, DISABLED_test_unsupported) { EXPECT_EQ(42, 42); }

// Test that times out (infinite loop)
TEST(CIValidation, test_timeout) {
  while (true) {
    // Infinite loop to trigger timeout
  }
}
