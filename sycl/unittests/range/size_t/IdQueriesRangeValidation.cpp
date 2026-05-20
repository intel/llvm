//==---- IdQueriesRangeValidation.cpp - Range validation unit tests -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// These tests validate the range checking logic for size_t mode by directly
// calling the checkValueRange functions from
// sycl/detail/id_queries_fit_in_int.hpp. In size_t mode, neither
// __SYCL_ID_QUERIES_FIT_IN_INT__ nor
// __SYCL_ID_QUERIES_FIT_IN_UINT__ is defined (no validation occurs).

#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <climits>

using namespace sycl;

// Tests for size_t mode (no validation macros defined)
TEST(IdQueriesRangeValidation, SizeT_NoValidation_LargeRange) {
  // Skip if size_t can't hold values larger than UINT_MAX
  if constexpr (sizeof(size_t) <= sizeof(unsigned int)) {
    GTEST_SKIP() << "size_t too small to test values beyond UINT_MAX";
  }
  // In size_t mode, no validation occurs, so even huge values should not throw
  range<1> r(static_cast<size_t>(UINT_MAX) + 1);
  EXPECT_NO_THROW(detail::checkValueRange(r));
}

TEST(IdQueriesRangeValidation, SizeT_NoValidation_LargeId) {
  // Skip if size_t can't hold values larger than UINT_MAX
  if constexpr (sizeof(size_t) <= sizeof(unsigned int)) {
    GTEST_SKIP() << "size_t too small to test values beyond UINT_MAX";
  }
  id<3> offset(1, static_cast<size_t>(UINT_MAX) + 1, 1);
  EXPECT_NO_THROW(detail::checkValueRange(offset));
}

TEST(IdQueriesRangeValidation, SizeT_NoValidation_LargeRangeWithOffset) {
  // Skip if size_t can't hold values larger than UINT_MAX
  if constexpr (sizeof(size_t) <= sizeof(unsigned int)) {
    GTEST_SKIP() << "size_t too small to test values beyond UINT_MAX";
  }
  range<1> r(static_cast<size_t>(UINT_MAX) + 1);
  id<1> offset(1);
  EXPECT_NO_THROW(detail::checkValueRange(r, offset));
}
