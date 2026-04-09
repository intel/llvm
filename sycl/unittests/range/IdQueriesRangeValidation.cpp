//==---- IdQueriesRangeValidation.cpp - Range validation unit tests -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/id_queries_fit_in_int.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <climits>

using namespace sycl;

// These tests validate the range checking logic by directly calling the
// checkValueRange functions from sycl/detail/id_queries_fit_in_int.hpp.
//
// The tests are compiled three times with different preprocessor macros:
// - __SYCL_ID_QUERIES_FIT_IN_INT__=1 for INT mode
// - __SYCL_ID_QUERIES_FIT_IN_UINT__=1 for UINT mode
// - Neither macro defined for size_t mode

#if __SYCL_ID_QUERIES_FIT_IN_INT__

// Tests for INT mode (when __SYCL_ID_QUERIES_FIT_IN_INT__ is defined)
TEST(IdQueriesRangeValidation, Int_Range1D_AtLimit) {
  range<1> r(INT_MAX);
  EXPECT_NO_THROW(detail::checkValueRange(r));
}

TEST(IdQueriesRangeValidation, Int_Range1D_ExceedsLimit) {
  // Skip if size_t can't hold values larger than INT_MAX
  if (sizeof(size_t) <= sizeof(int)) {
    GTEST_SKIP() << "size_t too small to test overflow beyond INT_MAX";
  }
  range<1> r(static_cast<size_t>(INT_MAX) + 1);
  EXPECT_THROW(detail::checkValueRange(r), exception);
}

TEST(IdQueriesRangeValidation, Int_Range2D_ProductExceedsLimit) {
  // 46341 * 46341 = 2147488281 > INT_MAX (2147483647)
  range<2> r(46341, 46341);
  EXPECT_THROW(detail::checkValueRange(r), exception);
}

TEST(IdQueriesRangeValidation, Int_Range2D_ProductAtLimit) {
  // 46340 * 46340 = 2147395600 < INT_MAX
  range<2> r(46340, 46340);
  EXPECT_NO_THROW(detail::checkValueRange(r));
}

TEST(IdQueriesRangeValidation, Int_Range3D_ProductExceedsLimit) {
  // 1290 * 1290 * 1290 = 2146689000 < INT_MAX, but 1291^3 > INT_MAX
  range<3> r(1291, 1291, 1291);
  EXPECT_THROW(detail::checkValueRange(r), exception);
}

TEST(IdQueriesRangeValidation, Int_Range3D_ProductAtLimit) {
  range<3> r(1290, 1290, 1290);
  EXPECT_NO_THROW(detail::checkValueRange(r));
}

TEST(IdQueriesRangeValidation, Int_Id_ComponentExceedsLimit) {
  // Skip if size_t can't hold values larger than INT_MAX
  if (sizeof(size_t) <= sizeof(int)) {
    GTEST_SKIP() << "size_t too small to test overflow beyond INT_MAX";
  }
  id<3> offset(1, static_cast<size_t>(INT_MAX) + 1, 1);
  EXPECT_THROW(detail::checkValueRange(offset), exception);
}

TEST(IdQueriesRangeValidation, Int_Id_ComponentAtLimit) {
  id<3> offset(1, INT_MAX, 1);
  EXPECT_NO_THROW(detail::checkValueRange(offset));
}

TEST(IdQueriesRangeValidation, Int_RangeWithOffset_SumExceedsLimit) {
  range<1> r(INT_MAX);
  id<1> offset(1);
  EXPECT_THROW(detail::checkValueRange(r, offset), exception);
}

TEST(IdQueriesRangeValidation, Int_RangeWithOffset_SumAtLimit) {
  range<1> r(INT_MAX - 1);
  id<1> offset(1);
  EXPECT_NO_THROW(detail::checkValueRange(r, offset));
}

#elif __SYCL_ID_QUERIES_FIT_IN_UINT__

// Tests for UINT mode (when __SYCL_ID_QUERIES_FIT_IN_UINT__ is defined)
TEST(IdQueriesRangeValidation, UInt_Range1D_AtLimit) {
  range<1> r(UINT_MAX);
  EXPECT_NO_THROW(detail::checkValueRange(r));
}

TEST(IdQueriesRangeValidation, UInt_Range1D_ExceedsLimit) {
  // Skip if size_t can't hold values larger than UINT_MAX
  if constexpr (sizeof(size_t) <= sizeof(unsigned int)) {
    GTEST_SKIP() << "size_t too small to test overflow beyond UINT_MAX";
  }
  range<1> r(static_cast<size_t>(UINT_MAX) + 1);
  EXPECT_THROW(detail::checkValueRange(r), exception);
}

TEST(IdQueriesRangeValidation, UInt_Range1D_AboveIntMax) {
  // Skip if size_t can't hold values larger than INT_MAX
  if constexpr (sizeof(size_t) <= sizeof(int)) {
    GTEST_SKIP() << "size_t too small to test values above INT_MAX";
  }
  // This should succeed in UINT mode but would fail in INT mode
  range<1> r(static_cast<size_t>(INT_MAX) + 1);
  EXPECT_NO_THROW(detail::checkValueRange(r));
}

TEST(IdQueriesRangeValidation, UInt_Range2D_ProductExceedsLimit) {
  // 65536 * 65536 = 4294967296 > UINT_MAX (4294967295)
  range<2> r(65536, 65536);
  EXPECT_THROW(detail::checkValueRange(r), exception);
}

TEST(IdQueriesRangeValidation, UInt_Range2D_ProductAtLimit) {
  // 65535 * 65535 = 4294836225 < UINT_MAX
  range<2> r(65535, 65535);
  EXPECT_NO_THROW(detail::checkValueRange(r));
}

TEST(IdQueriesRangeValidation, UInt_Id_ComponentExceedsLimit) {
  // Skip if size_t can't hold values larger than UINT_MAX
  if constexpr (sizeof(size_t) <= sizeof(unsigned int)) {
    GTEST_SKIP() << "size_t too small to test overflow beyond UINT_MAX";
  }
  id<3> offset(1, static_cast<size_t>(UINT_MAX) + 1, 1);
  EXPECT_THROW(detail::checkValueRange(offset), exception);
}

TEST(IdQueriesRangeValidation, UInt_Id_ComponentAtLimit) {
  id<3> offset(1, UINT_MAX, 1);
  EXPECT_NO_THROW(detail::checkValueRange(offset));
}

TEST(IdQueriesRangeValidation, UInt_RangeWithOffset_SumExceedsLimit) {
  range<1> r(UINT_MAX);
  id<1> offset(1);
  EXPECT_THROW(detail::checkValueRange(r, offset), exception);
}

TEST(IdQueriesRangeValidation, UInt_RangeWithOffset_SumAtLimit) {
  range<1> r(UINT_MAX - 1);
  id<1> offset(1);
  EXPECT_NO_THROW(detail::checkValueRange(r, offset));
}

#else

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

#endif
