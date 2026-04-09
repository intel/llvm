//==---- IdQueriesRangeValidation.cpp - Range validation unit tests -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// These tests validate the range checking logic for UINT mode by directly
// calling the checkValueRange functions from
// sycl/detail/id_queries_fit_in_int.hpp. The __SYCL_ID_QUERIES_FIT_IN_UINT__
// macro is defined via target_compile_definitions in the CMakeLists.txt file.

#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <climits>

using namespace sycl;

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
