//==---- IdQueriesRangeValidation.cpp - Range validation unit tests -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// These tests validate the range checking logic for INT mode by directly
// calling the checkValueRange functions from
// sycl/detail/id_queries_fit_in_int.hpp. The __SYCL_ID_QUERIES_FIT_IN_INT__
// macro is defined via target_compile_definitions in the CMakeLists.txt file.

#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <climits>

using namespace sycl;

TEST(IdQueriesRangeValidation, Int_Range1D_AtLimit) {
  range<1> r(INT_MAX);
  EXPECT_NO_THROW(detail::checkValueRange(r));
}

TEST(IdQueriesRangeValidation, Int_Range1D_ExceedsLimit) {
  // Skip if size_t can't hold values larger than INT_MAX
  if constexpr (sizeof(size_t) <= sizeof(int)) {
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
  if constexpr (sizeof(size_t) <= sizeof(int)) {
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
