//==---- PropertyUtils.cpp -------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <string_view>

#include <sycl/ext/oneapi/properties/property_utils.hpp>

using namespace sycl::ext::oneapi::experimental::detail;

void checkEquality(std::string_view LHS, std::string_view RHS) {
  ASSERT_EQ(LHS, RHS);
}

TEST(PropertyUtilsTest, SizeListToStrTest) {
  checkEquality(SizeListToStr<>::value, "");
  checkEquality(SizeListToStr<0>::value, "0");
  checkEquality(SizeListToStr<1>::value, "1");
  checkEquality(SizeListToStr<42>::value, "42");
  checkEquality(SizeListToStr<123>::value, "123");
  checkEquality(SizeListToStr<4321>::value, "4321");
  checkEquality(SizeListToStr<0, 1>::value, "0,1");
  checkEquality(SizeListToStr<1, 0>::value, "1,0");
  checkEquality(SizeListToStr<42, 43>::value, "42,43");
  checkEquality(SizeListToStr<0, 1, 42>::value, "0,1,42");
  checkEquality(SizeListToStr<1, 0, 42>::value, "1,0,42");
  checkEquality(SizeListToStr<1, 42, 0>::value, "1,42,0");
  checkEquality(SizeListToStr<0, 1, 42, 4321>::value, "0,1,42,4321");
  checkEquality(SizeListToStr<1, 0, 42, 4321>::value, "1,0,42,4321");
  checkEquality(SizeListToStr<1, 42, 0, 4321>::value, "1,42,0,4321");
  checkEquality(SizeListToStr<1, 42, 4321, 0>::value, "1,42,4321,0");
}
