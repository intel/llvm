//==------------------------ LibSYCLTimestamp.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/version.hpp>

#include <gtest/gtest.h>

#include <string>
#include <cctype>

#define _STR(e) #e
#define STR(e) _STR(e)

TEST(LibSYCLTimestamp, Format) {
  // __LIBSYCL_TIMESTAMP format is YYYYMMDD
  std::string Timestamp = STR(__LIBSYCL_TIMESTAMP);

  ASSERT_EQ(Timestamp.size(), 8u);

  for (char C : Timestamp) {
    ASSERT_TRUE(std::isdigit(C));
  }

  constexpr size_t Y0 = 0;
  constexpr size_t Y1 = 1;
  constexpr size_t M0 = 4;
  constexpr size_t M1 = 5;
  constexpr size_t D0 = 6;
  constexpr size_t D1 = 7;

  // Safe enough test for the next 900+ years
  ASSERT_EQ(Timestamp[Y0], '2');
  // Safe enough test for the next 70+ years
  ASSERT_EQ(Timestamp[Y1], '0');

  ASSERT_TRUE(Timestamp[M0] == '0' || Timestamp[M1] == '1');
  if (Timestamp[M0] == '1')
    ASSERT_TRUE(Timestamp[M1] >= '0' && Timestamp[M1] <= '2');
  ASSERT_FALSE(Timestamp[M0] == '0' && Timestamp[M1] == '0');

  ASSERT_TRUE(Timestamp[D0] >= '0' && Timestamp[D0] <= '3');
  if (Timestamp[D0] == '3')
    ASSERT_TRUE(Timestamp[D1] == '0' || Timestamp[D1] == '1');
  ASSERT_FALSE(Timestamp[D0] == '0' && Timestamp[D1] == '0');
}

TEST(LibSYCLTimestasmp, BasicAcceptance) {
  // Date when this feature was introduced
  ASSERT_TRUE(__LIBSYCL_TIMESTAMP >= 20241128);
}
