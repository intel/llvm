//==--------------- SpanTests.cpp - Unit tests for sycl::span -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests sycl::span functionality not covered by other test files:
// - STL algorithm compatibility
// - Iterator arithmetic operations
// - Byte span conversions (as_bytes, as_writable_bytes)
// - Alignment preservation
// - Edge cases (nullptr construction, etc.)
// - C++20 spec features not tested elsewhere
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <numeric>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

class SpanTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(SpanTest, DefaultConstructedSpanIsEmpty) {
  sycl::span<int> empty_span;
  EXPECT_EQ(empty_span.size(), 0u);
  EXPECT_TRUE(empty_span.empty());
}

TEST_F(SpanTest, CanConstructFromNullptrWithSize) {
  int *null_ptr = nullptr;
  constexpr auto size = 42u;

  sycl::span<int> sp_null(null_ptr, size);
  EXPECT_EQ(sp_null.size(), size);
  EXPECT_EQ(sp_null.data(), nullptr);
}

TEST_F(SpanTest, AccessElementsByIndexAndFrontBack) {
  auto arr[]{1, 2, 3};
  sycl::span sp(arr, std::size(arr));
  const auto size = sp.size();

  for (auto i{0u}; i != size; ++i)
    EXPECT_EQ(sp[i], i + 1);

  EXPECT_EQ(sp.front(), 1);
  EXPECT_EQ(sp.back(), 3);
}

TEST_F(SpanTest, BeginEndAndReverseIteratorsWork) {
  auto arr[]{1, 2, 3, 4, 5};
  sycl::span sp(arr, std::size(arr));

  auto it = sp.begin();
  EXPECT_EQ(*it, 1);
  EXPECT_EQ(*(it + 1), 2);
  EXPECT_EQ(sp.end() - sp.begin(), 5);

  auto rit = sp.rbegin();
  EXPECT_EQ(*rit, 5);
  EXPECT_EQ(*(rit + 1), 4);
}

TEST_F(SpanTest, ConstBeginEndIteratorsOnConstSpan) {
  auto arr[]{10, 20, 30};
  const sycl::span sp(arr, std::size(arr));

  auto cit = sp.cbegin();
  EXPECT_EQ(*cit, 10);

  auto rcit = sp.crbegin();
  EXPECT_EQ(*rcit, 30);
}

TEST_F(SpanTest, SubspanReturnsCorrectSlice) {
  auto arr[]{10, 20, 30, 40, 50};
  sycl::span sp(arr, std::size(arr));

  auto sub = sp.subspan(1, 3);
  EXPECT_EQ(sub.size(), 3u);
  EXPECT_EQ(sub[0], 20);
  EXPECT_EQ(sub[1], 30);
  EXPECT_EQ(sub[2], 40);
}

TEST_F(SpanTest, FirstAndLastReturnCorrectSubspans) {
  auto arr[]{10, 20, 30, 40, 50};
  sycl::span sp(arr, std::size(arr));

  auto first2 = sp.first(2);
  EXPECT_EQ(first2.size(), 2u);
  EXPECT_EQ(first2[0], 10);
  EXPECT_EQ(first2[1], 20);

  auto last2 = sp.last(2);
  EXPECT_EQ(last2.size(), 2u);
  EXPECT_EQ(last2[0], 40);
  EXPECT_EQ(last2[1], 50);
}

TEST_F(SpanTest, AsBytesReturnsReadOnlyByteView) {
  auto arr[]{10, 20, 30, 40, 50};
  sycl::span sp(arr, std::size(arr));

  auto bytes = sycl::as_bytes(sp);
  EXPECT_EQ(bytes.size(), sizeof(int) * 5);
  EXPECT_EQ(static_cast<const void *>(bytes.data()),
            static_cast<const void *>(arr));
}

TEST_F(SpanTest, AsWritableBytesAllowsByteWiseModification) {
  auto arr[]{10, 20, 30, 40, 50};
  sycl::span sp(arr, std::size(arr));

  auto writable_bytes = sycl::as_writable_bytes(sp);
  EXPECT_EQ(writable_bytes.size(), sizeof(int) * 5);
  EXPECT_EQ(static_cast<void *>(writable_bytes.data()),
            static_cast<void *>(arr));

  writable_bytes[0] = 0xFF;
  auto *byte_ptr = reinterpret_cast<unsigned char *>(arr);
  EXPECT_EQ(byte_ptr[0], 0xFF);
}

TEST_F(SpanTest, SizeBytesReturnsCorrectByteCount) {
  auto arr[]{1, 2, 3, 4, 5};
  sycl::span sp(arr, std::size(arr));

  EXPECT_EQ(sp.size_bytes(), sizeof(int) * 5);
}

TEST_F(SpanTest, EmptyMethodDetectsEmptySpans) {
  auto arr[]{1, 2, 3, 4, 5};
  sycl::span sp(arr, std::size(arr));
  sycl::span<int> empty_sp;

  EXPECT_FALSE(sp.empty());
  EXPECT_TRUE(empty_sp.empty());
}

TEST_F(SpanTest, RangeBasedForLoopIteratesAllElements) {
  auto arr[]{1, 2, 3, 4, 5};
  sycl::span sp(arr, std::size(arr));

  int sum = 0;
  for (const auto &elem : sp) {
    sum += elem;
  }
  EXPECT_EQ(sum, 15);
}

TEST_F(SpanTest, ModifyingSpanElementsModifiesUnderlying) {
  auto arr[]{1, 2, 3};
  sycl::span sp(arr, std::size(arr));

  sp[0] = 10;
  sp[1] = 20;
  sp[2] = 30;

  EXPECT_EQ(arr[0], 10);
  EXPECT_EQ(arr[1], 20);
  EXPECT_EQ(arr[2], 30);
}

TEST_F(SpanTest, SpanPreservesOriginalAlignment) {
  alignas(16) auto aligned_arr[]{1, 2, 3, 4};
  sycl::span sp(aligned_arr, std::size(aligned_arr));

  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(sp.data()) % 16, 0u);
}

TEST_F(SpanTest, IteratorReflectsUnderlyingDataChanges) {
  auto arr[]{1, 2, 3};
  sycl::span sp(arr, std::size(arr));
  auto it = sp.begin();
  arr[0] = 42;
  EXPECT_EQ(*it, 42); // Iterator should see the change
}

TEST_F(SpanTest, ConstReverseIteratorsProvideReverseAccess) {
  auto arr[]{1, 2, 3, 4, 5};
  sycl::span sp(arr, std::size(arr));

  auto crit = sp.crbegin();
  auto crend = sp.crend();
  EXPECT_EQ(*crit, 5);
  EXPECT_EQ(*(crend - 1), 1);
}

TEST_F(SpanTest, SpanWorksWithSTLAlgorithms) {
  auto arr[]{1, 2, 3, 4, 5, 6};
  sycl::span sp(arr, std::size(arr));

  auto found = std::find(sp.begin(), sp.end(), 4);
  EXPECT_NE(found, sp.end());
  EXPECT_EQ(*found, 4);
  EXPECT_EQ(found - sp.begin(), 3);

  int count_ones = std::count(sp.begin(), sp.end(), 1);
  EXPECT_EQ(count_ones, 1);

  int sum = std::accumulate(sp.begin(), sp.end(), 0);
  EXPECT_EQ(sum, 21); // 1+2+3+4+5+6

  bool all_positive =
      std::all_of(sp.begin(), sp.end(), [](int x) { return x > 0; });
  EXPECT_TRUE(all_positive);

  bool has_four =
      std::any_of(sp.begin(), sp.end(), [](int x) { return x == 4; });
  EXPECT_TRUE(has_four);

  bool no_negative =
      std::none_of(sp.begin(), sp.end(), [](int x) { return x < 0; });
  EXPECT_TRUE(no_negative);

  int result[6];
  std::transform(sp.begin(), sp.end(), result, [](int x) { return x * 2; });
  EXPECT_EQ(result[0], 2);
  EXPECT_EQ(result[5], 12);
}

TEST_F(SpanTest, IteratorSupportsFullArithmeticOperations) {
  auto arr[]{10, 20, 30, 40, 50};
  sycl::span sp(arr, std::size(arr));

  auto it = sp.begin();

  EXPECT_EQ(*it, 10);
  ++it;
  EXPECT_EQ(*it, 20);
  --it;
  EXPECT_EQ(*it, 10);

  it += 2;
  EXPECT_EQ(*it, 30);
  it -= 1;
  EXPECT_EQ(*it, 20);

  EXPECT_EQ(it[0], 20);
  EXPECT_EQ(it[2], 40);
  EXPECT_EQ(it[-1], 10);

  auto it2 = it + 2;
  EXPECT_EQ(*it2, 40);
  EXPECT_EQ(it2 - it, 2);

  EXPECT_TRUE(it < it2);
  EXPECT_TRUE(it <= it2);
  EXPECT_TRUE(it2 > it);
  EXPECT_TRUE(it2 >= it);
  EXPECT_FALSE(it == it2);
  EXPECT_TRUE(it != it2);
}

// C++20 spec tests missing from other test files

TEST_F(SpanTest, AtMethodThrowsOutOfRange) {
  auto arr[]{1, 2, 3};
  sycl::span sp(arr, std::size(arr));

  // Valid access
  EXPECT_EQ(sp.at(0), 1);
  EXPECT_EQ(sp.at(2), 3);

  // Out of range access should throw (on host)
#if !defined(__SYCL_DEVICE_ONLY__)
  EXPECT_THROW(sp.at(3), std::out_of_range);
  EXPECT_THROW(sp.at(10), std::out_of_range);

  // Test with dynamic extent
  sycl::span<int, sycl::dynamic_extent> dyn_sp(arr, 3);
  EXPECT_EQ(dyn_sp.at(0), 1);
  EXPECT_EQ(dyn_sp.at(2), 3);
  EXPECT_THROW(dyn_sp.at(3), std::out_of_range);
#endif // !defined(__SYCL_DEVICE_ONLY__)
}

TEST_F(SpanTest, InitializerListConstructor) {
  std::initializer_list<int> il = {5, 10, 15, 20};
  sycl::span<const int> sp(il.begin(), il.size());

  EXPECT_EQ(sp.size(), 4u);
  EXPECT_EQ(sp[0], 5);
  EXPECT_EQ(sp[3], 20);
}

// Note: These are C++20 template versions of first/last/subspan:
// If not supported, tests will use runtime versions instead
TEST_F(SpanTest, CompileTimeFirstAndLast) {
  auto arr[]{10, 20, 30, 40, 50};
  sycl::span sp(arr, std::size(arr));

  // Runtime versions (always available)
  auto first3 = sp.first(3);
  EXPECT_EQ(first3.size(), 3u);
  EXPECT_EQ(first3[0], 10);
  EXPECT_EQ(first3[2], 30);

  auto last2 = sp.last(2);
  EXPECT_EQ(last2.size(), 2u);
  EXPECT_EQ(last2[0], 40);
  EXPECT_EQ(last2[1], 50);
  auto first3_template = sp.first<3>();
  static_assert(decltype(first3_template)::extent == 3);
  EXPECT_EQ(first3_template.size(), 3u);
  EXPECT_EQ(first3_template[0], 10);

  auto last2_template = sp.last<2>();
  static_assert(decltype(last2_template)::extent == 2);
  EXPECT_EQ(last2_template.size(), 2u);
  EXPECT_EQ(last2_template[0], 40);
}

TEST_F(SpanTest, CompileTimeSubspan) {
  auto arr[]{1, 2, 3, 4, 5, 6};
  sycl::span sp(arr, std::size(arr));

  // Runtime version (always available)
  auto sub1 = sp.subspan(2, 3);
  EXPECT_EQ(sub1.size(), 3u);
  EXPECT_EQ(sub1[0], 3);
  EXPECT_EQ(sub1[2], 5);

  auto sub2 = sp.subspan(4);
  EXPECT_EQ(sub2.size(), 2u);
  EXPECT_EQ(sub2[0], 5);
  EXPECT_EQ(sub2[1], 6);

  auto sub1_template = sp.subspan<2, 3>();
  static_assert(decltype(sub1_template)::extent == 3);
  EXPECT_EQ(sub1_template.size(), 3u);
  EXPECT_EQ(sub1_template[0], 3);

  auto sub2_template = sp.subspan<4>();
  static_assert(decltype(sub2_template)::extent == sycl::dynamic_extent);
  EXPECT_EQ(sub2_template.size(), 2u);
  EXPECT_EQ(sub2_template[0], 5);
}
