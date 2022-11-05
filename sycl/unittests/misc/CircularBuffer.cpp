//==---- CircularBuffer.cpp ------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <detail/circular_buffer.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

// This test contains basic checks for sycl::detail::CircularBuffer
void checkEquality(const sycl::detail::CircularBuffer<int> &CB,
                   const std::vector<int> &V) {
  ASSERT_TRUE(std::equal(CB.begin(), CB.end(), V.begin()));
}

TEST(CircularBufferTest, CircularBufferTest) {
  const std::size_t Capacity = 6;
  sycl::detail::CircularBuffer<int> CB{Capacity};
  ASSERT_TRUE(CB.capacity() == Capacity);
  ASSERT_TRUE(CB.empty());

  size_t nextValue = 0;
  for (; nextValue < Capacity; ++nextValue) {
    ASSERT_TRUE(CB.size() == nextValue);
    CB.push_back(nextValue);
  }
  ASSERT_TRUE(CB.full() && CB.size() == CB.capacity());
  checkEquality(CB, {0, 1, 2, 3, 4, 5});

  CB.push_back(nextValue++);
  checkEquality(CB, {1, 2, 3, 4, 5, 6});
  CB.push_front(nextValue++);
  checkEquality(CB, {7, 1, 2, 3, 4, 5});

  ASSERT_TRUE(CB.front() == 7);
  ASSERT_TRUE(CB.back() == 5);

  CB.erase(CB.begin() + 2);
  checkEquality(CB, {7, 1, 3, 4, 5});
  CB.erase(CB.begin(), CB.begin() + 2);
  checkEquality(CB, {3, 4, 5});

  CB.pop_back();
  checkEquality(CB, {3, 4});
  CB.pop_front();
  checkEquality(CB, {4});
}
