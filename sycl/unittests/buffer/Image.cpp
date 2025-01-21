//==-------- buffer_location.cpp --- check buffer_location property --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

TEST(ImageTest, ImageGetSize) {
  constexpr size_t ElementsCount = 4;
  constexpr size_t ChannelsCount = 4;
  sycl::image<1> Image(sycl::image_channel_order::rgba,
                       sycl::image_channel_type::fp32, sycl::range<1>(ElementsCount));

  EXPECT_EQ(ElementsCount * ChannelsCount * sizeof(float), Image.get_size());
}
