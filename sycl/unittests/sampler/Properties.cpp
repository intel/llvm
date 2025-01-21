//==-------- Properties.cpp --- check properties handling in RT --- --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

TEST(SamplerTest, Properties) {
  try {
    sycl::sampler Sampler(
        sycl::coordinate_normalization_mode::unnormalized,
        sycl::addressing_mode::clamp, sycl::filtering_mode::nearest,
        sycl::property_list{sycl::property::buffer::use_host_ptr{}});
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }

  FAIL() << "Test must exit in exception handler. Exception is not thrown.";
}
