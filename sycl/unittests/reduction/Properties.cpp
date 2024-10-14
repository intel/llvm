//==-------- Properties.cpp --- check properties handling in RT --- --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <sycl/properties/reduction_properties.hpp>
#include <sycl/sycl.hpp>

TEST(ReductionTest, InvalidProperties) {
  int ReduVar = 0;
  try {
    auto Redu = sycl::reduction(
        &ReduVar, int{0}, sycl::plus<>(),
        sycl::property_list{sycl::property::buffer::use_host_ptr{}});
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }

  FAIL() << "Test must exit in exception handler. Exception is not thrown.";
}

TEST(ReductionTest, ValidPropertyInitializeToIdentity) {
  int ReduVar = 0;
  try {
    auto Redu = sycl::reduction(
        &ReduVar, int{0}, sycl::plus<>(),
        sycl::property_list{
            sycl::property::reduction::initialize_to_identity{}});
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}
