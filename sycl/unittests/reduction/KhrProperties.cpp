//==---- KhrProperties.cpp --- sycl_khr_properties reduction properties ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

namespace kp = sycl::khr::property;
using sycl::khr::properties;

TEST(KhrReductionProperties, InitializeToIdentityUSM) {
  int ReduVar = 0;
  EXPECT_TRUE(
      sycl::reduction(&ReduVar, sycl::plus<int>(), kp::initialize_to_identity{})
          .initializeToIdentity());
  EXPECT_TRUE(sycl::reduction(&ReduVar, sycl::plus<int>(),
                              kp::initialize_to_identity{true})
                  .initializeToIdentity());
  EXPECT_FALSE(sycl::reduction(&ReduVar, sycl::plus<int>(),
                               kp::initialize_to_identity{false})
                   .initializeToIdentity());
  EXPECT_TRUE(sycl::reduction(&ReduVar, 0, sycl::plus<int>(),
                              properties{kp::initialize_to_identity{}})
                  .initializeToIdentity());
  EXPECT_FALSE(sycl::reduction(&ReduVar, 0, sycl::plus<int>(),
                               properties{kp::initialize_to_identity{false}})
                   .initializeToIdentity());
  EXPECT_FALSE(sycl::reduction(&ReduVar, sycl::plus<int>(), properties{})
                   .initializeToIdentity());
}

TEST(KhrReductionProperties, InitializeToIdentitySpan) {
  int ReduVars[4] = {};
  sycl::span<int, 4> Span{ReduVars};
  EXPECT_TRUE(
      sycl::reduction(Span, sycl::plus<int>(), kp::initialize_to_identity{})
          .initializeToIdentity());
  EXPECT_FALSE(sycl::reduction(Span, 0, sycl::plus<int>(),
                               kp::initialize_to_identity{false})
                   .initializeToIdentity());
}
