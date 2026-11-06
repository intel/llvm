//==---- KhrProperties.cpp --- sycl_khr_properties reduction properties ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <type_traits>

namespace kp = sycl::khr::property;
namespace syclex = sycl::ext::oneapi::experimental;
using namespace sycl::khr;

struct OtherClass {};

static_assert(is_property_v<kp::initialize_to_identity>);
static_assert(is_property_key_v<kp::key::initialize_to_identity>);
static_assert(!is_property_key_compile_time_v<kp::key::initialize_to_identity>);
static_assert(
    is_property_key_for_v<kp::key::initialize_to_identity, kp::tag::reduction>);
static_assert(
    is_property_for_v<kp::initialize_to_identity, kp::tag::reduction>);
static_assert(is_property_list_for_v<properties<kp::initialize_to_identity>,
                                     kp::tag::reduction>);
static_assert(!is_property_for_v<kp::initialize_to_identity, OtherClass>);
static_assert(!is_property_for_v<kp::initialize_to_identity, sycl::queue>);
static_assert(std::is_trivially_copyable_v<kp::initialize_to_identity>);
static_assert(std::is_trivially_copyable_v<kp::key::initialize_to_identity>);

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
  EXPECT_TRUE(sycl::reduction(Span, sycl::plus<int>(),
                              properties{kp::initialize_to_identity{}})
                  .initializeToIdentity());
  EXPECT_FALSE(sycl::reduction(Span, 0, sycl::plus<int>(),
                               kp::initialize_to_identity{false})
                   .initializeToIdentity());
}

TEST(KhrReductionProperties, InitializeToIdentityBuffer) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Q;
  sycl::buffer<int, 1> Buf{1};
  Q.submit([&](sycl::handler &CGH) {
    EXPECT_TRUE(sycl::reduction(Buf, CGH, sycl::plus<int>(),
                                kp::initialize_to_identity{})
                    .initializeToIdentity());
    EXPECT_FALSE(sycl::reduction(Buf, CGH, 0, sycl::plus<int>(),
                                 properties{kp::initialize_to_identity{false}})
                     .initializeToIdentity());
  });
}

TEST(KhrReductionProperties, OldStyleOverloads) {
  int ReduVar = 0;
  // Old-style construction must still resolve unambiguously.
  EXPECT_FALSE(
      sycl::reduction(&ReduVar, sycl::plus<int>()).initializeToIdentity());
  EXPECT_FALSE(
      sycl::reduction(&ReduVar, sycl::plus<int>(), {}).initializeToIdentity());
  EXPECT_TRUE(
      sycl::reduction(&ReduVar, sycl::plus<int>(),
                      sycl::property_list{
                          sycl::property::reduction::initialize_to_identity{}})
          .initializeToIdentity());
  // The experimental properties overloads coexist with the khr ones.
  EXPECT_TRUE(
      sycl::reduction(&ReduVar, sycl::plus<int>(),
                      syclex::properties(syclex::initialize_to_identity))
          .initializeToIdentity());
}

TEST(KhrReductionProperties, UnqualifiedCall) {
  using namespace sycl;
  int ReduVar = 0;
  // The reduction tag must not shadow sycl::reduction.
  EXPECT_TRUE(reduction(&ReduVar, plus<int>(), kp::initialize_to_identity{})
                  .initializeToIdentity());
}
