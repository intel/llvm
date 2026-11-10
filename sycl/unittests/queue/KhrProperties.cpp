//==----- KhrProperties.cpp --- sycl_khr_properties queue properties -------==//
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
using namespace sycl::khr;

struct OtherClass {};

static_assert(is_property_v<kp::enable_profiling> &&
              is_property_v<kp::in_order>);
static_assert(is_property_key_v<kp::key::enable_profiling> &&
              is_property_key_v<kp::key::in_order>);
static_assert(!is_property_key_compile_time_v<kp::key::enable_profiling> &&
              !is_property_key_compile_time_v<kp::key::in_order>);
static_assert(is_property_for_v<kp::enable_profiling, sycl::queue> &&
              is_property_for_v<kp::in_order, sycl::queue>);
static_assert(!is_property_for_v<kp::enable_profiling, OtherClass>);
// A property list is not itself a property.
static_assert(!is_property_v<properties<kp::enable_profiling>>);
static_assert(is_property_list_for_v<
              decltype(properties{kp::enable_profiling{true}, kp::in_order{}}),
              sycl::queue>);
static_assert(std::is_trivially_copyable_v<kp::enable_profiling> &&
              std::is_trivially_copyable_v<kp::in_order>);
// The khr constructors are explicit, like the property_list ones.
static_assert(!std::is_convertible_v<kp::in_order, sycl::queue>);

TEST(KhrQueueProperties, SingleProperty) {
  sycl::unittest::UrMock<> Mock;
  EXPECT_TRUE(sycl::queue{kp::in_order{}}.is_in_order());
  EXPECT_FALSE(sycl::queue{kp::in_order{false}}.is_in_order());
  EXPECT_TRUE(sycl::queue{kp::enable_profiling{}}.khr_is_profiling_enabled());
  EXPECT_FALSE(
      sycl::queue{kp::enable_profiling{false}}.khr_is_profiling_enabled());
}

TEST(KhrQueueProperties, PropertyList) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Q1{properties{kp::enable_profiling{}, kp::in_order{}}};
  EXPECT_TRUE(Q1.is_in_order());
  EXPECT_TRUE(Q1.khr_is_profiling_enabled());

  sycl::queue Q2{properties{kp::enable_profiling{false}, kp::in_order{true}}};
  EXPECT_TRUE(Q2.is_in_order());
  EXPECT_FALSE(Q2.khr_is_profiling_enabled());

  sycl::queue Q3{properties{}};
  EXPECT_FALSE(Q3.is_in_order());
  EXPECT_FALSE(Q3.khr_is_profiling_enabled());
}

TEST(KhrQueueProperties, Constructors) {
  sycl::unittest::UrMock<> Mock;
  sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::async_handler Handler = [](sycl::exception_list) {};
  auto Selector = sycl::default_selector_v;
  properties Props{kp::in_order{}};

  EXPECT_TRUE(sycl::queue(Props).is_in_order());
  EXPECT_TRUE(sycl::queue(Handler, Props).is_in_order());
  EXPECT_TRUE(sycl::queue(Selector, Props).is_in_order());
  EXPECT_TRUE(sycl::queue(Selector, Handler, Props).is_in_order());
  EXPECT_TRUE(sycl::queue(Dev, Props).is_in_order());
  EXPECT_TRUE(sycl::queue(Dev, Handler, Props).is_in_order());
  EXPECT_TRUE(sycl::queue(Ctx, Selector, Props).is_in_order());
  EXPECT_TRUE(sycl::queue(Ctx, Selector, Handler, Props).is_in_order());
  EXPECT_TRUE(sycl::queue(Ctx, Dev, Props).is_in_order());
  EXPECT_TRUE(sycl::queue(Ctx, Dev, Handler, Props).is_in_order());
}

TEST(KhrQueueProperties, OldStyleConstructors) {
  sycl::unittest::UrMock<> Mock;
  // Old-style construction must still resolve unambiguously.
  EXPECT_FALSE(sycl::queue{}.is_in_order());
  EXPECT_TRUE(
      sycl::queue{sycl::property_list{sycl::property::queue::in_order{}}}
          .is_in_order());
  EXPECT_TRUE(sycl::queue{sycl::property::queue::enable_profiling{}}
                  .khr_is_profiling_enabled());
}
