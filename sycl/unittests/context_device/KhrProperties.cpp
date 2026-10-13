//==---- KhrProperties.cpp --- sycl_khr_properties context constructors ----==//
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
#include <vector>

using namespace sycl::khr;

static_assert(is_property_list_for_v<empty_properties_t, sycl::context>);
static_assert(std::is_constructible_v<sycl::context, empty_properties_t>);
static_assert(std::is_constructible_v<sycl::context, sycl::device,
                                      sycl::async_handler, properties<>>);
// The khr constructors are explicit, like the property_list ones.
static_assert(!std::is_convertible_v<empty_properties_t, sycl::context>);

TEST(KhrContextProperties, Constructors) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  sycl::device Dev = Plt.get_devices()[0];
  std::vector<sycl::device> Devs{Dev};
  sycl::async_handler Handler = [](sycl::exception_list) {};
  properties Props{};

  sycl::context C1(Props);
  sycl::context C2(Handler, empty_properties_t{});
  sycl::context C3(Dev, Props);
  sycl::context C4(Dev, Handler, Props);
  sycl::context C5(Plt, Props);
  sycl::context C6(Plt, Handler, Props);
  sycl::context C7(Devs, Props);
  sycl::context C8(Devs, Handler, Props);

  for (const sycl::context &C : {C1, C2, C3, C4, C5, C6, C7, C8}) {
    EXPECT_EQ(C.get_platform(), Plt);
    EXPECT_FALSE(C.get_devices().empty());
  }
  EXPECT_EQ(C3.get_devices(), Devs);
  EXPECT_EQ(C4.get_devices(), Devs);
  EXPECT_EQ(C7.get_devices(), Devs);
  EXPECT_EQ(C8.get_devices(), Devs);
}

TEST(KhrContextProperties, OldStyleConstructors) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  sycl::device Dev = Plt.get_devices()[0];
  sycl::async_handler Handler = [](sycl::exception_list) {};

  // Old-style construction must still resolve unambiguously.
  sycl::context C1{};
  sycl::context C2(sycl::property_list{});
  sycl::context C3(Handler);
  sycl::context C4(Dev);
  sycl::context C5(Dev, sycl::property_list{});
  sycl::context C6(Plt);
  sycl::context C7(std::vector<sycl::device>{Dev}, Handler);
  EXPECT_EQ(C4.get_platform(), Plt);
  EXPECT_EQ(C5.get_devices()[0], Dev);
}
