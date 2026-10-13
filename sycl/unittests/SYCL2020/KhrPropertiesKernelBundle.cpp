//==- KhrPropertiesKernelBundle.cpp - sycl_khr_properties kernel bundles ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <gtest/gtest.h>
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <type_traits>
#include <vector>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

class KhrKernelBundleKernel;
MOCK_INTEGRATION_HEADER(KhrKernelBundleKernel)

static sycl::unittest::MockDeviceImage Img =
    sycl::unittest::generateDefaultImage({"KhrKernelBundleKernel"});
static sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};

using InputBundle = sycl::kernel_bundle<sycl::bundle_state::input>;
using ObjectBundle = sycl::kernel_bundle<sycl::bundle_state::object>;
using ExecBundle = sycl::kernel_bundle<sycl::bundle_state::executable>;

static_assert(is_property_list_for_v<empty_properties_t, kp::tag::compile>);
static_assert(is_property_list_for_v<empty_properties_t, kp::tag::link>);
static_assert(is_property_list_for_v<empty_properties_t, kp::tag::build>);
static_assert(std::is_same_v<decltype(sycl::compile(std::declval<InputBundle>(),
                                                    properties{})),
                             ObjectBundle>);
static_assert(std::is_same_v<decltype(sycl::link(std::declval<ObjectBundle>(),
                                                 empty_properties_t{})),
                             ExecBundle>);
static_assert(std::is_same_v<decltype(sycl::build(std::declval<InputBundle>(),
                                                  properties{})),
                             ExecBundle>);

static InputBundle getInputBundle(const sycl::context &Ctx,
                                  const sycl::device &Dev) {
  return sycl::get_kernel_bundle<KhrKernelBundleKernel,
                                 sycl::bundle_state::input>(Ctx, {Dev});
}

TEST(KhrKernelBundleProperties, Compile) {
  sycl::unittest::UrMock<> Mock;
  sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};
  std::vector<sycl::device> Devs{Dev};
  InputBundle Input = getInputBundle(Ctx, Dev);

  EXPECT_FALSE(sycl::compile(Input, properties{}).empty());
  EXPECT_FALSE(sycl::compile(Input, Devs, empty_properties_t{}).empty());
}

TEST(KhrKernelBundleProperties, Link) {
  sycl::unittest::UrMock<> Mock;
  sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};
  std::vector<sycl::device> Devs{Dev};
  ObjectBundle Obj = sycl::compile(getInputBundle(Ctx, Dev));
  std::vector<ObjectBundle> Objs{Obj};

  EXPECT_FALSE(sycl::link(Objs, Devs, properties{}).empty());
  EXPECT_FALSE(sycl::link(Obj, Devs, properties{}).empty());
  EXPECT_FALSE(sycl::link(Objs, properties{}).empty());
  EXPECT_FALSE(sycl::link(Obj, properties{}).empty());
}

TEST(KhrKernelBundleProperties, Build) {
  sycl::unittest::UrMock<> Mock;
  sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};
  std::vector<sycl::device> Devs{Dev};
  InputBundle Input = getInputBundle(Ctx, Dev);

  EXPECT_FALSE(sycl::build(Input, properties{}).empty());
  EXPECT_FALSE(sycl::build(Input, Devs, empty_properties_t{}).empty());
}

TEST(KhrKernelBundleProperties, OldStyleOverloads) {
  sycl::unittest::UrMock<> Mock;
  sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};
  std::vector<sycl::device> Devs{Dev};
  InputBundle Input = getInputBundle(Ctx, Dev);

  // Old-style calls must still resolve unambiguously.
  ObjectBundle Obj = sycl::compile(Input);
  EXPECT_FALSE(Obj.empty());
  EXPECT_FALSE(sycl::compile(Input, Devs, sycl::property_list{}).empty());
  EXPECT_FALSE(sycl::link(Obj).empty());
  EXPECT_FALSE(sycl::link(Obj, Devs).empty());
  EXPECT_FALSE(sycl::build(Input).empty());
  EXPECT_FALSE(sycl::build(Input, sycl::property_list{}).empty());
}

TEST(KhrKernelBundleProperties, UnqualifiedCall) {
  using namespace sycl;
  unittest::UrMock<> Mock;
  device Dev = platform().get_devices()[0];
  context Ctx{Dev};
  // The kernel bundle tags must not shadow the free functions.
  EXPECT_FALSE(build(getInputBundle(Ctx, Dev), properties{}).empty());
}
