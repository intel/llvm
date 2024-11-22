//==----- DeviceGetInfoAspects.cpp --- info::device::aspects unit test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <algorithm>

using namespace sycl;

static bool containsAspect(const std::vector<sycl::aspect> &DeviceAspects,
                           const aspect Aspect) {
  return std::find(DeviceAspects.begin(), DeviceAspects.end(), Aspect) !=
         DeviceAspects.end();
}

TEST(DeviceGetInfo, SupportedDeviceAspects) {
  sycl::unittest::UrMock<> Mock;

  sycl::platform Plt = sycl::platform();
  sycl::device Dev = Plt.get_devices()[0];

  std::vector<sycl::aspect> DeviceAspects =
      Dev.get_info<info::device::aspects>();

  // Tests to examine aspects of default mock device, as defined in
  // helpers/UrMockAdapter.hpp so these tests all need to be kept in sync with
  // changes to that file.
  EXPECT_TRUE(containsAspect(DeviceAspects, aspect::gpu));
  EXPECT_TRUE(containsAspect(DeviceAspects, aspect::fp16));
  EXPECT_TRUE(containsAspect(DeviceAspects, aspect::fp64));

  EXPECT_FALSE(containsAspect(DeviceAspects, aspect::host));
  EXPECT_FALSE(containsAspect(DeviceAspects, aspect::cpu));
}
