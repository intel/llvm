//==---- HasExtension.cpp --- Spec constants default values unit test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

using namespace sycl;

TEST(HasExtensionID, HasExtensionCallsCorrectPluginMethods) {
  sycl::unittest::PiMock Mock;

  sycl::platform Plt = Mock.getPlatform();
  sycl::device Dev = Plt.get_devices()[0];

  bool PlatformHasExtension = opencl::has_extension(Plt, "cl_khr_subgroups");
  EXPECT_TRUE(PlatformHasExtension);

  bool DeviceHasExtension = opencl::has_extension(Dev, "cl_khr_fp64");
  EXPECT_TRUE(DeviceHasExtension);
}
