//==---- HasExtension.cpp --- Spec constants default values unit test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

using namespace sycl;

TEST(HasExtensionID, HasExtensionCallsCorrectAdapterMethods) {
  sycl::unittest::UrMock<> Mock;

  sycl::platform Plt = sycl::platform();
  sycl::device Dev = Plt.get_devices()[0];

  bool PlatformHasSubgroups = opencl::has_extension(Plt, "cl_khr_subgroups");
  EXPECT_TRUE(PlatformHasSubgroups);

  bool DeviceHasFP64 = opencl::has_extension(Dev, "cl_khr_fp64");
  EXPECT_TRUE(DeviceHasFP64);

  bool PlatformNotHasErroneousExtension =
      opencl::has_extension(Plt, "test_for_unknown_platform_extension");
  EXPECT_FALSE(PlatformNotHasErroneousExtension);

  bool DeviceNotHasErroneousExtension =
      opencl::has_extension(Dev, "test_for_unknown_device_extension");
  EXPECT_FALSE(DeviceNotHasErroneousExtension);
}
