//==----- DeviceRefCounter - Kernel build options processing unit test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <helpers/PiMock.hpp>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

int DevRefCounter = 0;

static pi_result redefinedDevicesGetAfter(pi_platform platform,
                                          pi_device_type device_type,
                                          pi_uint32 num_entries,
                                          pi_device *devices,
                                          pi_uint32 *num_devices) {
  if (devices)
    DevRefCounter += num_entries;
  return PI_SUCCESS;
}

static pi_result redefinedDeviceRetainAfter(pi_device device) {
  DevRefCounter++;
  return PI_SUCCESS;
}

static pi_result redefinedDeviceReleaseAfter(pi_device device) {
  DevRefCounter--;
  return PI_SUCCESS;
}

TEST(DevRefCounter, DevRefCounter) {
  {
    sycl::unittest::PiMock Mock;
    sycl::platform Plt = Mock.getPlatform();

    Mock.redefineAfter<sycl::detail::PiApiKind::piDevicesGet>(
        redefinedDevicesGetAfter);
    Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceRetain>(
        redefinedDeviceRetainAfter);
    Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceRelease>(
        redefinedDeviceReleaseAfter);

    Plt.get_devices();
  }
  EXPECT_EQ(DevRefCounter, 0);
}
