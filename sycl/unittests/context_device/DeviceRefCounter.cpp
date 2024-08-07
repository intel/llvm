//==----- DeviceRefCounter - Kernel build options processing unit test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

int DevRefCounter = 0;

static ur_result_t redefinedDevicesGetAfter(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.pphDevices)
    DevRefCounter += *params.pNumEntries;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedDeviceRetainAfter(void *) {
  DevRefCounter++;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedDeviceReleaseAfter(void *) {
  DevRefCounter--;
  return UR_RESULT_SUCCESS;
}

TEST(DevRefCounter, DevRefCounter) {
  {
    sycl::unittest::UrMock<> Mock;
    sycl::platform Plt = sycl::platform();

    mock::getCallbacks().set_after_callback("urDeviceGet",
                                            &redefinedDevicesGetAfter);
    mock::getCallbacks().set_after_callback("urDeviceRetain",
                                            &redefinedDeviceRetainAfter);
    mock::getCallbacks().set_after_callback("urDeviceRelease",
                                            &redefinedDeviceReleaseAfter);

    Plt.get_devices();
  }
  EXPECT_EQ(DevRefCounter, 0);
}
