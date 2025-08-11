//==----- DeviceRefCounter - Kernel build options processing unit test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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

#ifndef WIN32
// This test passes because the UrMock emulates teardown on Linux, but
// on Windows there is a difference so this test is skipped.
TEST(DevRefCounter, DevRefCounter) {
  {
    sycl::unittest::UrMock<> Mock;

    mock::getCallbacks().set_after_callback("urDeviceGet",
                                            &redefinedDevicesGetAfter);
    mock::getCallbacks().set_after_callback("urDeviceRetain",
                                            &redefinedDeviceRetainAfter);
    mock::getCallbacks().set_after_callback("urDeviceRelease",
                                            &redefinedDeviceReleaseAfter);
    sycl::platform Plt = sycl::platform();

    Plt.get_devices();
  } // <- ~UrMock destructor called here.
  EXPECT_EQ(DevRefCounter, 0);
}
#endif // !WIN32
