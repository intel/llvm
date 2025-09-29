//==----- DeviceRefCounter - Kernel build options processing unit test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <detail/global_handler.hpp>
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
    EXPECT_EQ(DevRefCounter, 0);

    mock::getCallbacks().set_after_callback("urDeviceGet",
                                            &redefinedDevicesGetAfter);
    mock::getCallbacks().set_after_callback("urDeviceRetain",
                                            &redefinedDeviceRetainAfter);
    mock::getCallbacks().set_after_callback("urDeviceRelease",
                                            &redefinedDeviceReleaseAfter);
    sycl::platform Plt = sycl::platform();

    Plt.get_devices();
    EXPECT_NE(DevRefCounter, 0);
    // This is the behavior that SYCL performs at shutdown, but there
    // are timing differences Lin/Win and shared/static that make
    // it not map correctly into our mock.
    // So for this test, we just do it.
    sycl::detail::GlobalHandler::instance().getPlatformCache().clear();
  }
  EXPECT_EQ(DevRefCounter, 0);
}
