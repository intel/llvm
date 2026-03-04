//==- DeviceIndex.cpp -- sycl_ext_oneapi_platform_device_index unit tests --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

namespace {
const auto DEVICE1 = reinterpret_cast<ur_device_handle_t>(1u);
const auto DEVICE2 = reinterpret_cast<ur_device_handle_t>(2u);
const auto DEVICE3 = reinterpret_cast<ur_device_handle_t>(3u);

ur_result_t redefine_urDeviceGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 3;
  if (*params.pphDevices && *params.pNumEntries > 0) {
    (*params.pphDevices)[0] = DEVICE1;
    (*params.pphDevices)[1] = DEVICE2;
    (*params.pphDevices)[2] = DEVICE3;
  }
  return UR_RESULT_SUCCESS;
}

} // namespace

TEST(sycl_ext_oneapi_platform_device_index, CheckDeviceIndexes) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefine_urDeviceGet);

  sycl::platform plt = sycl::platform();
  auto devs = plt.get_devices();

  ASSERT_EQ(devs.size(), 3ull);

  for (size_t i = 0; i < devs.size(); i++)
    ASSERT_EQ(devs[i].ext_oneapi_index_within_platform(), i);
}
