//==------------------- NoDeviceIPVersion.cpp ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sycl/ext/oneapi/experimental/device_architecture.hpp"
#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

static ur_result_t afterDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_IP_VERSION) {
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  return UR_RESULT_SUCCESS;
}

namespace syclex = sycl::ext::oneapi::experimental;
TEST(NoDeviceIPVersionTest, NoDeviceIPVersion) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &afterDeviceGetInfo);
  sycl::platform Plt = sycl::platform();
  auto Dev = Plt.get_devices()[0];
  if (Dev.get_backend() != sycl::backend::opencl &&
      Dev.get_backend() != sycl::backend::ext_oneapi_level_zero) {
    GTEST_SKIP();
  }

  syclex::architecture DevArch =
      Dev.get_info<syclex::info::device::architecture>();
  ASSERT_TRUE(DevArch == syclex::architecture::unknown ||
              DevArch == syclex::architecture::x86_64);
}
