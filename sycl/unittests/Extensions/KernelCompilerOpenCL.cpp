//==------------------- KernelCompilerOpenCL.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

static uint32_t MockIPVersion = 0;

static ur_result_t afterDeviceGetInfo(void *pParams) {
  auto Params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*Params.ppropName == UR_DEVICE_INFO_IP_VERSION && *Params.ppPropValue)
    *static_cast<uint32_t *>(*Params.ppPropValue) = MockIPVersion;
  return UR_RESULT_SUCCESS;
}

// ocloc query succeeds even for a device it cannot target, so check that
// such a device is still reported as unable to build OpenCL C sources.
TEST(KernelCompilerOpenCL, CanBuildUnknownDevice) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &afterDeviceGetInfo);
  sycl::device Dev = sycl::platform().get_devices()[0];

  // DG2 (12.55.8), supported by ocloc.
  MockIPVersion = 0x030dc008;
  if (!Dev.ext_oneapi_can_build(syclex::source_language::opencl))
    GTEST_SKIP() << "ocloc is not available";

  // Not an IP version of any Intel GPU (31.7.3).
  MockIPVersion = 0x07c1c003;
  EXPECT_FALSE(Dev.ext_oneapi_can_build(syclex::source_language::opencl));
  EXPECT_EQ(Dev.ext_oneapi_cl_profile(), "");
}
