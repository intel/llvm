//==---------- support_native.cpp --- Check support is correctly reported --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_mock_helpers.hpp"

#include <sycl/sycl.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

template <bool Support>
static ur_result_t redefinedDeviceGetInfoAfter(void *pParams) {
  auto &Params = *reinterpret_cast<ur_device_get_info_params_t *>(pParams);
  if (*Params.ppropName == UR_DEVICE_INFO_USE_NATIVE_ASSERT) {
    if (*Params.ppPropValue)
      *reinterpret_cast<ur_bool_t *>(*Params.ppPropValue) = Support;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(ur_bool_t);
  }
  return UR_RESULT_SUCCESS;
}

TEST(SupportNativeAssert, True) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter<true>);

  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];

  ASSERT_TRUE(Dev.has(sycl::aspect::ext_oneapi_native_assert));
}

TEST(SupportNativeAssert, False) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter<false>);

  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];

  ASSERT_FALSE(Dev.has(sycl::aspect::ext_oneapi_native_assert));
}
