//==--- DeviceIsIntegratedGPU.cpp - oneapi_device_is_integrated_gpu test ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sycl/platform.hpp"
#include <detail/device_impl.hpp>
#include <sycl/sycl.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

namespace {
template <bool IsIntegratedGPU, ur_device_type_t URDeviceType>
static ur_result_t redefinedDeviceGetInfoAfter(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_IS_INTEGRATED_GPU) {
    auto *Result = reinterpret_cast<ur_bool_t *>(*params.ppPropValue);
    *Result = IsIntegratedGPU;
  }

  if (*params.ppropName == UR_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<ur_device_type_t *>(*params.ppPropValue);
    *Result = URDeviceType;
  }

  return UR_RESULT_SUCCESS;
}
} // namespace

TEST(DeviceIsIntegratedGPU, DeviceIsNotIntegratedGPUOnGPUDevice) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback(
      "urDeviceGetInfo", &redefinedDeviceGetInfoAfter</*IsIntegratedGPU=*/false,
                                                      UR_DEVICE_TYPE_GPU>);
  sycl::device Device = sycl::platform().get_devices()[0];
  ASSERT_FALSE(Device.has(sycl::aspect::ext_oneapi_is_integrated_gpu));
}

TEST(DeviceIsIntegratedGPU, DeviceIsIntegratedGPUOnGPUDevice) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback(
      "urDeviceGetInfo", &redefinedDeviceGetInfoAfter</*IsIntegratedGPU=*/true,
                                                      UR_DEVICE_TYPE_GPU>);
  sycl::device Device = sycl::platform().get_devices()[0];
  ASSERT_TRUE(Device.has(sycl::aspect::ext_oneapi_is_integrated_gpu));
}

TEST(DeviceIsIntegratedGPU, DeviceIsNotIntegratedGPUOnCPUDevice) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback(
      "urDeviceGetInfo", &redefinedDeviceGetInfoAfter</*IsIntegratedGPU=*/false,
                                                      UR_DEVICE_TYPE_CPU>);
  sycl::device Device = sycl::platform().get_devices()[0];
  ASSERT_FALSE(Device.has(sycl::aspect::ext_oneapi_is_integrated_gpu));
}

TEST(DeviceIsIntegratedGPU, DeviceIsIntegratedGPUOnCPUDevice) {
  sycl::unittest::UrMock<> Mock;
  // Not much sense here but if for some reason UR_DEVICE_INFO_IS_INTEGRATED_GPU
  // is true on CPU device, we check that
  // sycl::aspect::ext_oneapi_is_integrated_gpu must be false as stated in the
  // extension spec.
  mock::getCallbacks().set_after_callback(
      "urDeviceGetInfo", &redefinedDeviceGetInfoAfter</*IsIntegratedGPU=*/true,
                                                      UR_DEVICE_TYPE_CPU>);
  sycl::device Device = sycl::platform().get_devices()[0];
  ASSERT_FALSE(Device.has(sycl::aspect::ext_oneapi_is_integrated_gpu));
}
