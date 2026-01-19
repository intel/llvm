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

ur_result_t redefinedDevicePartitionAfter(void *pParams) {
  auto params = *static_cast<ur_device_partition_params_t *>(pParams);
  if (*params.pphSubDevices) {
    for (size_t I = 0; I < *params.pNumDevices; ++I) {
      *params.pphSubDevices[I] = reinterpret_cast<ur_device_handle_t>(1000 + I);
    }
  }
  if (*params.ppNumDevicesRet)
    **params.ppNumDevicesRet = *params.pNumDevices;

  DevRefCounter += *params.pNumDevices;
  return UR_RESULT_SUCCESS;
}

static constexpr size_t NumSubDevices = 2;

ur_result_t redefinedDeviceGetInfoAfter(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_SUPPORTED_PARTITIONS) {
    if (*params.ppPropValue) {
      auto *Result =
          reinterpret_cast<ur_device_partition_t *>(*params.ppPropValue);
      *Result = UR_DEVICE_PARTITION_EQUALLY;
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_device_partition_t);
  } else if (*params.ppropName == UR_DEVICE_INFO_MAX_COMPUTE_UNITS) {
    auto *Result = reinterpret_cast<uint32_t *>(*params.ppPropValue);
    *Result = NumSubDevices;
  }
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

TEST(SubDevRefCounter, SubDevRefCounter) {
  {
    DevRefCounter = 0;
    sycl::unittest::UrMock<> Mock;
    mock::getCallbacks().set_after_callback("urDeviceGet",
                                            &redefinedDevicesGetAfter);
    mock::getCallbacks().set_after_callback("urDeviceRetain",
                                            &redefinedDeviceRetainAfter);
    mock::getCallbacks().set_after_callback("urDeviceRelease",
                                            &redefinedDeviceReleaseAfter);
    mock::getCallbacks().set_before_callback("urDevicePartition",
                                             &redefinedDevicePartitionAfter);
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &redefinedDeviceGetInfoAfter);
    sycl::platform Plt = sycl::platform();

    auto Devs = Plt.get_devices();
    if (!Devs.empty()) {
      auto Subdevs = Devs[0]
                         .create_sub_devices<
                             sycl::info::partition_property::partition_equally>(
                             NumSubDevices);
    }
    EXPECT_NE(DevRefCounter, 0);
    sycl::detail::GlobalHandler::instance().getPlatformCache().clear();
  }
  EXPECT_EQ(DevRefCounter, 0);
}
