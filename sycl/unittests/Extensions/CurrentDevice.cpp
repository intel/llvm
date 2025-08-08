//==---- CurrentDevice.cpp -- sycl_ext_oneapi_current_device unit tests ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>
#include <thread>

namespace {
const auto DEVICE_CPU = reinterpret_cast<ur_device_handle_t>(1u);
const auto DEVICE_GPU = reinterpret_cast<ur_device_handle_t>(2u);

ur_result_t redefine_urDeviceGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 2;
  if (*params.pphDevices && *params.pNumEntries > 0) {
    (*params.pphDevices)[0] = DEVICE_CPU;
    (*params.pphDevices)[1] = DEVICE_GPU;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_TYPE: {
    ur_device_type_t UrDeviceType = UR_DEVICE_TYPE_DEFAULT;
    if (*params.phDevice == DEVICE_CPU) {
      UrDeviceType = UR_DEVICE_TYPE_CPU;
    } else if (*params.phDevice == DEVICE_GPU) {
      UrDeviceType = UR_DEVICE_TYPE_GPU;
    }
    if (*params.ppPropValue)
      *static_cast<ur_device_type_t *>(*params.ppPropValue) = UrDeviceType;
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(UrDeviceType);
    return UR_RESULT_SUCCESS;
  }
  default:
    return UR_RESULT_SUCCESS;
  }
}

void callable_set_get_eq(sycl::device dev) {
  sycl::ext::oneapi::experimental::this_thread::set_current_device(dev);
  ASSERT_NO_FATAL_FAILURE(
      sycl::ext::oneapi::experimental::this_thread::get_current_device() = dev);
}
} // namespace

TEST(CurrentDeviceTest, CheckGetCurrentDeviceReturnDefaultDeviceInHostThread) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefine_urDeviceGet);
  ASSERT_EQ(sycl::ext::oneapi::experimental::this_thread::get_current_device(),
            sycl::device{sycl::default_selector_v});
}

TEST(CurrentDeviceTest,
     CheckGetCurrentDeviceReturnDefaultSelectorByDefaultInTwoThreads) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefine_urDeviceGet);
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo);

  sycl::platform Plt = sycl::platform();

  ASSERT_EQ(Plt.get_devices().size(), 2ull);

  sycl::device cpu_device = Plt.get_devices()[0];
  sycl::device gpu_device = Plt.get_devices()[1];

  ASSERT_TRUE(cpu_device.is_cpu());
  ASSERT_TRUE(gpu_device.is_gpu());

  std::thread t1(callable_set_get_eq, cpu_device);
  std::thread t2(callable_set_get_eq, gpu_device);

  t1.join();
  t2.join();
}
