//==---- CurrentDevice.cpp -- sycl_ext_oneapi_current_device unit tests ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/oneapi/experimental/current_device.hpp>
#include <sycl/device.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>
#include <thread>

constexpr size_t NumberOfDevices = 2;

std::vector<ur_device_handle_t> GlobalDevicesHandle{
    mock::createDummyHandle<ur_device_handle_t>(),
    mock::createDummyHandle<ur_device_handle_t>(),
};

inline ur_result_t redefinedMockDevicesGet(void *pParams) {
  auto params = *reinterpret_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = NumberOfDevices;

  if (*params.pphDevices && *params.pNumEntries > 0) {
    for (size_t i = 0; i < NumberOfDevices; ++i)
      (*params.pphDevices)[i] = GlobalDevicesHandle[i];
  }

  return UR_RESULT_SUCCESS;
}

class CurrentDeviceTest : public ::testing::Test {
public:
  CurrentDeviceTest() : Mock{} {}

protected:
  sycl::unittest::UrMock<> Mock;

  void SetTwoDevices() {
    mock::getCallbacks().set_replace_callback("urDeviceGet",
                                              &redefinedMockDevicesGet);
  }

  template <ur_device_type_t UrDeviceType> inline void ChangeDeviceTypeTo() {
    mock::getCallbacks().set_replace_callback(
        "urDeviceGetInfo",
        &sycl::unittest::MockAdapter::mock_urDeviceGetInfo<UrDeviceType>);
  }
};

void callable_get_eq() {
  ASSERT_EQ(sycl::ext::oneapi::experimental::this_thread::get_current_device(),
            sycl::device{sycl::default_selector_v});
}

void callable_set_get_eq(sycl::device dev) {
  sycl::ext::oneapi::experimental::this_thread::set_current_device(dev);
  ASSERT_EQ(sycl::ext::oneapi::experimental::this_thread::get_current_device(),
            dev);
}

TEST_F(CurrentDeviceTest,
       CheckGetCurrentDeviceReturnDefaultDeviceInHostThread) {
  ASSERT_EQ(sycl::ext::oneapi::experimental::this_thread::get_current_device(),
            sycl::device{sycl::default_selector_v});
}

TEST_F(CurrentDeviceTest,
       CheckGetCurrentDeviceReturnDefaultSelectorByDefaultInTwoThreads) {
  SetTwoDevices();

  ChangeDeviceTypeTo<UR_DEVICE_TYPE_CPU>();
  sycl::device cpu_device;

  ChangeDeviceTypeTo<UR_DEVICE_TYPE_GPU>();
  sycl::device gpu_device;

  ASSERT_TRUE(cpu_device.is_cpu());
  ASSERT_TRUE(gpu_device.is_gpu());

  std::thread t1(callable_set_get_eq, cpu_device);
  std::thread t2(callable_set_get_eq, gpu_device);

  t1.join();
  t2.join();
}
