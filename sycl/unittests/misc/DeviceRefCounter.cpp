//==----- DeviceRefCounter - Kernel build options processing unit test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <detail/global_handler.hpp>

#include <gtest/gtest.h>

#include <unordered_map>
#include <atomic>

#include <sycl/sycl.hpp>

std::unordered_map<void *, std::atomic<int>> DevRefCounter;

static pi_result redefinedDevicesGet(pi_platform platform,
                                     pi_device_type device_type,
                                     pi_uint32 num_entries, pi_device *devices,
                                     pi_uint32 *num_devices) {
  if (devices && num_entries > 0) {
    *devices = (pi_device) new int;
    DevRefCounter[*devices] = 1;
  }
  if (num_devices)
    *num_devices = 1;
  return PI_SUCCESS;
}

static pi_result redefinedDeviceRetain(pi_device device) {
  DevRefCounter[device]++;
  return PI_SUCCESS;
}

static pi_result redefinedDeviceGetInfo(pi_device device,
                                        pi_device_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

static pi_result redefinedDeviceRelease(pi_device device) {
  int NewVal = --DevRefCounter[device];
  if (NewVal == 0)
    delete reinterpret_cast<int *>(device);
  return PI_SUCCESS;
}

TEST(DevRefCounter, DevRefCounter) {
  {
    sycl::platform Plt{sycl::default_selector()};
    if (Plt.is_host()) {
      std::cerr << "Test is not supported on host, skipping\n";
      return; // test is not supported on host.
    }

    sycl::unittest::PiMock Mock{Plt};
    setupDefaultMockAPIs(Mock);

    Mock.redefine<sycl::detail::PiApiKind::piDevicesGet>(redefinedDevicesGet);
    Mock.redefine<sycl::detail::PiApiKind::piDeviceRetain>(
        redefinedDeviceRetain);
    Mock.redefine<sycl::detail::PiApiKind::piDeviceRelease>(
        redefinedDeviceRelease);

    Mock.redefine<sycl::detail::PiApiKind::piDeviceGetInfo>(
        redefinedDeviceGetInfo);

    Plt.get_devices();
  }
  for (auto &El : DevRefCounter)
    EXPECT_EQ(El.second, 0);
}
