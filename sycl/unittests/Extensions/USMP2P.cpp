//==------------------------- USMP2P.cpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <sycl/sycl.hpp>

int check = 0;

pi_result redefinedDevicesGet(pi_platform platform, pi_device_type device_type,
                              pi_uint32 num_entries, pi_device *devices,
                              pi_uint32 *num_devices) {
  if (num_devices)
    *num_devices = 2;
  if (devices && num_entries > 0) {
    devices[0] = reinterpret_cast<pi_device>(1);
    devices[1] = reinterpret_cast<pi_device>(2);
  }
  return PI_SUCCESS;
}

pi_result redefinedEnablePeerAccess(pi_device command_device,
                                    pi_device peer_device) {
  check = 3;
  return PI_SUCCESS;
}

pi_result redefinedDisablePeerAccess(pi_device command_device,
                                     pi_device peer_device) {
  check = 4;
  return PI_SUCCESS;
}

pi_result redefinedPeerAccessGetInfo(pi_device command_device,
                                     pi_device peer_device, pi_peer_attr attr,
                                     size_t param_value_size, void *param_value,
                                     size_t *param_value_size_ret) {

  if (param_value)
    *static_cast<pi_int32 *>(param_value) = 1;
  if (param_value_size_ret)
    *param_value_size_ret = sizeof(pi_int32);

  if (attr == PI_PEER_ACCESS_SUPPORTED) {
    check = 1;
  } else if (attr == PI_PEER_ATOMICS_SUPPORTED) {
    check = 2;
  }
  return PI_SUCCESS;
}

TEST(USMP2PTest, USMP2PTest) {

  sycl::unittest::PiMock Mock;

  Mock.redefine<sycl::detail::PiApiKind::piDevicesGet>(redefinedDevicesGet);
  Mock.redefine<sycl::detail::PiApiKind::piextEnablePeerAccess>(
      redefinedEnablePeerAccess);
  Mock.redefine<sycl::detail::PiApiKind::piextDisablePeerAccess>(
      redefinedDisablePeerAccess);
  Mock.redefine<sycl::detail::PiApiKind::piextPeerAccessGetInfo>(
      redefinedPeerAccessGetInfo);

  sycl::platform Plt = Mock.getPlatform();

  auto Dev1 = Plt.get_devices()[0];
  auto Dev2 = Plt.get_devices()[1];

  ASSERT_TRUE(Dev1.ext_oneapi_can_access_peer(
      Dev2, sycl::ext::oneapi::peer_access::access_supported));
  ASSERT_EQ(check, 1);
  ASSERT_TRUE(Dev1.ext_oneapi_can_access_peer(
      Dev2, sycl::ext::oneapi::peer_access::atomics_supported));
  ASSERT_EQ(check, 2);

  Dev1.ext_oneapi_enable_peer_access(Dev2);
  ASSERT_EQ(check, 3);
  Dev1.ext_oneapi_disable_peer_access(Dev2);
  ASSERT_EQ(check, 4);
}
