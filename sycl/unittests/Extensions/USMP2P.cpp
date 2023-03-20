//==------------------------- USMP2P.cpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <helpers/PiMock.hpp>
#include <gtest/gtest.h>

TEST(USMP2PTest, USMP2PTest) {

  sycl::unittest::PiMock Mock1;
  sycl::platform Plt1 = Mock1.getPlatform();

  sycl::unittest::PiMock Mock2;
  sycl::platform Plt2 = Mock2.getPlatform();

  auto Dev1 = Plt1.get_devices()[0];
  auto Dev2 = Plt2.get_devices()[0];

  ASSERT_TRUE(Dev1.ext_oneapi_can_access_peer(
      Dev2, sycl::ext::oneapi::peer_access::access_supported));
  ASSERT_TRUE(Dev1.ext_oneapi_can_access_peer(
      Dev2, sycl::ext::oneapi::peer_access::atomics_supported));

  Dev1.ext_oneapi_enable_peer_access(Dev2);
  Dev1.ext_oneapi_disable_peer_access(Dev2);
}
