//==------------------------- USMP2P.cpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

int check = 0;

ur_result_t redefinedDeviceGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 2;
  if (*params.pphDevices && *params.pNumEntries > 0) {
    (*params.pphDevices)[0] = reinterpret_cast<ur_device_handle_t>(1);
    (*params.pphDevices)[1] = reinterpret_cast<ur_device_handle_t>(2);
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEnablePeerAccess(void *) {
  check = 3;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedDisablePeerAccess(void *) {
  check = 4;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedPeerAccessGetInfo(void *pParams) {
  auto params =
      *static_cast<ur_usm_p2p_peer_access_get_info_exp_params_t *>(pParams);

  if (*params.ppPropValue)
    *static_cast<int32_t *>(*params.ppPropValue) = 1;
  if (*params.ppPropSizeRet)
    **params.ppPropSizeRet = sizeof(int32_t);

  if (*params.ppropName == UR_EXP_PEER_INFO_UR_PEER_ACCESS_SUPPORTED) {
    check = 1;
  } else if (*params.ppropName == UR_EXP_PEER_INFO_UR_PEER_ATOMICS_SUPPORTED) {
    check = 2;
  }
  return UR_RESULT_SUCCESS;
}

TEST(USMP2PTest, USMP2PTest) {

  sycl::unittest::UrMock<> Mock;

  mock::getCallbacks().set_replace_callback("urDeviceGet", &redefinedDeviceGet);
  mock::getCallbacks().set_replace_callback("urUsmP2PEnablePeerAccessExp",
                                            &redefinedEnablePeerAccess);
  mock::getCallbacks().set_replace_callback("urUsmP2PDisablePeerAccessExp",
                                            &redefinedDisablePeerAccess);
  mock::getCallbacks().set_replace_callback("urUsmP2PPeerAccessGetInfoExp",
                                            &redefinedPeerAccessGetInfo);

  sycl::platform Plt = sycl::platform();

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
