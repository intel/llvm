// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"

using urP2PTest = uur::urAllDevicesTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urP2PTest);

TEST_P(urP2PTest, Success) {

  if (devices.size() < 2) {
    GTEST_SKIP();
  }

  ur_bool_t usm_p2p_support = false;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_USM_P2P_SUPPORT_EXP,
                                 sizeof(usm_p2p_support), &usm_p2p_support,
                                 nullptr));
  if (!usm_p2p_support) {
    GTEST_SKIP() << "EXP usm p2p feature is not supported.";
  }

  size_t returned_size = 0;
  int value;
  ASSERT_SUCCESS(urUsmP2PPeerAccessGetInfoExp(
      /// [in] handle of the command device object
      devices[0],
      /// [in] handle of the peer device object
      devices[1], UR_EXP_PEER_INFO_UR_PEER_ACCESS_SUPPORT, sizeof(int), &value,
      &returned_size));
  // Note that whilst it is not currently specified to be a requirement in the
  // specification, currently all supported backends return value = 1 for the
  // UR_EXP_PEER_INFO_UR_PEER_ACCESS_SUPPORT query when the query is true
  // (matching the native query return values). Generally different backends can
  // return different values for a given device query; however it is
  // advisable that for boolean queries they return the same values to indicate
  // true/false. When this extension is moved out of experimental status such
  // boolean return values should be specified by the extension.
  ASSERT_EQ(value, 1);

  // Just check that this doesn't throw since supporting peer atomics is
  // optional and can depend on backend/device.
  ASSERT_SUCCESS(urUsmP2PPeerAccessGetInfoExp(
      devices[0], devices[1], UR_EXP_PEER_INFO_UR_PEER_ATOMICS_SUPPORT,
      sizeof(int), &value, &returned_size));

  ASSERT_SUCCESS(urUsmP2PEnablePeerAccessExp(devices[0], devices[1]));
  ASSERT_SUCCESS(urUsmP2PDisablePeerAccessExp(devices[0], devices[1]));
}
