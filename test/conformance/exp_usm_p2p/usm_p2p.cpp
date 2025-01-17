// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"

using urP2PTest = uur::urAllDevicesTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urP2PTest);

TEST_P(urP2PTest, Success) {

  if (devices.size() < 2) {
    GTEST_SKIP();
  }

  size_t returned_size;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_EXTENSIONS, 0,
                                 nullptr, &returned_size));

  std::unique_ptr<char[]> returned_extensions(new char[returned_size]);

  ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_EXTENSIONS,
                                 returned_size, returned_extensions.get(),
                                 nullptr));

  std::string_view extensions_string(returned_extensions.get());
  const bool usm_p2p_support =
      extensions_string.find(UR_USM_P2P_EXTENSION_STRING_EXP) !=
      std::string::npos;

  if (!usm_p2p_support) {
    GTEST_SKIP() << "EXP usm p2p feature is not supported.";
  }

  int value;
  ASSERT_SUCCESS(urUsmP2PPeerAccessGetInfoExp(
      devices[0], ///< [in] handle of the command device object
      devices[1], ///< [in] handle of the peer device object
      UR_EXP_PEER_INFO_UR_PEER_ACCESS_SUPPORTED, sizeof(int), &value,
      &returned_size));
  // Note that whilst it is not currently specified to be a requirement in the
  // specification, currently all supported backends return value = 1 for the
  // UR_EXP_PEER_INFO_UR_PEER_ACCESS_SUPPORTED query when the query is true
  // (matching the native query return values). Generally different backends can
  // return different values for a given device query; however it is
  // advisable that for boolean queries they return the same values to indicate
  // true/false. When this extension is moved out of experimental status such
  // boolean return values should be specified by the extension.
  ASSERT_EQ(value, 1);

  // Just check that this doesn't throw since supporting peer atomics is
  // optional and can depend on backend/device.
  ASSERT_SUCCESS(urUsmP2PPeerAccessGetInfoExp(
      devices[0], devices[1], UR_EXP_PEER_INFO_UR_PEER_ATOMICS_SUPPORTED,
      sizeof(int), &value, &returned_size));

  ASSERT_SUCCESS(urUsmP2PEnablePeerAccessExp(devices[0], devices[1]));
  ASSERT_SUCCESS(urUsmP2PDisablePeerAccessExp(devices[0], devices[1]));
}
