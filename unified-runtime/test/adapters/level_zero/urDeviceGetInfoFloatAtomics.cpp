// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %maybe-v1 ./float_atomics_ext-test
// RUN: %maybe-v2 ./float_atomics_ext-test

#include "ze_api.h"
#include <uur/fixtures.h>

#include <cstring>
#include <string>
#include <vector>

using urLevelZeroFloatAtomicsExtTest = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urLevelZeroFloatAtomicsExtTest);

TEST_P(urLevelZeroFloatAtomicsExtTest, ExtensionMatchesFp16FlagsOnCRI) {
  // Query the extension string from UR.
  size_t size = 0;
  ASSERT_SUCCESS(
      urDeviceGetInfo(device, UR_DEVICE_INFO_EXTENSIONS, 0, nullptr, &size));
  std::vector<char> extensions(size);
  ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_EXTENSIONS, size,
                                 extensions.data(), nullptr));
  const bool urReports =
      std::string(extensions.data()).find("cl_ext_float_atomics") !=
      std::string::npos;

  // Query the same information natively.
  ur_native_handle_t nativeDevice = 0;
  ASSERT_SUCCESS(urDeviceGetNativeHandle(device, &nativeDevice));
  auto zeDevice = reinterpret_cast<ze_device_handle_t>(nativeDevice);

  ur_platform_handle_t platform = nullptr;
  ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_PLATFORM,
                                 sizeof(platform), &platform, nullptr));
  ur_native_handle_t nativePlatform = 0;
  ASSERT_SUCCESS(urPlatformGetNativeHandle(platform, &nativePlatform));
  auto zeDriver = reinterpret_cast<ze_driver_handle_t>(nativePlatform);

  uint32_t count = 0;
  ASSERT_EQ(zeDriverGetExtensionProperties(zeDriver, &count, nullptr),
            ZE_RESULT_SUCCESS);
  std::vector<ze_driver_extension_properties_t> driverExts(count);
  ASSERT_EQ(zeDriverGetExtensionProperties(zeDriver, &count, driverExts.data()),
            ZE_RESULT_SUCCESS);
  bool driverHasExt = false;
  for (const auto &ext : driverExts)
    if (std::strcmp(ext.name, ZE_FLOAT_ATOMICS_EXT_NAME) == 0)
      driverHasExt = true;

  ze_float_atomic_ext_properties_t floatProps = {};
  floatProps.stype = ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES;
  if (driverHasExt) {
    ze_device_module_properties_t moduleProps = {};
    moduleProps.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
    moduleProps.pNext = &floatProps;
    ASSERT_EQ(zeDeviceGetModuleProperties(zeDevice, &moduleProps),
              ZE_RESULT_SUCCESS);
  }

  uint32_t ipVersion = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_IP_VERSION,
                                 sizeof(ipVersion), &ipVersion, nullptr));
  // Only CRI (and its steppings) reports the extension.
  const bool isCRI = (ipVersion & 0xffffc000) == 0x08c2c000;

  ASSERT_EQ(urReports, isCRI && driverHasExt && floatProps.fp16Flags != 0);
}
