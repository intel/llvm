//===--- pi_unified_runtime.cpp - Unified Runtime PI Plugin  -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

#include <cstring>

#include <pi2ur.hpp>
#include <pi_unified_runtime.hpp>

// Stub function to where all not yet supported PI API are bound
static void DieUnsupported() {
  die("Unified Runtime: functionality is not supported");
}

// All PI API interfaces are C interfaces
extern "C" {
__SYCL_EXPORT pi_result piPlatformsGet(pi_uint32 num_entries,
                                       pi_platform *platforms,
                                       pi_uint32 *num_platforms) {
  return pi2ur::piPlatformsGet(num_entries, platforms, num_platforms);
}

__SYCL_EXPORT pi_result piPlatformGetInfo(pi_platform Platform,
                                          pi_platform_info ParamName,
                                          size_t ParamValueSize,
                                          void *ParamValue,
                                          size_t *ParamValueSizeRet) {
  return pi2ur::piPlatformGetInfo(Platform, ParamName, ParamValueSize,
                                  ParamValue, ParamValueSizeRet);
}

__SYCL_EXPORT pi_result piDevicesGet(pi_platform Platform,
                                     pi_device_type DeviceType,
                                     pi_uint32 NumEntries, pi_device *Devices,
                                     pi_uint32 *NumDevices) {
  (void)Platform;
  (void)DeviceType;
  (void)NumEntries;
  (void)Devices;
  // Report no devices, stab to have a minimal SYCL test running
  if (NumDevices) {
    *NumDevices = 0;
  }
  return PI_SUCCESS;
}

// This interface is not in Unified Runtime currently
__SYCL_EXPORT pi_result piTearDown(void *) { return PI_SUCCESS; }

// This interface is not in Unified Runtime currently
__SYCL_EXPORT pi_result piPluginInit(pi_plugin *PluginInit) {
  PI_ASSERT(PluginInit, PI_ERROR_INVALID_VALUE);

  const char SupportedVersion[] = _PI_UNIFIED_RUNTIME_PLUGIN_VERSION_STRING;

  // Check that the major version matches in PiVersion and SupportedVersion
  _PI_PLUGIN_VERSION_CHECK(PluginInit->PiVersion, SupportedVersion);

  // TODO: handle versioning/targets properly.
  size_t PluginVersionSize = sizeof(PluginInit->PluginVersion);

  PI_ASSERT(strlen(_PI_UNIFIED_RUNTIME_PLUGIN_VERSION_STRING) <
                PluginVersionSize,
            PI_ERROR_INVALID_VALUE);

  strncpy(PluginInit->PluginVersion, SupportedVersion, PluginVersionSize);

  // Bind interfaces that are already supported and "die" for unsupported ones
#define _PI_API(api)                                                           \
  (PluginInit->PiFunctionTable).api = (decltype(&::api))(&DieUnsupported);
#include <sycl/detail/pi.def>

#define _PI_API(api)                                                           \
  (PluginInit->PiFunctionTable).api = (decltype(&::api))(&api);

  _PI_API(piPlatformsGet)
  _PI_API(piPlatformGetInfo)
  _PI_API(piDevicesGet)
  _PI_API(piTearDown)

  return PI_SUCCESS;
}

} // extern "C
