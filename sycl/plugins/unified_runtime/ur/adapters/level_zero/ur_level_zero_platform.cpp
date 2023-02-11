//===--------- ur_level_zero_platform.cpp - Level Zero Adapter --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur_level_zero_platform.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urInit(
    ur_device_init_flags_t
        DeviceFlags ///< [in] device initialization flags.
                    ///< must be 0 (default) or a combination of
                    ///< ::ur_device_init_flag_t.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urTearDown(
    void *Params ///< [in] pointer to tear down parameters
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetApiVersion(
    ur_platform_handle_t Driver, ///< [in] handle of the platform
    ur_api_version_t *Version    ///< [out] api version
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetNativeHandle(
    ur_platform_handle_t Platform,     ///< [in] handle of the platform.
    ur_native_handle_t *NativePlatform ///< [out] a pointer to the native
                                       ///< handle of the platform.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformCreateWithNativeHandle(
    ur_native_handle_t
        NativePlatform,            ///< [in] the native handle of the platform.
    ur_platform_handle_t *Platform ///< [out] pointer to the handle of the
                                   ///< platform object created.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urGetLastResult(
    ur_platform_handle_t Platform, ///< [in] handle of the platform instance
    const char **Message ///< [out] pointer to a string containing adapter
                         ///< specific result in string representation.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
