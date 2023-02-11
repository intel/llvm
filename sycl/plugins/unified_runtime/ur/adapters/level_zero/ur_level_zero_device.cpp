//===--------- ur_level_zero_device.cpp - Level Zero Adapter ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur_level_zero_device.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urDeviceSelectBinary(
    ur_device_handle_t
        Device, ///< [in] handle of the device to select binary for.
    const uint8_t **Binaries, ///< [in] the array of binaries to select from.
    uint32_t NumBinaries, ///< [in] the number of binaries passed in ppBinaries.
                          ///< Must greater than or equal to zero otherwise
                          ///< ::UR_RESULT_ERROR_INVALID_VALUE is returned.
    uint32_t
        *SelectedBinary ///< [out] the index of the selected binary in the input
                        ///< array of binaries. If a suitable binary was not
                        ///< found the function returns ${X}_INVALID_BINARY.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetNativeHandle(
    ur_device_handle_t Device, ///< [in] handle of the device.
    ur_native_handle_t
        *NativeDevice ///< [out] a pointer to the native handle of the device.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceCreateWithNativeHandle(
    ur_native_handle_t NativeDevice, ///< [in] the native handle of the device.
    ur_platform_handle_t Platform,   ///< [in] handle of the platform instance
    ur_device_handle_t
        *Device ///< [out] pointer to the handle of the device object created.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetGlobalTimestamps(
    ur_device_handle_t Device, ///< [in] handle of the device instance
    uint64_t *DeviceTimestamp, ///< [out][optional] pointer to the Device's
                               ///< global timestamp that correlates with the
                               ///< Host's global timestamp value
    uint64_t *HostTimestamp    ///< [out][optional] pointer to the Host's global
                               ///< timestamp that correlates with the Device's
                               ///< global timestamp value
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
