/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_trcddi.cpp
 *
 */

#include "ur_tracing_layer.hpp"
#include <iostream>
#include <stdio.h>

namespace tracing_layer {
///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urInit
__urdlllocal ur_result_t UR_APICALL
urInit(
    ur_device_init_flags_t device_flags ///< [in] device initialization flags.
                                        ///< must be 0 (default) or a combination of ::ur_device_init_flag_t.
) {
    auto pfnInit = context.urDdiTable.Global.pfnInit;

    if (nullptr == pfnInit) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_init_params_t params = {&device_flags};
    uint64_t instance = context.notify_begin(0, "urInit", &params);

    ur_result_t result = pfnInit(device_flags);

    context.notify_end(0, "urInit", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urTearDown
__urdlllocal ur_result_t UR_APICALL
urTearDown(
    void *pParams ///< [in] pointer to tear down parameters
) {
    auto pfnTearDown = context.urDdiTable.Global.pfnTearDown;

    if (nullptr == pfnTearDown) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_tear_down_params_t params = {&pParams};
    uint64_t instance = context.notify_begin(1, "urTearDown", &params);

    ur_result_t result = pfnTearDown(pParams);

    context.notify_end(1, "urTearDown", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGet
__urdlllocal ur_result_t UR_APICALL
urPlatformGet(
    uint32_t NumEntries,               ///< [in] the number of platforms to be added to phPlatforms.
                                       ///< If phPlatforms is not NULL, then NumEntries should be greater than
                                       ///< zero, otherwise ::UR_RESULT_ERROR_INVALID_SIZE,
                                       ///< will be returned.
    ur_platform_handle_t *phPlatforms, ///< [out][optional][range(0, NumEntries)] array of handle of platforms.
                                       ///< If NumEntries is less than the number of platforms available, then
                                       ///< ::urPlatformGet shall only retrieve that number of platforms.
    uint32_t *pNumPlatforms            ///< [out][optional] returns the total number of platforms available.
) {
    auto pfnGet = context.urDdiTable.Platform.pfnGet;

    if (nullptr == pfnGet) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_platform_get_params_t params = {&NumEntries, &phPlatforms, &pNumPlatforms};
    uint64_t instance = context.notify_begin(2, "urPlatformGet", &params);

    ur_result_t result = pfnGet(NumEntries, phPlatforms, pNumPlatforms);

    context.notify_end(2, "urPlatformGet", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetInfo
__urdlllocal ur_result_t UR_APICALL
urPlatformGetInfo(
    ur_platform_handle_t hPlatform,      ///< [in] handle of the platform
    ur_platform_info_t PlatformInfoType, ///< [in] type of the info to retrieve
    size_t Size,                         ///< [in] the number of bytes pointed to by pPlatformInfo.
    void *pPlatformInfo,                 ///< [out][optional] array of bytes holding the info.
                                         ///< If Size is not equal to or greater to the real number of bytes needed
                                         ///< to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
                                         ///< returned and pPlatformInfo is not used.
    size_t *pSizeRet                     ///< [out][optional] pointer to the actual number of bytes being queried by pPlatformInfo.
) {
    auto pfnGetInfo = context.urDdiTable.Platform.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_platform_get_info_params_t params = {&hPlatform, &PlatformInfoType, &Size, &pPlatformInfo, &pSizeRet};
    uint64_t instance = context.notify_begin(3, "urPlatformGetInfo", &params);

    ur_result_t result = pfnGetInfo(hPlatform, PlatformInfoType, Size, pPlatformInfo, pSizeRet);

    context.notify_end(3, "urPlatformGetInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetApiVersion
__urdlllocal ur_result_t UR_APICALL
urPlatformGetApiVersion(
    ur_platform_handle_t hDriver, ///< [in] handle of the platform
    ur_api_version_t *pVersion    ///< [out] api version
) {
    auto pfnGetApiVersion = context.urDdiTable.Platform.pfnGetApiVersion;

    if (nullptr == pfnGetApiVersion) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_platform_get_api_version_params_t params = {&hDriver, &pVersion};
    uint64_t instance = context.notify_begin(4, "urPlatformGetApiVersion", &params);

    ur_result_t result = pfnGetApiVersion(hDriver, pVersion);

    context.notify_end(4, "urPlatformGetApiVersion", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetNativeHandle
__urdlllocal ur_result_t UR_APICALL
urPlatformGetNativeHandle(
    ur_platform_handle_t hPlatform,      ///< [in] handle of the platform.
    ur_native_handle_t *phNativePlatform ///< [out] a pointer to the native handle of the platform.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Platform.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_platform_get_native_handle_params_t params = {&hPlatform, &phNativePlatform};
    uint64_t instance = context.notify_begin(5, "urPlatformGetNativeHandle", &params);

    ur_result_t result = pfnGetNativeHandle(hPlatform, phNativePlatform);

    context.notify_end(5, "urPlatformGetNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL
urPlatformCreateWithNativeHandle(
    ur_native_handle_t hNativePlatform, ///< [in] the native handle of the platform.
    ur_platform_handle_t *phPlatform    ///< [out] pointer to the handle of the platform object created.
) {
    auto pfnCreateWithNativeHandle = context.urDdiTable.Platform.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_platform_create_with_native_handle_params_t params = {&hNativePlatform, &phPlatform};
    uint64_t instance = context.notify_begin(6, "urPlatformCreateWithNativeHandle", &params);

    ur_result_t result = pfnCreateWithNativeHandle(hNativePlatform, phPlatform);

    context.notify_end(6, "urPlatformCreateWithNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urGetLastResult
__urdlllocal ur_result_t UR_APICALL
urGetLastResult(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform instance
    const char **ppMessage          ///< [out] pointer to a string containing adapter specific result in string
                                    ///< representation.
) {
    auto pfnGetLastResult = context.urDdiTable.Global.pfnGetLastResult;

    if (nullptr == pfnGetLastResult) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_get_last_result_params_t params = {&hPlatform, &ppMessage};
    uint64_t instance = context.notify_begin(7, "urGetLastResult", &params);

    ur_result_t result = pfnGetLastResult(hPlatform, ppMessage);

    context.notify_end(7, "urGetLastResult", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGet
__urdlllocal ur_result_t UR_APICALL
urDeviceGet(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform instance
    ur_device_type_t DeviceType,    ///< [in] the type of the devices.
    uint32_t NumEntries,            ///< [in] the number of devices to be added to phDevices.
                                    ///< If phDevices in not NULL then NumEntries should be greater than zero,
                                    ///< otherwise ::UR_RESULT_ERROR_INVALID_VALUE,
                                    ///< will be returned.
    ur_device_handle_t *phDevices,  ///< [out][optional][range(0, NumEntries)] array of handle of devices.
                                    ///< If NumEntries is less than the number of devices available, then
                                    ///< platform shall only retrieve that number of devices.
    uint32_t *pNumDevices           ///< [out][optional] pointer to the number of devices.
                                    ///< pNumDevices will be updated with the total number of devices available.
) {
    auto pfnGet = context.urDdiTable.Device.pfnGet;

    if (nullptr == pfnGet) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_device_get_params_t params = {&hPlatform, &DeviceType, &NumEntries, &phDevices, &pNumDevices};
    uint64_t instance = context.notify_begin(8, "urDeviceGet", &params);

    ur_result_t result = pfnGet(hPlatform, DeviceType, NumEntries, phDevices, pNumDevices);

    context.notify_end(8, "urDeviceGet", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGetInfo
__urdlllocal ur_result_t UR_APICALL
urDeviceGetInfo(
    ur_device_handle_t hDevice, ///< [in] handle of the device instance
    ur_device_info_t infoType,  ///< [in] type of the info to retrieve
    size_t propSize,            ///< [in] the number of bytes pointed to by pDeviceInfo.
    void *pDeviceInfo,          ///< [out][optional] array of bytes holding the info.
                                ///< If propSize is not equal to or greater than the real number of bytes
                                ///< needed to return the info
                                ///< then the ::UR_RESULT_ERROR_INVALID_VALUE error is returned and
                                ///< pDeviceInfo is not used.
    size_t *pPropSizeRet        ///< [out][optional] pointer to the actual size in bytes of the queried infoType.
) {
    auto pfnGetInfo = context.urDdiTable.Device.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_device_get_info_params_t params = {&hDevice, &infoType, &propSize, &pDeviceInfo, &pPropSizeRet};
    uint64_t instance = context.notify_begin(9, "urDeviceGetInfo", &params);

    ur_result_t result = pfnGetInfo(hDevice, infoType, propSize, pDeviceInfo, pPropSizeRet);

    context.notify_end(9, "urDeviceGetInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRetain
__urdlllocal ur_result_t UR_APICALL
urDeviceRetain(
    ur_device_handle_t hDevice ///< [in] handle of the device to get a reference of.
) {
    auto pfnRetain = context.urDdiTable.Device.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_device_retain_params_t params = {&hDevice};
    uint64_t instance = context.notify_begin(10, "urDeviceRetain", &params);

    ur_result_t result = pfnRetain(hDevice);

    context.notify_end(10, "urDeviceRetain", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRelease
__urdlllocal ur_result_t UR_APICALL
urDeviceRelease(
    ur_device_handle_t hDevice ///< [in] handle of the device to release.
) {
    auto pfnRelease = context.urDdiTable.Device.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_device_release_params_t params = {&hDevice};
    uint64_t instance = context.notify_begin(11, "urDeviceRelease", &params);

    ur_result_t result = pfnRelease(hDevice);

    context.notify_end(11, "urDeviceRelease", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDevicePartition
__urdlllocal ur_result_t UR_APICALL
urDevicePartition(
    ur_device_handle_t hDevice,                        ///< [in] handle of the device to partition.
    const ur_device_partition_property_t *pProperties, ///< [in] null-terminated array of <$_device_partition_t enum, value> pairs.
    uint32_t NumDevices,                               ///< [in] the number of sub-devices.
    ur_device_handle_t *phSubDevices,                  ///< [out][optional][range(0, NumDevices)] array of handle of devices.
                                                       ///< If NumDevices is less than the number of sub-devices available, then
                                                       ///< the function shall only retrieve that number of sub-devices.
    uint32_t *pNumDevicesRet                           ///< [out][optional] pointer to the number of sub-devices the device can be
                                                       ///< partitioned into according to the partitioning property.
) {
    auto pfnPartition = context.urDdiTable.Device.pfnPartition;

    if (nullptr == pfnPartition) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_device_partition_params_t params = {&hDevice, &pProperties, &NumDevices, &phSubDevices, &pNumDevicesRet};
    uint64_t instance = context.notify_begin(12, "urDevicePartition", &params);

    ur_result_t result = pfnPartition(hDevice, pProperties, NumDevices, phSubDevices, pNumDevicesRet);

    context.notify_end(12, "urDevicePartition", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceSelectBinary
__urdlllocal ur_result_t UR_APICALL
urDeviceSelectBinary(
    ur_device_handle_t hDevice, ///< [in] handle of the device to select binary for.
    const uint8_t **ppBinaries, ///< [in] the array of binaries to select from.
    uint32_t NumBinaries,       ///< [in] the number of binaries passed in ppBinaries.
                                ///< Must greater than or equal to zero otherwise
                                ///< ::UR_RESULT_ERROR_INVALID_VALUE is returned.
    uint32_t *pSelectedBinary   ///< [out] the index of the selected binary in the input array of binaries.
                                ///< If a suitable binary was not found the function returns ${X}_INVALID_BINARY.
) {
    auto pfnSelectBinary = context.urDdiTable.Device.pfnSelectBinary;

    if (nullptr == pfnSelectBinary) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_device_select_binary_params_t params = {&hDevice, &ppBinaries, &NumBinaries, &pSelectedBinary};
    uint64_t instance = context.notify_begin(13, "urDeviceSelectBinary", &params);

    ur_result_t result = pfnSelectBinary(hDevice, ppBinaries, NumBinaries, pSelectedBinary);

    context.notify_end(13, "urDeviceSelectBinary", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGetNativeHandle
__urdlllocal ur_result_t UR_APICALL
urDeviceGetNativeHandle(
    ur_device_handle_t hDevice,        ///< [in] handle of the device.
    ur_native_handle_t *phNativeDevice ///< [out] a pointer to the native handle of the device.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Device.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_device_get_native_handle_params_t params = {&hDevice, &phNativeDevice};
    uint64_t instance = context.notify_begin(14, "urDeviceGetNativeHandle", &params);

    ur_result_t result = pfnGetNativeHandle(hDevice, phNativeDevice);

    context.notify_end(14, "urDeviceGetNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL
urDeviceCreateWithNativeHandle(
    ur_native_handle_t hNativeDevice, ///< [in] the native handle of the device.
    ur_platform_handle_t hPlatform,   ///< [in] handle of the platform instance
    ur_device_handle_t *phDevice      ///< [out] pointer to the handle of the device object created.
) {
    auto pfnCreateWithNativeHandle = context.urDdiTable.Device.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_device_create_with_native_handle_params_t params = {&hNativeDevice, &hPlatform, &phDevice};
    uint64_t instance = context.notify_begin(15, "urDeviceCreateWithNativeHandle", &params);

    ur_result_t result = pfnCreateWithNativeHandle(hNativeDevice, hPlatform, phDevice);

    context.notify_end(15, "urDeviceCreateWithNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGetGlobalTimestamps
__urdlllocal ur_result_t UR_APICALL
urDeviceGetGlobalTimestamps(
    ur_device_handle_t hDevice, ///< [in] handle of the device instance
    uint64_t *pDeviceTimestamp, ///< [out][optional] pointer to the Device's global timestamp that
                                ///< correlates with the Host's global timestamp value
    uint64_t *pHostTimestamp    ///< [out][optional] pointer to the Host's global timestamp that
                                ///< correlates with the Device's global timestamp value
) {
    auto pfnGetGlobalTimestamps = context.urDdiTable.Device.pfnGetGlobalTimestamps;

    if (nullptr == pfnGetGlobalTimestamps) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_device_get_global_timestamps_params_t params = {&hDevice, &pDeviceTimestamp, &pHostTimestamp};
    uint64_t instance = context.notify_begin(16, "urDeviceGetGlobalTimestamps", &params);

    ur_result_t result = pfnGetGlobalTimestamps(hDevice, pDeviceTimestamp, pHostTimestamp);

    context.notify_end(16, "urDeviceGetGlobalTimestamps", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreate
__urdlllocal ur_result_t UR_APICALL
urContextCreate(
    uint32_t DeviceCount,                       ///< [in] the number of devices given in phDevices
    const ur_device_handle_t *phDevices,        ///< [in][range(0, DeviceCount)] array of handle of devices.
    const ur_context_properties_t *pProperties, ///< [in][optional] pointer to context creation properties.
    ur_context_handle_t *phContext              ///< [out] pointer to handle of context object created
) {
    auto pfnCreate = context.urDdiTable.Context.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_context_create_params_t params = {&DeviceCount, &phDevices, &pProperties, &phContext};
    uint64_t instance = context.notify_begin(17, "urContextCreate", &params);

    ur_result_t result = pfnCreate(DeviceCount, phDevices, pProperties, phContext);

    context.notify_end(17, "urContextCreate", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRetain
__urdlllocal ur_result_t UR_APICALL
urContextRetain(
    ur_context_handle_t hContext ///< [in] handle of the context to get a reference of.
) {
    auto pfnRetain = context.urDdiTable.Context.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_context_retain_params_t params = {&hContext};
    uint64_t instance = context.notify_begin(18, "urContextRetain", &params);

    ur_result_t result = pfnRetain(hContext);

    context.notify_end(18, "urContextRetain", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRelease
__urdlllocal ur_result_t UR_APICALL
urContextRelease(
    ur_context_handle_t hContext ///< [in] handle of the context to release.
) {
    auto pfnRelease = context.urDdiTable.Context.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_context_release_params_t params = {&hContext};
    uint64_t instance = context.notify_begin(19, "urContextRelease", &params);

    ur_result_t result = pfnRelease(hContext);

    context.notify_end(19, "urContextRelease", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextGetInfo
__urdlllocal ur_result_t UR_APICALL
urContextGetInfo(
    ur_context_handle_t hContext,      ///< [in] handle of the context
    ur_context_info_t ContextInfoType, ///< [in] type of the info to retrieve
    size_t propSize,                   ///< [in] the number of bytes of memory pointed to by pContextInfo.
    void *pContextInfo,                ///< [out][optional] array of bytes holding the info.
                                       ///< if propSize is not equal to or greater than the real number of bytes
                                       ///< needed to return
                                       ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                       ///< pContextInfo is not used.
    size_t *pPropSizeRet               ///< [out][optional] pointer to the actual size in bytes of data queried by ContextInfoType.
) {
    auto pfnGetInfo = context.urDdiTable.Context.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_context_get_info_params_t params = {&hContext, &ContextInfoType, &propSize, &pContextInfo, &pPropSizeRet};
    uint64_t instance = context.notify_begin(20, "urContextGetInfo", &params);

    ur_result_t result = pfnGetInfo(hContext, ContextInfoType, propSize, pContextInfo, pPropSizeRet);

    context.notify_end(20, "urContextGetInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextGetNativeHandle
__urdlllocal ur_result_t UR_APICALL
urContextGetNativeHandle(
    ur_context_handle_t hContext,       ///< [in] handle of the context.
    ur_native_handle_t *phNativeContext ///< [out] a pointer to the native handle of the context.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Context.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_context_get_native_handle_params_t params = {&hContext, &phNativeContext};
    uint64_t instance = context.notify_begin(21, "urContextGetNativeHandle", &params);

    ur_result_t result = pfnGetNativeHandle(hContext, phNativeContext);

    context.notify_end(21, "urContextGetNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL
urContextCreateWithNativeHandle(
    ur_native_handle_t hNativeContext, ///< [in] the native handle of the context.
    ur_context_handle_t *phContext     ///< [out] pointer to the handle of the context object created.
) {
    auto pfnCreateWithNativeHandle = context.urDdiTable.Context.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_context_create_with_native_handle_params_t params = {&hNativeContext, &phContext};
    uint64_t instance = context.notify_begin(22, "urContextCreateWithNativeHandle", &params);

    ur_result_t result = pfnCreateWithNativeHandle(hNativeContext, phContext);

    context.notify_end(22, "urContextCreateWithNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextSetExtendedDeleter
__urdlllocal ur_result_t UR_APICALL
urContextSetExtendedDeleter(
    ur_context_handle_t hContext,             ///< [in] handle of the context.
    ur_context_extended_deleter_t pfnDeleter, ///< [in] Function pointer to extended deleter.
    void *pUserData                           ///< [in][out][optional] pointer to data to be passed to callback.
) {
    auto pfnSetExtendedDeleter = context.urDdiTable.Context.pfnSetExtendedDeleter;

    if (nullptr == pfnSetExtendedDeleter) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_context_set_extended_deleter_params_t params = {&hContext, &pfnDeleter, &pUserData};
    uint64_t instance = context.notify_begin(23, "urContextSetExtendedDeleter", &params);

    ur_result_t result = pfnSetExtendedDeleter(hContext, pfnDeleter, pUserData);

    context.notify_end(23, "urContextSetExtendedDeleter", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemImageCreate
__urdlllocal ur_result_t UR_APICALL
urMemImageCreate(
    ur_context_handle_t hContext,          ///< [in] handle of the context object
    ur_mem_flags_t flags,                  ///< [in] allocation and usage information flags
    const ur_image_format_t *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc,     ///< [in] pointer to image description
    void *pHost,                           ///< [in] pointer to the buffer data
    ur_mem_handle_t *phMem                 ///< [out] pointer to handle of image object created
) {
    auto pfnImageCreate = context.urDdiTable.Mem.pfnImageCreate;

    if (nullptr == pfnImageCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_mem_image_create_params_t params = {&hContext, &flags, &pImageFormat, &pImageDesc, &pHost, &phMem};
    uint64_t instance = context.notify_begin(24, "urMemImageCreate", &params);

    ur_result_t result = pfnImageCreate(hContext, flags, pImageFormat, pImageDesc, pHost, phMem);

    context.notify_end(24, "urMemImageCreate", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferCreate
__urdlllocal ur_result_t UR_APICALL
urMemBufferCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_mem_flags_t flags,         ///< [in] allocation and usage information flags
    size_t size,                  ///< [in] size in bytes of the memory object to be allocated
    void *pHost,                  ///< [in][optional] pointer to the buffer data
    ur_mem_handle_t *phBuffer     ///< [out] pointer to handle of the memory buffer created
) {
    auto pfnBufferCreate = context.urDdiTable.Mem.pfnBufferCreate;

    if (nullptr == pfnBufferCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_mem_buffer_create_params_t params = {&hContext, &flags, &size, &pHost, &phBuffer};
    uint64_t instance = context.notify_begin(25, "urMemBufferCreate", &params);

    ur_result_t result = pfnBufferCreate(hContext, flags, size, pHost, phBuffer);

    context.notify_end(25, "urMemBufferCreate", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRetain
__urdlllocal ur_result_t UR_APICALL
urMemRetain(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to get access
) {
    auto pfnRetain = context.urDdiTable.Mem.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_mem_retain_params_t params = {&hMem};
    uint64_t instance = context.notify_begin(26, "urMemRetain", &params);

    ur_result_t result = pfnRetain(hMem);

    context.notify_end(26, "urMemRetain", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRelease
__urdlllocal ur_result_t UR_APICALL
urMemRelease(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to release
) {
    auto pfnRelease = context.urDdiTable.Mem.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_mem_release_params_t params = {&hMem};
    uint64_t instance = context.notify_begin(27, "urMemRelease", &params);

    ur_result_t result = pfnRelease(hMem);

    context.notify_end(27, "urMemRelease", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferPartition
__urdlllocal ur_result_t UR_APICALL
urMemBufferPartition(
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object to allocate from
    ur_mem_flags_t flags,                     ///< [in] allocation and usage information flags
    ur_buffer_create_type_t bufferCreateType, ///< [in] buffer creation type
    ur_buffer_region_t *pBufferCreateInfo,    ///< [in] pointer to buffer create region information
    ur_mem_handle_t *phMem                    ///< [out] pointer to the handle of sub buffer created
) {
    auto pfnBufferPartition = context.urDdiTable.Mem.pfnBufferPartition;

    if (nullptr == pfnBufferPartition) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_mem_buffer_partition_params_t params = {&hBuffer, &flags, &bufferCreateType, &pBufferCreateInfo, &phMem};
    uint64_t instance = context.notify_begin(28, "urMemBufferPartition", &params);

    ur_result_t result = pfnBufferPartition(hBuffer, flags, bufferCreateType, pBufferCreateInfo, phMem);

    context.notify_end(28, "urMemBufferPartition", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetNativeHandle
__urdlllocal ur_result_t UR_APICALL
urMemGetNativeHandle(
    ur_mem_handle_t hMem,           ///< [in] handle of the mem.
    ur_native_handle_t *phNativeMem ///< [out] a pointer to the native handle of the mem.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Mem.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_mem_get_native_handle_params_t params = {&hMem, &phNativeMem};
    uint64_t instance = context.notify_begin(29, "urMemGetNativeHandle", &params);

    ur_result_t result = pfnGetNativeHandle(hMem, phNativeMem);

    context.notify_end(29, "urMemGetNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL
urMemCreateWithNativeHandle(
    ur_native_handle_t hNativeMem, ///< [in] the native handle of the mem.
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    ur_mem_handle_t *phMem         ///< [out] pointer to the handle of the mem object created.
) {
    auto pfnCreateWithNativeHandle = context.urDdiTable.Mem.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_mem_create_with_native_handle_params_t params = {&hNativeMem, &hContext, &phMem};
    uint64_t instance = context.notify_begin(30, "urMemCreateWithNativeHandle", &params);

    ur_result_t result = pfnCreateWithNativeHandle(hNativeMem, hContext, phMem);

    context.notify_end(30, "urMemCreateWithNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetInfo
__urdlllocal ur_result_t UR_APICALL
urMemGetInfo(
    ur_mem_handle_t hMemory,   ///< [in] handle to the memory object being queried.
    ur_mem_info_t MemInfoType, ///< [in] type of the info to retrieve.
    size_t propSize,           ///< [in] the number of bytes of memory pointed to by pMemInfo.
    void *pMemInfo,            ///< [out][optional] array of bytes holding the info.
                               ///< If propSize is less than the real number of bytes needed to return
                               ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                               ///< pMemInfo is not used.
    size_t *pPropSizeRet       ///< [out][optional] pointer to the actual size in bytes of data queried by pMemInfo.
) {
    auto pfnGetInfo = context.urDdiTable.Mem.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_mem_get_info_params_t params = {&hMemory, &MemInfoType, &propSize, &pMemInfo, &pPropSizeRet};
    uint64_t instance = context.notify_begin(31, "urMemGetInfo", &params);

    ur_result_t result = pfnGetInfo(hMemory, MemInfoType, propSize, pMemInfo, pPropSizeRet);

    context.notify_end(31, "urMemGetInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemImageGetInfo
__urdlllocal ur_result_t UR_APICALL
urMemImageGetInfo(
    ur_mem_handle_t hMemory,     ///< [in] handle to the image object being queried.
    ur_image_info_t ImgInfoType, ///< [in] type of image info to retrieve.
    size_t propSize,             ///< [in] the number of bytes of memory pointer to by pImgInfo.
    void *pImgInfo,              ///< [out][optional] array of bytes holding the info.
                                 ///< If propSize is less than the real number of bytes needed to return
                                 ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                 ///< pImgInfo is not used.
    size_t *pPropSizeRet         ///< [out][optional] pointer to the actual size in bytes of data queried by pImgInfo.
) {
    auto pfnImageGetInfo = context.urDdiTable.Mem.pfnImageGetInfo;

    if (nullptr == pfnImageGetInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_mem_image_get_info_params_t params = {&hMemory, &ImgInfoType, &propSize, &pImgInfo, &pPropSizeRet};
    uint64_t instance = context.notify_begin(32, "urMemImageGetInfo", &params);

    ur_result_t result = pfnImageGetInfo(hMemory, ImgInfoType, propSize, pImgInfo, pPropSizeRet);

    context.notify_end(32, "urMemImageGetInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerCreate
__urdlllocal ur_result_t UR_APICALL
urSamplerCreate(
    ur_context_handle_t hContext,        ///< [in] handle of the context object
    const ur_sampler_property_t *pProps, ///< [in] specifies a list of sampler property names and their
                                         ///< corresponding values.
    ur_sampler_handle_t *phSampler       ///< [out] pointer to handle of sampler object created
) {
    auto pfnCreate = context.urDdiTable.Sampler.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_sampler_create_params_t params = {&hContext, &pProps, &phSampler};
    uint64_t instance = context.notify_begin(33, "urSamplerCreate", &params);

    ur_result_t result = pfnCreate(hContext, pProps, phSampler);

    context.notify_end(33, "urSamplerCreate", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRetain
__urdlllocal ur_result_t UR_APICALL
urSamplerRetain(
    ur_sampler_handle_t hSampler ///< [in] handle of the sampler object to get access
) {
    auto pfnRetain = context.urDdiTable.Sampler.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_sampler_retain_params_t params = {&hSampler};
    uint64_t instance = context.notify_begin(34, "urSamplerRetain", &params);

    ur_result_t result = pfnRetain(hSampler);

    context.notify_end(34, "urSamplerRetain", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRelease
__urdlllocal ur_result_t UR_APICALL
urSamplerRelease(
    ur_sampler_handle_t hSampler ///< [in] handle of the sampler object to release
) {
    auto pfnRelease = context.urDdiTable.Sampler.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_sampler_release_params_t params = {&hSampler};
    uint64_t instance = context.notify_begin(35, "urSamplerRelease", &params);

    ur_result_t result = pfnRelease(hSampler);

    context.notify_end(35, "urSamplerRelease", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerGetInfo
__urdlllocal ur_result_t UR_APICALL
urSamplerGetInfo(
    ur_sampler_handle_t hSampler, ///< [in] handle of the sampler object
    ur_sampler_info_t propName,   ///< [in] name of the sampler property to query
    size_t propValueSize,         ///< [in] size in bytes of the sampler property value provided
    void *pPropValue,             ///< [out] value of the sampler property
    size_t *pPropSizeRet          ///< [out] size in bytes returned in sampler property value
) {
    auto pfnGetInfo = context.urDdiTable.Sampler.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_sampler_get_info_params_t params = {&hSampler, &propName, &propValueSize, &pPropValue, &pPropSizeRet};
    uint64_t instance = context.notify_begin(36, "urSamplerGetInfo", &params);

    ur_result_t result = pfnGetInfo(hSampler, propName, propValueSize, pPropValue, pPropSizeRet);

    context.notify_end(36, "urSamplerGetInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerGetNativeHandle
__urdlllocal ur_result_t UR_APICALL
urSamplerGetNativeHandle(
    ur_sampler_handle_t hSampler,       ///< [in] handle of the sampler.
    ur_native_handle_t *phNativeSampler ///< [out] a pointer to the native handle of the sampler.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Sampler.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_sampler_get_native_handle_params_t params = {&hSampler, &phNativeSampler};
    uint64_t instance = context.notify_begin(37, "urSamplerGetNativeHandle", &params);

    ur_result_t result = pfnGetNativeHandle(hSampler, phNativeSampler);

    context.notify_end(37, "urSamplerGetNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL
urSamplerCreateWithNativeHandle(
    ur_native_handle_t hNativeSampler, ///< [in] the native handle of the sampler.
    ur_context_handle_t hContext,      ///< [in] handle of the context object
    ur_sampler_handle_t *phSampler     ///< [out] pointer to the handle of the sampler object created.
) {
    auto pfnCreateWithNativeHandle = context.urDdiTable.Sampler.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_sampler_create_with_native_handle_params_t params = {&hNativeSampler, &hContext, &phSampler};
    uint64_t instance = context.notify_begin(38, "urSamplerCreateWithNativeHandle", &params);

    ur_result_t result = pfnCreateWithNativeHandle(hNativeSampler, hContext, phSampler);

    context.notify_end(38, "urSamplerCreateWithNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMHostAlloc
__urdlllocal ur_result_t UR_APICALL
urUSMHostAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_usm_desc_t *pUSMDesc,      ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t pool,    ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t size,                  ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t align,               ///< [in] alignment of the USM memory object
                                  ///< Must be zero or a power of 2.
                                  ///< Must be equal to or smaller than the size of the largest data type
                                  ///< supported by `hDevice`.
    void **ppMem                  ///< [out] pointer to USM host memory object
) {
    auto pfnHostAlloc = context.urDdiTable.USM.pfnHostAlloc;

    if (nullptr == pfnHostAlloc) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_usm_host_alloc_params_t params = {&hContext, &pUSMDesc, &pool, &size, &align, &ppMem};
    uint64_t instance = context.notify_begin(39, "urUSMHostAlloc", &params);

    ur_result_t result = pfnHostAlloc(hContext, pUSMDesc, pool, size, align, ppMem);

    context.notify_end(39, "urUSMHostAlloc", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMDeviceAlloc
__urdlllocal ur_result_t UR_APICALL
urUSMDeviceAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_usm_desc_t *pUSMDesc,      ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t pool,    ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t size,                  ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t align,               ///< [in] alignment of the USM memory object
                                  ///< Must be zero or a power of 2.
                                  ///< Must be equal to or smaller than the size of the largest data type
                                  ///< supported by `hDevice`.
    void **ppMem                  ///< [out] pointer to USM device memory object
) {
    auto pfnDeviceAlloc = context.urDdiTable.USM.pfnDeviceAlloc;

    if (nullptr == pfnDeviceAlloc) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_usm_device_alloc_params_t params = {&hContext, &hDevice, &pUSMDesc, &pool, &size, &align, &ppMem};
    uint64_t instance = context.notify_begin(40, "urUSMDeviceAlloc", &params);

    ur_result_t result = pfnDeviceAlloc(hContext, hDevice, pUSMDesc, pool, size, align, ppMem);

    context.notify_end(40, "urUSMDeviceAlloc", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMSharedAlloc
__urdlllocal ur_result_t UR_APICALL
urUSMSharedAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_usm_desc_t *pUSMDesc,      ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t pool,    ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t size,                  ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t align,               ///< [in] alignment of the USM memory object.
                                  ///< Must be zero or a power of 2.
                                  ///< Must be equal to or smaller than the size of the largest data type
                                  ///< supported by `hDevice`.
    void **ppMem                  ///< [out] pointer to USM shared memory object
) {
    auto pfnSharedAlloc = context.urDdiTable.USM.pfnSharedAlloc;

    if (nullptr == pfnSharedAlloc) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_usm_shared_alloc_params_t params = {&hContext, &hDevice, &pUSMDesc, &pool, &size, &align, &ppMem};
    uint64_t instance = context.notify_begin(41, "urUSMSharedAlloc", &params);

    ur_result_t result = pfnSharedAlloc(hContext, hDevice, pUSMDesc, pool, size, align, ppMem);

    context.notify_end(41, "urUSMSharedAlloc", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMFree
__urdlllocal ur_result_t UR_APICALL
urUSMFree(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to USM memory object
) {
    auto pfnFree = context.urDdiTable.USM.pfnFree;

    if (nullptr == pfnFree) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_usm_free_params_t params = {&hContext, &pMem};
    uint64_t instance = context.notify_begin(42, "urUSMFree", &params);

    ur_result_t result = pfnFree(hContext, pMem);

    context.notify_end(42, "urUSMFree", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMGetMemAllocInfo
__urdlllocal ur_result_t UR_APICALL
urUSMGetMemAllocInfo(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const void *pMem,             ///< [in] pointer to USM memory object
    ur_usm_alloc_info_t propName, ///< [in] the name of the USM allocation property to query
    size_t propValueSize,         ///< [in] size in bytes of the USM allocation property value
    void *pPropValue,             ///< [out][optional] value of the USM allocation property
    size_t *pPropValueSizeRet     ///< [out][optional] bytes returned in USM allocation property
) {
    auto pfnGetMemAllocInfo = context.urDdiTable.USM.pfnGetMemAllocInfo;

    if (nullptr == pfnGetMemAllocInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_usm_get_mem_alloc_info_params_t params = {&hContext, &pMem, &propName, &propValueSize, &pPropValue, &pPropValueSizeRet};
    uint64_t instance = context.notify_begin(43, "urUSMGetMemAllocInfo", &params);

    ur_result_t result = pfnGetMemAllocInfo(hContext, pMem, propName, propValueSize, pPropValue, pPropValueSizeRet);

    context.notify_end(43, "urUSMGetMemAllocInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolCreate
__urdlllocal ur_result_t UR_APICALL
urUSMPoolCreate(
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    ur_usm_pool_desc_t *pPoolDesc, ///< [in] pointer to USM pool descriptor. Can be chained with
                                   ///< ::ur_usm_pool_limits_desc_t
    ur_usm_pool_handle_t *ppPool   ///< [out] pointer to USM memory pool
) {
    auto pfnPoolCreate = context.urDdiTable.USM.pfnPoolCreate;

    if (nullptr == pfnPoolCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_usm_pool_create_params_t params = {&hContext, &pPoolDesc, &ppPool};
    uint64_t instance = context.notify_begin(44, "urUSMPoolCreate", &params);

    ur_result_t result = pfnPoolCreate(hContext, pPoolDesc, ppPool);

    context.notify_end(44, "urUSMPoolCreate", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolDestroy
__urdlllocal ur_result_t UR_APICALL
urUSMPoolDestroy(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_usm_pool_handle_t pPool    ///< [in] pointer to USM memory pool
) {
    auto pfnPoolDestroy = context.urDdiTable.USM.pfnPoolDestroy;

    if (nullptr == pfnPoolDestroy) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_usm_pool_destroy_params_t params = {&hContext, &pPool};
    uint64_t instance = context.notify_begin(45, "urUSMPoolDestroy", &params);

    ur_result_t result = pfnPoolDestroy(hContext, pPool);

    context.notify_end(45, "urUSMPoolDestroy", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urModuleCreate
__urdlllocal ur_result_t UR_APICALL
urModuleCreate(
    ur_context_handle_t hContext,         ///< [in] handle of the context instance.
    const void *pIL,                      ///< [in] pointer to IL string.
    size_t length,                        ///< [in] length of IL in bytes.
    const char *pOptions,                 ///< [in] pointer to compiler options null-terminated string.
    ur_modulecreate_callback_t pfnNotify, ///< [in][optional] A function pointer to a notification routine that is
                                          ///< called when program compilation is complete.
    void *pUserData,                      ///< [in][optional] Passed as an argument when pfnNotify is called.
    ur_module_handle_t *phModule          ///< [out] pointer to handle of Module object created.
) {
    auto pfnCreate = context.urDdiTable.Module.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_module_create_params_t params = {&hContext, &pIL, &length, &pOptions, &pfnNotify, &pUserData, &phModule};
    uint64_t instance = context.notify_begin(46, "urModuleCreate", &params);

    ur_result_t result = pfnCreate(hContext, pIL, length, pOptions, pfnNotify, pUserData, phModule);

    context.notify_end(46, "urModuleCreate", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urModuleRetain
__urdlllocal ur_result_t UR_APICALL
urModuleRetain(
    ur_module_handle_t hModule ///< [in] handle for the Module to retain
) {
    auto pfnRetain = context.urDdiTable.Module.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_module_retain_params_t params = {&hModule};
    uint64_t instance = context.notify_begin(47, "urModuleRetain", &params);

    ur_result_t result = pfnRetain(hModule);

    context.notify_end(47, "urModuleRetain", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urModuleRelease
__urdlllocal ur_result_t UR_APICALL
urModuleRelease(
    ur_module_handle_t hModule ///< [in] handle for the Module to release
) {
    auto pfnRelease = context.urDdiTable.Module.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_module_release_params_t params = {&hModule};
    uint64_t instance = context.notify_begin(48, "urModuleRelease", &params);

    ur_result_t result = pfnRelease(hModule);

    context.notify_end(48, "urModuleRelease", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urModuleGetNativeHandle
__urdlllocal ur_result_t UR_APICALL
urModuleGetNativeHandle(
    ur_module_handle_t hModule,        ///< [in] handle of the module.
    ur_native_handle_t *phNativeModule ///< [out] a pointer to the native handle of the module.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Module.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_module_get_native_handle_params_t params = {&hModule, &phNativeModule};
    uint64_t instance = context.notify_begin(49, "urModuleGetNativeHandle", &params);

    ur_result_t result = pfnGetNativeHandle(hModule, phNativeModule);

    context.notify_end(49, "urModuleGetNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urModuleCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL
urModuleCreateWithNativeHandle(
    ur_native_handle_t hNativeModule, ///< [in] the native handle of the module.
    ur_context_handle_t hContext,     ///< [in] handle of the context instance.
    ur_module_handle_t *phModule      ///< [out] pointer to the handle of the module object created.
) {
    auto pfnCreateWithNativeHandle = context.urDdiTable.Module.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_module_create_with_native_handle_params_t params = {&hNativeModule, &hContext, &phModule};
    uint64_t instance = context.notify_begin(50, "urModuleCreateWithNativeHandle", &params);

    ur_result_t result = pfnCreateWithNativeHandle(hNativeModule, hContext, phModule);

    context.notify_end(50, "urModuleCreateWithNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreate
__urdlllocal ur_result_t UR_APICALL
urProgramCreate(
    ur_context_handle_t hContext,               ///< [in] handle of the context instance
    uint32_t count,                             ///< [in] number of module handles in module list.
    const ur_module_handle_t *phModules,        ///< [in][range(0, count)] pointer to array of modules.
    const char *pOptions,                       ///< [in][optional] pointer to linker options null-terminated string.
    const ur_program_properties_t *pProperties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t *phProgram              ///< [out] pointer to handle of program object created.
) {
    auto pfnCreate = context.urDdiTable.Program.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_program_create_params_t params = {&hContext, &count, &phModules, &pOptions, &pProperties, &phProgram};
    uint64_t instance = context.notify_begin(51, "urProgramCreate", &params);

    ur_result_t result = pfnCreate(hContext, count, phModules, pOptions, pProperties, phProgram);

    context.notify_end(51, "urProgramCreate", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithBinary
__urdlllocal ur_result_t UR_APICALL
urProgramCreateWithBinary(
    ur_context_handle_t hContext,               ///< [in] handle of the context instance
    ur_device_handle_t hDevice,                 ///< [in] handle to device associated with binary.
    size_t size,                                ///< [in] size in bytes.
    const uint8_t *pBinary,                     ///< [in] pointer to binary.
    const ur_program_properties_t *pProperties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t *phProgram              ///< [out] pointer to handle of Program object created.
) {
    auto pfnCreateWithBinary = context.urDdiTable.Program.pfnCreateWithBinary;

    if (nullptr == pfnCreateWithBinary) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_program_create_with_binary_params_t params = {&hContext, &hDevice, &size, &pBinary, &pProperties, &phProgram};
    uint64_t instance = context.notify_begin(52, "urProgramCreateWithBinary", &params);

    ur_result_t result = pfnCreateWithBinary(hContext, hDevice, size, pBinary, pProperties, phProgram);

    context.notify_end(52, "urProgramCreateWithBinary", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRetain
__urdlllocal ur_result_t UR_APICALL
urProgramRetain(
    ur_program_handle_t hProgram ///< [in] handle for the Program to retain
) {
    auto pfnRetain = context.urDdiTable.Program.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_program_retain_params_t params = {&hProgram};
    uint64_t instance = context.notify_begin(53, "urProgramRetain", &params);

    ur_result_t result = pfnRetain(hProgram);

    context.notify_end(53, "urProgramRetain", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRelease
__urdlllocal ur_result_t UR_APICALL
urProgramRelease(
    ur_program_handle_t hProgram ///< [in] handle for the Program to release
) {
    auto pfnRelease = context.urDdiTable.Program.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_program_release_params_t params = {&hProgram};
    uint64_t instance = context.notify_begin(54, "urProgramRelease", &params);

    ur_result_t result = pfnRelease(hProgram);

    context.notify_end(54, "urProgramRelease", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetFunctionPointer
__urdlllocal ur_result_t UR_APICALL
urProgramGetFunctionPointer(
    ur_device_handle_t hDevice,   ///< [in] handle of the device to retrieve pointer for.
    ur_program_handle_t hProgram, ///< [in] handle of the program to search for function in.
                                  ///< The program must already be built to the specified device, or
                                  ///< otherwise ::UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE is returned.
    const char *pFunctionName,    ///< [in] A null-terminates string denoting the mangled function name.
    void **ppFunctionPointer      ///< [out] Returns the pointer to the function if it is found in the program.
) {
    auto pfnGetFunctionPointer = context.urDdiTable.Program.pfnGetFunctionPointer;

    if (nullptr == pfnGetFunctionPointer) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_program_get_function_pointer_params_t params = {&hDevice, &hProgram, &pFunctionName, &ppFunctionPointer};
    uint64_t instance = context.notify_begin(55, "urProgramGetFunctionPointer", &params);

    ur_result_t result = pfnGetFunctionPointer(hDevice, hProgram, pFunctionName, ppFunctionPointer);

    context.notify_end(55, "urProgramGetFunctionPointer", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetInfo
__urdlllocal ur_result_t UR_APICALL
urProgramGetInfo(
    ur_program_handle_t hProgram, ///< [in] handle of the Program object
    ur_program_info_t propName,   ///< [in] name of the Program property to query
    size_t propSize,              ///< [in] the size of the Program property.
    void *pProgramInfo,           ///< [in,out][optional] array of bytes of holding the program info property.
                                  ///< If propSize is not equal to or greater than the real number of bytes
                                  ///< needed to return
                                  ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                  ///< pProgramInfo is not used.
    size_t *pPropSizeRet          ///< [out][optional] pointer to the actual size in bytes of data copied to pProgramInfo.
) {
    auto pfnGetInfo = context.urDdiTable.Program.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_program_get_info_params_t params = {&hProgram, &propName, &propSize, &pProgramInfo, &pPropSizeRet};
    uint64_t instance = context.notify_begin(56, "urProgramGetInfo", &params);

    ur_result_t result = pfnGetInfo(hProgram, propName, propSize, pProgramInfo, pPropSizeRet);

    context.notify_end(56, "urProgramGetInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetBuildInfo
__urdlllocal ur_result_t UR_APICALL
urProgramGetBuildInfo(
    ur_program_handle_t hProgram,     ///< [in] handle of the Program object
    ur_device_handle_t hDevice,       ///< [in] handle of the Device object
    ur_program_build_info_t propName, ///< [in] name of the Program build info to query
    size_t propSize,                  ///< [in] size of the Program build info property.
    void *pPropValue,                 ///< [in,out][optional] value of the Program build property.
                                      ///< If propSize is not equal to or greater than the real number of bytes
                                      ///< needed to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE
                                      ///< error is returned and pKernelInfo is not used.
    size_t *pPropSizeRet              ///< [out][optional] pointer to the actual size in bytes of data being
                                      ///< queried by propName.
) {
    auto pfnGetBuildInfo = context.urDdiTable.Program.pfnGetBuildInfo;

    if (nullptr == pfnGetBuildInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_program_get_build_info_params_t params = {&hProgram, &hDevice, &propName, &propSize, &pPropValue, &pPropSizeRet};
    uint64_t instance = context.notify_begin(57, "urProgramGetBuildInfo", &params);

    ur_result_t result = pfnGetBuildInfo(hProgram, hDevice, propName, propSize, pPropValue, pPropSizeRet);

    context.notify_end(57, "urProgramGetBuildInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramSetSpecializationConstants
__urdlllocal ur_result_t UR_APICALL
urProgramSetSpecializationConstants(
    ur_program_handle_t hProgram,                           ///< [in] handle of the Program object
    uint32_t count,                                         ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t *pSpecConstants ///< [in][range(0, count)] array of specialization constant value
                                                            ///< descriptions
) {
    auto pfnSetSpecializationConstants = context.urDdiTable.Program.pfnSetSpecializationConstants;

    if (nullptr == pfnSetSpecializationConstants) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_program_set_specialization_constants_params_t params = {&hProgram, &count, &pSpecConstants};
    uint64_t instance = context.notify_begin(58, "urProgramSetSpecializationConstants", &params);

    ur_result_t result = pfnSetSpecializationConstants(hProgram, count, pSpecConstants);

    context.notify_end(58, "urProgramSetSpecializationConstants", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetNativeHandle
__urdlllocal ur_result_t UR_APICALL
urProgramGetNativeHandle(
    ur_program_handle_t hProgram,       ///< [in] handle of the program.
    ur_native_handle_t *phNativeProgram ///< [out] a pointer to the native handle of the program.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Program.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_program_get_native_handle_params_t params = {&hProgram, &phNativeProgram};
    uint64_t instance = context.notify_begin(59, "urProgramGetNativeHandle", &params);

    ur_result_t result = pfnGetNativeHandle(hProgram, phNativeProgram);

    context.notify_end(59, "urProgramGetNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL
urProgramCreateWithNativeHandle(
    ur_native_handle_t hNativeProgram, ///< [in] the native handle of the program.
    ur_context_handle_t hContext,      ///< [in] handle of the context instance
    ur_program_handle_t *phProgram     ///< [out] pointer to the handle of the program object created.
) {
    auto pfnCreateWithNativeHandle = context.urDdiTable.Program.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_program_create_with_native_handle_params_t params = {&hNativeProgram, &hContext, &phProgram};
    uint64_t instance = context.notify_begin(60, "urProgramCreateWithNativeHandle", &params);

    ur_result_t result = pfnCreateWithNativeHandle(hNativeProgram, hContext, phProgram);

    context.notify_end(60, "urProgramCreateWithNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelCreate
__urdlllocal ur_result_t UR_APICALL
urKernelCreate(
    ur_program_handle_t hProgram, ///< [in] handle of the program instance
    const char *pKernelName,      ///< [in] pointer to null-terminated string.
    ur_kernel_handle_t *phKernel  ///< [out] pointer to handle of kernel object created.
) {
    auto pfnCreate = context.urDdiTable.Kernel.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_create_params_t params = {&hProgram, &pKernelName, &phKernel};
    uint64_t instance = context.notify_begin(61, "urKernelCreate", &params);

    ur_result_t result = pfnCreate(hProgram, pKernelName, phKernel);

    context.notify_end(61, "urKernelCreate", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgValue
__urdlllocal ur_result_t UR_APICALL
urKernelSetArgValue(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex,          ///< [in] argument index in range [0, num args - 1]
    size_t argSize,             ///< [in] size of argument type
    const void *pArgValue       ///< [in] argument value represented as matching arg type.
) {
    auto pfnSetArgValue = context.urDdiTable.Kernel.pfnSetArgValue;

    if (nullptr == pfnSetArgValue) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_set_arg_value_params_t params = {&hKernel, &argIndex, &argSize, &pArgValue};
    uint64_t instance = context.notify_begin(62, "urKernelSetArgValue", &params);

    ur_result_t result = pfnSetArgValue(hKernel, argIndex, argSize, pArgValue);

    context.notify_end(62, "urKernelSetArgValue", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgLocal
__urdlllocal ur_result_t UR_APICALL
urKernelSetArgLocal(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex,          ///< [in] argument index in range [0, num args - 1]
    size_t argSize              ///< [in] size of the local buffer to be allocated by the runtime
) {
    auto pfnSetArgLocal = context.urDdiTable.Kernel.pfnSetArgLocal;

    if (nullptr == pfnSetArgLocal) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_set_arg_local_params_t params = {&hKernel, &argIndex, &argSize};
    uint64_t instance = context.notify_begin(63, "urKernelSetArgLocal", &params);

    ur_result_t result = pfnSetArgLocal(hKernel, argIndex, argSize);

    context.notify_end(63, "urKernelSetArgLocal", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetInfo
__urdlllocal ur_result_t UR_APICALL
urKernelGetInfo(
    ur_kernel_handle_t hKernel, ///< [in] handle of the Kernel object
    ur_kernel_info_t propName,  ///< [in] name of the Kernel property to query
    size_t propSize,            ///< [in] the size of the Kernel property value.
    void *pKernelInfo,          ///< [in,out][optional] array of bytes holding the kernel info property.
                                ///< If propSize is not equal to or greater than the real number of bytes
                                ///< needed to return
                                ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                ///< pKernelInfo is not used.
    size_t *pPropSizeRet        ///< [out][optional] pointer to the actual size in bytes of data being
                                ///< queried by propName.
) {
    auto pfnGetInfo = context.urDdiTable.Kernel.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_get_info_params_t params = {&hKernel, &propName, &propSize, &pKernelInfo, &pPropSizeRet};
    uint64_t instance = context.notify_begin(64, "urKernelGetInfo", &params);

    ur_result_t result = pfnGetInfo(hKernel, propName, propSize, pKernelInfo, pPropSizeRet);

    context.notify_end(64, "urKernelGetInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetGroupInfo
__urdlllocal ur_result_t UR_APICALL
urKernelGetGroupInfo(
    ur_kernel_handle_t hKernel,      ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice,      ///< [in] handle of the Device object
    ur_kernel_group_info_t propName, ///< [in] name of the work Group property to query
    size_t propSize,                 ///< [in] size of the Kernel Work Group property value
    void *pPropValue,                ///< [in,out][optional][range(0, propSize)] value of the Kernel Work Group
                                     ///< property.
    size_t *pPropSizeRet             ///< [out][optional] pointer to the actual size in bytes of data being
                                     ///< queried by propName.
) {
    auto pfnGetGroupInfo = context.urDdiTable.Kernel.pfnGetGroupInfo;

    if (nullptr == pfnGetGroupInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_get_group_info_params_t params = {&hKernel, &hDevice, &propName, &propSize, &pPropValue, &pPropSizeRet};
    uint64_t instance = context.notify_begin(65, "urKernelGetGroupInfo", &params);

    ur_result_t result = pfnGetGroupInfo(hKernel, hDevice, propName, propSize, pPropValue, pPropSizeRet);

    context.notify_end(65, "urKernelGetGroupInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetSubGroupInfo
__urdlllocal ur_result_t UR_APICALL
urKernelGetSubGroupInfo(
    ur_kernel_handle_t hKernel,          ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice,          ///< [in] handle of the Device object
    ur_kernel_sub_group_info_t propName, ///< [in] name of the SubGroup property to query
    size_t propSize,                     ///< [in] size of the Kernel SubGroup property value
    void *pPropValue,                    ///< [in,out][range(0, propSize)][optional] value of the Kernel SubGroup
                                         ///< property.
    size_t *pPropSizeRet                 ///< [out][optional] pointer to the actual size in bytes of data being
                                         ///< queried by propName.
) {
    auto pfnGetSubGroupInfo = context.urDdiTable.Kernel.pfnGetSubGroupInfo;

    if (nullptr == pfnGetSubGroupInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_get_sub_group_info_params_t params = {&hKernel, &hDevice, &propName, &propSize, &pPropValue, &pPropSizeRet};
    uint64_t instance = context.notify_begin(66, "urKernelGetSubGroupInfo", &params);

    ur_result_t result = pfnGetSubGroupInfo(hKernel, hDevice, propName, propSize, pPropValue, pPropSizeRet);

    context.notify_end(66, "urKernelGetSubGroupInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRetain
__urdlllocal ur_result_t UR_APICALL
urKernelRetain(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to retain
) {
    auto pfnRetain = context.urDdiTable.Kernel.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_retain_params_t params = {&hKernel};
    uint64_t instance = context.notify_begin(67, "urKernelRetain", &params);

    ur_result_t result = pfnRetain(hKernel);

    context.notify_end(67, "urKernelRetain", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRelease
__urdlllocal ur_result_t UR_APICALL
urKernelRelease(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to release
) {
    auto pfnRelease = context.urDdiTable.Kernel.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_release_params_t params = {&hKernel};
    uint64_t instance = context.notify_begin(68, "urKernelRelease", &params);

    ur_result_t result = pfnRelease(hKernel);

    context.notify_end(68, "urKernelRelease", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgPointer
__urdlllocal ur_result_t UR_APICALL
urKernelSetArgPointer(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex,          ///< [in] argument index in range [0, num args - 1]
    size_t argSize,             ///< [in] size of argument type
    const void *pArgValue       ///< [in][optional] SVM pointer to memory location holding the argument
                                ///< value. If null then argument value is considered null.
) {
    auto pfnSetArgPointer = context.urDdiTable.Kernel.pfnSetArgPointer;

    if (nullptr == pfnSetArgPointer) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_set_arg_pointer_params_t params = {&hKernel, &argIndex, &argSize, &pArgValue};
    uint64_t instance = context.notify_begin(69, "urKernelSetArgPointer", &params);

    ur_result_t result = pfnSetArgPointer(hKernel, argIndex, argSize, pArgValue);

    context.notify_end(69, "urKernelSetArgPointer", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetExecInfo
__urdlllocal ur_result_t UR_APICALL
urKernelSetExecInfo(
    ur_kernel_handle_t hKernel,     ///< [in] handle of the kernel object
    ur_kernel_exec_info_t propName, ///< [in] name of the execution attribute
    size_t propSize,                ///< [in] size in byte the attribute value
    const void *pPropValue          ///< [in][range(0, propSize)] pointer to memory location holding the
                                    ///< property value.
) {
    auto pfnSetExecInfo = context.urDdiTable.Kernel.pfnSetExecInfo;

    if (nullptr == pfnSetExecInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_set_exec_info_params_t params = {&hKernel, &propName, &propSize, &pPropValue};
    uint64_t instance = context.notify_begin(70, "urKernelSetExecInfo", &params);

    ur_result_t result = pfnSetExecInfo(hKernel, propName, propSize, pPropValue);

    context.notify_end(70, "urKernelSetExecInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgSampler
__urdlllocal ur_result_t UR_APICALL
urKernelSetArgSampler(
    ur_kernel_handle_t hKernel,   ///< [in] handle of the kernel object
    uint32_t argIndex,            ///< [in] argument index in range [0, num args - 1]
    ur_sampler_handle_t hArgValue ///< [in] handle of Sampler object.
) {
    auto pfnSetArgSampler = context.urDdiTable.Kernel.pfnSetArgSampler;

    if (nullptr == pfnSetArgSampler) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_set_arg_sampler_params_t params = {&hKernel, &argIndex, &hArgValue};
    uint64_t instance = context.notify_begin(71, "urKernelSetArgSampler", &params);

    ur_result_t result = pfnSetArgSampler(hKernel, argIndex, hArgValue);

    context.notify_end(71, "urKernelSetArgSampler", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgMemObj
__urdlllocal ur_result_t UR_APICALL
urKernelSetArgMemObj(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex,          ///< [in] argument index in range [0, num args - 1]
    ur_mem_handle_t hArgValue   ///< [in][optional] handle of Memory object.
) {
    auto pfnSetArgMemObj = context.urDdiTable.Kernel.pfnSetArgMemObj;

    if (nullptr == pfnSetArgMemObj) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_set_arg_mem_obj_params_t params = {&hKernel, &argIndex, &hArgValue};
    uint64_t instance = context.notify_begin(72, "urKernelSetArgMemObj", &params);

    ur_result_t result = pfnSetArgMemObj(hKernel, argIndex, hArgValue);

    context.notify_end(72, "urKernelSetArgMemObj", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetSpecializationConstants
__urdlllocal ur_result_t UR_APICALL
urKernelSetSpecializationConstants(
    ur_kernel_handle_t hKernel,                             ///< [in] handle of the kernel object
    uint32_t count,                                         ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t *pSpecConstants ///< [in] array of specialization constant value descriptions
) {
    auto pfnSetSpecializationConstants = context.urDdiTable.Kernel.pfnSetSpecializationConstants;

    if (nullptr == pfnSetSpecializationConstants) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_set_specialization_constants_params_t params = {&hKernel, &count, &pSpecConstants};
    uint64_t instance = context.notify_begin(73, "urKernelSetSpecializationConstants", &params);

    ur_result_t result = pfnSetSpecializationConstants(hKernel, count, pSpecConstants);

    context.notify_end(73, "urKernelSetSpecializationConstants", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetNativeHandle
__urdlllocal ur_result_t UR_APICALL
urKernelGetNativeHandle(
    ur_kernel_handle_t hKernel,        ///< [in] handle of the kernel.
    ur_native_handle_t *phNativeKernel ///< [out] a pointer to the native handle of the kernel.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Kernel.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_get_native_handle_params_t params = {&hKernel, &phNativeKernel};
    uint64_t instance = context.notify_begin(74, "urKernelGetNativeHandle", &params);

    ur_result_t result = pfnGetNativeHandle(hKernel, phNativeKernel);

    context.notify_end(74, "urKernelGetNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL
urKernelCreateWithNativeHandle(
    ur_native_handle_t hNativeKernel, ///< [in] the native handle of the kernel.
    ur_context_handle_t hContext,     ///< [in] handle of the context object
    ur_kernel_handle_t *phKernel      ///< [out] pointer to the handle of the kernel object created.
) {
    auto pfnCreateWithNativeHandle = context.urDdiTable.Kernel.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_kernel_create_with_native_handle_params_t params = {&hNativeKernel, &hContext, &phKernel};
    uint64_t instance = context.notify_begin(75, "urKernelCreateWithNativeHandle", &params);

    ur_result_t result = pfnCreateWithNativeHandle(hNativeKernel, hContext, phKernel);

    context.notify_end(75, "urKernelCreateWithNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueGetInfo
__urdlllocal ur_result_t UR_APICALL
urQueueGetInfo(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_queue_info_t propName, ///< [in] name of the queue property to query
    size_t propValueSize,     ///< [in] size in bytes of the queue property value provided
    void *pPropValue,         ///< [out][optional] value of the queue property
    size_t *pPropSizeRet      ///< [out][optional] size in bytes returned in queue property value
) {
    auto pfnGetInfo = context.urDdiTable.Queue.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_queue_get_info_params_t params = {&hQueue, &propName, &propValueSize, &pPropValue, &pPropSizeRet};
    uint64_t instance = context.notify_begin(76, "urQueueGetInfo", &params);

    ur_result_t result = pfnGetInfo(hQueue, propName, propValueSize, pPropValue, pPropSizeRet);

    context.notify_end(76, "urQueueGetInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueCreate
__urdlllocal ur_result_t UR_APICALL
urQueueCreate(
    ur_context_handle_t hContext,      ///< [in] handle of the context object
    ur_device_handle_t hDevice,        ///< [in] handle of the device object
    const ur_queue_property_t *pProps, ///< [in][optional] specifies a list of queue properties and their
                                       ///< corresponding values.
                                       ///< Each property name is immediately followed by the corresponding
                                       ///< desired value.
                                       ///< The list is terminated with a 0.
                                       ///< If a property value is not specified, then its default value will be used.
    ur_queue_handle_t *phQueue         ///< [out] pointer to handle of queue object created
) {
    auto pfnCreate = context.urDdiTable.Queue.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_queue_create_params_t params = {&hContext, &hDevice, &pProps, &phQueue};
    uint64_t instance = context.notify_begin(77, "urQueueCreate", &params);

    ur_result_t result = pfnCreate(hContext, hDevice, pProps, phQueue);

    context.notify_end(77, "urQueueCreate", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRetain
__urdlllocal ur_result_t UR_APICALL
urQueueRetain(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to get access
) {
    auto pfnRetain = context.urDdiTable.Queue.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_queue_retain_params_t params = {&hQueue};
    uint64_t instance = context.notify_begin(78, "urQueueRetain", &params);

    ur_result_t result = pfnRetain(hQueue);

    context.notify_end(78, "urQueueRetain", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRelease
__urdlllocal ur_result_t UR_APICALL
urQueueRelease(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to release
) {
    auto pfnRelease = context.urDdiTable.Queue.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_queue_release_params_t params = {&hQueue};
    uint64_t instance = context.notify_begin(79, "urQueueRelease", &params);

    ur_result_t result = pfnRelease(hQueue);

    context.notify_end(79, "urQueueRelease", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueGetNativeHandle
__urdlllocal ur_result_t UR_APICALL
urQueueGetNativeHandle(
    ur_queue_handle_t hQueue,         ///< [in] handle of the queue.
    ur_native_handle_t *phNativeQueue ///< [out] a pointer to the native handle of the queue.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Queue.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_queue_get_native_handle_params_t params = {&hQueue, &phNativeQueue};
    uint64_t instance = context.notify_begin(80, "urQueueGetNativeHandle", &params);

    ur_result_t result = pfnGetNativeHandle(hQueue, phNativeQueue);

    context.notify_end(80, "urQueueGetNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL
urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue, ///< [in] the native handle of the queue.
    ur_context_handle_t hContext,    ///< [in] handle of the context object
    ur_queue_handle_t *phQueue       ///< [out] pointer to the handle of the queue object created.
) {
    auto pfnCreateWithNativeHandle = context.urDdiTable.Queue.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_queue_create_with_native_handle_params_t params = {&hNativeQueue, &hContext, &phQueue};
    uint64_t instance = context.notify_begin(81, "urQueueCreateWithNativeHandle", &params);

    ur_result_t result = pfnCreateWithNativeHandle(hNativeQueue, hContext, phQueue);

    context.notify_end(81, "urQueueCreateWithNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFinish
__urdlllocal ur_result_t UR_APICALL
urQueueFinish(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be finished.
) {
    auto pfnFinish = context.urDdiTable.Queue.pfnFinish;

    if (nullptr == pfnFinish) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_queue_finish_params_t params = {&hQueue};
    uint64_t instance = context.notify_begin(82, "urQueueFinish", &params);

    ur_result_t result = pfnFinish(hQueue);

    context.notify_end(82, "urQueueFinish", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFlush
__urdlllocal ur_result_t UR_APICALL
urQueueFlush(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be flushed.
) {
    auto pfnFlush = context.urDdiTable.Queue.pfnFlush;

    if (nullptr == pfnFlush) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_queue_flush_params_t params = {&hQueue};
    uint64_t instance = context.notify_begin(83, "urQueueFlush", &params);

    ur_result_t result = pfnFlush(hQueue);

    context.notify_end(83, "urQueueFlush", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventGetInfo
__urdlllocal ur_result_t UR_APICALL
urEventGetInfo(
    ur_event_handle_t hEvent, ///< [in] handle of the event object
    ur_event_info_t propName, ///< [in] the name of the event property to query
    size_t propValueSize,     ///< [in] size in bytes of the event property value
    void *pPropValue,         ///< [out][optional] value of the event property
    size_t *pPropValueSizeRet ///< [out][optional] bytes returned in event property
) {
    auto pfnGetInfo = context.urDdiTable.Event.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_event_get_info_params_t params = {&hEvent, &propName, &propValueSize, &pPropValue, &pPropValueSizeRet};
    uint64_t instance = context.notify_begin(84, "urEventGetInfo", &params);

    ur_result_t result = pfnGetInfo(hEvent, propName, propValueSize, pPropValue, pPropValueSizeRet);

    context.notify_end(84, "urEventGetInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventGetProfilingInfo
__urdlllocal ur_result_t UR_APICALL
urEventGetProfilingInfo(
    ur_event_handle_t hEvent,     ///< [in] handle of the event object
    ur_profiling_info_t propName, ///< [in] the name of the profiling property to query
    size_t propValueSize,         ///< [in] size in bytes of the profiling property value
    void *pPropValue,             ///< [out][optional] value of the profiling property
    size_t *pPropValueSizeRet     ///< [out][optional] pointer to the actual size in bytes returned in
                                  ///< propValue
) {
    auto pfnGetProfilingInfo = context.urDdiTable.Event.pfnGetProfilingInfo;

    if (nullptr == pfnGetProfilingInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_event_get_profiling_info_params_t params = {&hEvent, &propName, &propValueSize, &pPropValue, &pPropValueSizeRet};
    uint64_t instance = context.notify_begin(85, "urEventGetProfilingInfo", &params);

    ur_result_t result = pfnGetProfilingInfo(hEvent, propName, propValueSize, pPropValue, pPropValueSizeRet);

    context.notify_end(85, "urEventGetProfilingInfo", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventWait
__urdlllocal ur_result_t UR_APICALL
urEventWait(
    uint32_t numEvents,                      ///< [in] number of events in the event list
    const ur_event_handle_t *phEventWaitList ///< [in][range(0, numEvents)] pointer to a list of events to wait for
                                             ///< completion
) {
    auto pfnWait = context.urDdiTable.Event.pfnWait;

    if (nullptr == pfnWait) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_event_wait_params_t params = {&numEvents, &phEventWaitList};
    uint64_t instance = context.notify_begin(86, "urEventWait", &params);

    ur_result_t result = pfnWait(numEvents, phEventWaitList);

    context.notify_end(86, "urEventWait", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRetain
__urdlllocal ur_result_t UR_APICALL
urEventRetain(
    ur_event_handle_t hEvent ///< [in] handle of the event object
) {
    auto pfnRetain = context.urDdiTable.Event.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_event_retain_params_t params = {&hEvent};
    uint64_t instance = context.notify_begin(87, "urEventRetain", &params);

    ur_result_t result = pfnRetain(hEvent);

    context.notify_end(87, "urEventRetain", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRelease
__urdlllocal ur_result_t UR_APICALL
urEventRelease(
    ur_event_handle_t hEvent ///< [in] handle of the event object
) {
    auto pfnRelease = context.urDdiTable.Event.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_event_release_params_t params = {&hEvent};
    uint64_t instance = context.notify_begin(88, "urEventRelease", &params);

    ur_result_t result = pfnRelease(hEvent);

    context.notify_end(88, "urEventRelease", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventGetNativeHandle
__urdlllocal ur_result_t UR_APICALL
urEventGetNativeHandle(
    ur_event_handle_t hEvent,         ///< [in] handle of the event.
    ur_native_handle_t *phNativeEvent ///< [out] a pointer to the native handle of the event.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Event.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_event_get_native_handle_params_t params = {&hEvent, &phNativeEvent};
    uint64_t instance = context.notify_begin(89, "urEventGetNativeHandle", &params);

    ur_result_t result = pfnGetNativeHandle(hEvent, phNativeEvent);

    context.notify_end(89, "urEventGetNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL
urEventCreateWithNativeHandle(
    ur_native_handle_t hNativeEvent, ///< [in] the native handle of the event.
    ur_context_handle_t hContext,    ///< [in] handle of the context object
    ur_event_handle_t *phEvent       ///< [out] pointer to the handle of the event object created.
) {
    auto pfnCreateWithNativeHandle = context.urDdiTable.Event.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_event_create_with_native_handle_params_t params = {&hNativeEvent, &hContext, &phEvent};
    uint64_t instance = context.notify_begin(90, "urEventCreateWithNativeHandle", &params);

    ur_result_t result = pfnCreateWithNativeHandle(hNativeEvent, hContext, phEvent);

    context.notify_end(90, "urEventCreateWithNativeHandle", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventSetCallback
__urdlllocal ur_result_t UR_APICALL
urEventSetCallback(
    ur_event_handle_t hEvent,       ///< [in] handle of the event object
    ur_execution_info_t execStatus, ///< [in] execution status of the event
    ur_event_callback_t pfnNotify,  ///< [in] execution status of the event
    void *pUserData                 ///< [in][out][optional] pointer to data to be passed to callback.
) {
    auto pfnSetCallback = context.urDdiTable.Event.pfnSetCallback;

    if (nullptr == pfnSetCallback) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_event_set_callback_params_t params = {&hEvent, &execStatus, &pfnNotify, &pUserData};
    uint64_t instance = context.notify_begin(91, "urEventSetCallback", &params);

    ur_result_t result = pfnSetCallback(hEvent, execStatus, pfnNotify, pUserData);

    context.notify_end(91, "urEventSetCallback", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueKernelLaunch
__urdlllocal ur_result_t UR_APICALL
urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_kernel_handle_t hKernel,               ///< [in] handle of the kernel object
    uint32_t workDim,                         ///< [in] number of dimensions, from 1 to 3, to specify the global and
                                              ///< work-group work-items
    const size_t *pGlobalWorkOffset,          ///< [in] pointer to an array of workDim unsigned values that specify the
                                              ///< offset used to calculate the global ID of a work-item
    const size_t *pGlobalWorkSize,            ///< [in] pointer to an array of workDim unsigned values that specify the
                                              ///< number of global work-items in workDim that will execute the kernel
                                              ///< function
    const size_t *pLocalWorkSize,             ///< [in][optional] pointer to an array of workDim unsigned values that
                                              ///< specify the number of local work-items forming a work-group that will
                                              ///< execute the kernel function.
                                              ///< If nullptr, the runtime implementation will choose the work-group
                                              ///< size.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular kernel execution instance.
) {
    auto pfnKernelLaunch = context.urDdiTable.Enqueue.pfnKernelLaunch;

    if (nullptr == pfnKernelLaunch) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_kernel_launch_params_t params = {&hQueue, &hKernel, &workDim, &pGlobalWorkOffset, &pGlobalWorkSize, &pLocalWorkSize, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(92, "urEnqueueKernelLaunch", &params);

    ur_result_t result = pfnKernelLaunch(hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(92, "urEnqueueKernelLaunch", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueEventsWait
__urdlllocal ur_result_t UR_APICALL
urEnqueueEventsWait(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
                                              ///< previously enqueued commands
                                              ///< must be complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnEventsWait = context.urDdiTable.Enqueue.pfnEventsWait;

    if (nullptr == pfnEventsWait) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_events_wait_params_t params = {&hQueue, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(93, "urEnqueueEventsWait", &params);

    ur_result_t result = pfnEventsWait(hQueue, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(93, "urEnqueueEventsWait", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueEventsWaitWithBarrier
__urdlllocal ur_result_t UR_APICALL
urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
                                              ///< previously enqueued commands
                                              ///< must be complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnEventsWaitWithBarrier = context.urDdiTable.Enqueue.pfnEventsWaitWithBarrier;

    if (nullptr == pfnEventsWaitWithBarrier) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_events_wait_with_barrier_params_t params = {&hQueue, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(94, "urEnqueueEventsWaitWithBarrier", &params);

    ur_result_t result = pfnEventsWaitWithBarrier(hQueue, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(94, "urEnqueueEventsWaitWithBarrier", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferRead
__urdlllocal ur_result_t UR_APICALL
urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object
    bool blockingRead,                        ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,                            ///< [in] offset in bytes in the buffer object
    size_t size,                              ///< [in] size in bytes of data being read
    void *pDst,                               ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnMemBufferRead = context.urDdiTable.Enqueue.pfnMemBufferRead;

    if (nullptr == pfnMemBufferRead) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_mem_buffer_read_params_t params = {&hQueue, &hBuffer, &blockingRead, &offset, &size, &pDst, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(95, "urEnqueueMemBufferRead", &params);

    ur_result_t result = pfnMemBufferRead(hQueue, hBuffer, blockingRead, offset, size, pDst, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(95, "urEnqueueMemBufferRead", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferWrite
__urdlllocal ur_result_t UR_APICALL
urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object
    bool blockingWrite,                       ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,                            ///< [in] offset in bytes in the buffer object
    size_t size,                              ///< [in] size in bytes of data being written
    const void *pSrc,                         ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnMemBufferWrite = context.urDdiTable.Enqueue.pfnMemBufferWrite;

    if (nullptr == pfnMemBufferWrite) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_mem_buffer_write_params_t params = {&hQueue, &hBuffer, &blockingWrite, &offset, &size, &pSrc, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(96, "urEnqueueMemBufferWrite", &params);

    ur_result_t result = pfnMemBufferWrite(hQueue, hBuffer, blockingWrite, offset, size, pSrc, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(96, "urEnqueueMemBufferWrite", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferReadRect
__urdlllocal ur_result_t UR_APICALL
urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object
    bool blockingRead,                        ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOffset,            ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOffset,              ///< [in] 3D offset in the host region
    ur_rect_region_t region,                  ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t bufferRowPitch,                    ///< [in] length of each row in bytes in the buffer object
    size_t bufferSlicePitch,                  ///< [in] length of each 2D slice in bytes in the buffer object being read
    size_t hostRowPitch,                      ///< [in] length of each row in bytes in the host memory region pointed by
                                              ///< dst
    size_t hostSlicePitch,                    ///< [in] length of each 2D slice in bytes in the host memory region
                                              ///< pointed by dst
    void *pDst,                               ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnMemBufferReadRect = context.urDdiTable.Enqueue.pfnMemBufferReadRect;

    if (nullptr == pfnMemBufferReadRect) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_mem_buffer_read_rect_params_t params = {&hQueue, &hBuffer, &blockingRead, &bufferOffset, &hostOffset, &region, &bufferRowPitch, &bufferSlicePitch, &hostRowPitch, &hostSlicePitch, &pDst, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(97, "urEnqueueMemBufferReadRect", &params);

    ur_result_t result = pfnMemBufferReadRect(hQueue, hBuffer, blockingRead, bufferOffset, hostOffset, region, bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(97, "urEnqueueMemBufferReadRect", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferWriteRect
__urdlllocal ur_result_t UR_APICALL
urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object
    bool blockingWrite,                       ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOffset,            ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOffset,              ///< [in] 3D offset in the host region
    ur_rect_region_t region,                  ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t bufferRowPitch,                    ///< [in] length of each row in bytes in the buffer object
    size_t bufferSlicePitch,                  ///< [in] length of each 2D slice in bytes in the buffer object being
                                              ///< written
    size_t hostRowPitch,                      ///< [in] length of each row in bytes in the host memory region pointed by
                                              ///< src
    size_t hostSlicePitch,                    ///< [in] length of each 2D slice in bytes in the host memory region
                                              ///< pointed by src
    void *pSrc,                               ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] points to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnMemBufferWriteRect = context.urDdiTable.Enqueue.pfnMemBufferWriteRect;

    if (nullptr == pfnMemBufferWriteRect) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_mem_buffer_write_rect_params_t params = {&hQueue, &hBuffer, &blockingWrite, &bufferOffset, &hostOffset, &region, &bufferRowPitch, &bufferSlicePitch, &hostRowPitch, &hostSlicePitch, &pSrc, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(98, "urEnqueueMemBufferWriteRect", &params);

    ur_result_t result = pfnMemBufferWriteRect(hQueue, hBuffer, blockingWrite, bufferOffset, hostOffset, region, bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(98, "urEnqueueMemBufferWriteRect", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferCopy
__urdlllocal ur_result_t UR_APICALL
urEnqueueMemBufferCopy(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBufferSrc,               ///< [in] handle of the src buffer object
    ur_mem_handle_t hBufferDst,               ///< [in] handle of the dest buffer object
    size_t srcOffset,                         ///< [in] offset into hBufferSrc to begin copying from
    size_t dstOffset,                         ///< [in] offset info hBufferDst to begin copying into
    size_t size,                              ///< [in] size in bytes of data being copied
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnMemBufferCopy = context.urDdiTable.Enqueue.pfnMemBufferCopy;

    if (nullptr == pfnMemBufferCopy) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_mem_buffer_copy_params_t params = {&hQueue, &hBufferSrc, &hBufferDst, &srcOffset, &dstOffset, &size, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(99, "urEnqueueMemBufferCopy", &params);

    ur_result_t result = pfnMemBufferCopy(hQueue, hBufferSrc, hBufferDst, srcOffset, dstOffset, size, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(99, "urEnqueueMemBufferCopy", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferCopyRect
__urdlllocal ur_result_t UR_APICALL
urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBufferSrc,               ///< [in] handle of the source buffer object
    ur_mem_handle_t hBufferDst,               ///< [in] handle of the dest buffer object
    ur_rect_offset_t srcOrigin,               ///< [in] 3D offset in the source buffer
    ur_rect_offset_t dstOrigin,               ///< [in] 3D offset in the destination buffer
    ur_rect_region_t srcRegion,               ///< [in] source 3D rectangular region descriptor: width, height, depth
    size_t srcRowPitch,                       ///< [in] length of each row in bytes in the source buffer object
    size_t srcSlicePitch,                     ///< [in] length of each 2D slice in bytes in the source buffer object
    size_t dstRowPitch,                       ///< [in] length of each row in bytes in the destination buffer object
    size_t dstSlicePitch,                     ///< [in] length of each 2D slice in bytes in the destination buffer object
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnMemBufferCopyRect = context.urDdiTable.Enqueue.pfnMemBufferCopyRect;

    if (nullptr == pfnMemBufferCopyRect) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_mem_buffer_copy_rect_params_t params = {&hQueue, &hBufferSrc, &hBufferDst, &srcOrigin, &dstOrigin, &srcRegion, &srcRowPitch, &srcSlicePitch, &dstRowPitch, &dstSlicePitch, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(100, "urEnqueueMemBufferCopyRect", &params);

    ur_result_t result = pfnMemBufferCopyRect(hQueue, hBufferSrc, hBufferDst, srcOrigin, dstOrigin, srcRegion, srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(100, "urEnqueueMemBufferCopyRect", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferFill
__urdlllocal ur_result_t UR_APICALL
urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object
    const void *pPattern,                     ///< [in] pointer to the fill pattern
    size_t patternSize,                       ///< [in] size in bytes of the pattern
    size_t offset,                            ///< [in] offset into the buffer
    size_t size,                              ///< [in] fill size in bytes, must be a multiple of patternSize
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnMemBufferFill = context.urDdiTable.Enqueue.pfnMemBufferFill;

    if (nullptr == pfnMemBufferFill) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_mem_buffer_fill_params_t params = {&hQueue, &hBuffer, &pPattern, &patternSize, &offset, &size, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(101, "urEnqueueMemBufferFill", &params);

    ur_result_t result = pfnMemBufferFill(hQueue, hBuffer, pPattern, patternSize, offset, size, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(101, "urEnqueueMemBufferFill", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemImageRead
__urdlllocal ur_result_t UR_APICALL
urEnqueueMemImageRead(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hImage,                   ///< [in] handle of the image object
    bool blockingRead,                        ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t origin,                  ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_region_t region,                  ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                              ///< image
    size_t rowPitch,                          ///< [in] length of each row in bytes
    size_t slicePitch,                        ///< [in] length of each 2D slice of the 3D image
    void *pDst,                               ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnMemImageRead = context.urDdiTable.Enqueue.pfnMemImageRead;

    if (nullptr == pfnMemImageRead) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_mem_image_read_params_t params = {&hQueue, &hImage, &blockingRead, &origin, &region, &rowPitch, &slicePitch, &pDst, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(102, "urEnqueueMemImageRead", &params);

    ur_result_t result = pfnMemImageRead(hQueue, hImage, blockingRead, origin, region, rowPitch, slicePitch, pDst, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(102, "urEnqueueMemImageRead", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemImageWrite
__urdlllocal ur_result_t UR_APICALL
urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hImage,                   ///< [in] handle of the image object
    bool blockingWrite,                       ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t origin,                  ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_region_t region,                  ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                              ///< image
    size_t inputRowPitch,                     ///< [in] length of each row in bytes
    size_t inputSlicePitch,                   ///< [in] length of each 2D slice of the 3D image
    void *pSrc,                               ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnMemImageWrite = context.urDdiTable.Enqueue.pfnMemImageWrite;

    if (nullptr == pfnMemImageWrite) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_mem_image_write_params_t params = {&hQueue, &hImage, &blockingWrite, &origin, &region, &inputRowPitch, &inputSlicePitch, &pSrc, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(103, "urEnqueueMemImageWrite", &params);

    ur_result_t result = pfnMemImageWrite(hQueue, hImage, blockingWrite, origin, region, inputRowPitch, inputSlicePitch, pSrc, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(103, "urEnqueueMemImageWrite", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemImageCopy
__urdlllocal ur_result_t UR_APICALL
urEnqueueMemImageCopy(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hImageSrc,                ///< [in] handle of the src image object
    ur_mem_handle_t hImageDst,                ///< [in] handle of the dest image object
    ur_rect_offset_t srcOrigin,               ///< [in] defines the (x,y,z) offset in pixels in the source 1D, 2D, or 3D
                                              ///< image
    ur_rect_offset_t dstOrigin,               ///< [in] defines the (x,y,z) offset in pixels in the destination 1D, 2D,
                                              ///< or 3D image
    ur_rect_region_t region,                  ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                              ///< image
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnMemImageCopy = context.urDdiTable.Enqueue.pfnMemImageCopy;

    if (nullptr == pfnMemImageCopy) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_mem_image_copy_params_t params = {&hQueue, &hImageSrc, &hImageDst, &srcOrigin, &dstOrigin, &region, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(104, "urEnqueueMemImageCopy", &params);

    ur_result_t result = pfnMemImageCopy(hQueue, hImageSrc, hImageDst, srcOrigin, dstOrigin, region, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(104, "urEnqueueMemImageCopy", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferMap
__urdlllocal ur_result_t UR_APICALL
urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object
    bool blockingMap,                         ///< [in] indicates blocking (true), non-blocking (false)
    ur_map_flags_t mapFlags,                  ///< [in] flags for read, write, readwrite mapping
    size_t offset,                            ///< [in] offset in bytes of the buffer region being mapped
    size_t size,                              ///< [in] size in bytes of the buffer region being mapped
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent,               ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
    void **ppRetMap                           ///< [in,out] return mapped pointer.  TODO: move it before
                                              ///< numEventsInWaitList?
) {
    auto pfnMemBufferMap = context.urDdiTable.Enqueue.pfnMemBufferMap;

    if (nullptr == pfnMemBufferMap) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_mem_buffer_map_params_t params = {&hQueue, &hBuffer, &blockingMap, &mapFlags, &offset, &size, &numEventsInWaitList, &phEventWaitList, &phEvent, &ppRetMap};
    uint64_t instance = context.notify_begin(105, "urEnqueueMemBufferMap", &params);

    ur_result_t result = pfnMemBufferMap(hQueue, hBuffer, blockingMap, mapFlags, offset, size, numEventsInWaitList, phEventWaitList, phEvent, ppRetMap);

    context.notify_end(105, "urEnqueueMemBufferMap", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemUnmap
__urdlllocal ur_result_t UR_APICALL
urEnqueueMemUnmap(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hMem,                     ///< [in] handle of the memory (buffer or image) object
    void *pMappedPtr,                         ///< [in] mapped host address
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnMemUnmap = context.urDdiTable.Enqueue.pfnMemUnmap;

    if (nullptr == pfnMemUnmap) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_mem_unmap_params_t params = {&hQueue, &hMem, &pMappedPtr, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(106, "urEnqueueMemUnmap", &params);

    ur_result_t result = pfnMemUnmap(hQueue, hMem, pMappedPtr, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(106, "urEnqueueMemUnmap", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMMemset
__urdlllocal ur_result_t UR_APICALL
urEnqueueUSMMemset(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    void *ptr,                                ///< [in] pointer to USM memory object
    int value,                                ///< [in] value to fill. It is interpreted as an 8-bit value and the upper
                                              ///< 24 bits are ignored
    size_t count,                             ///< [in] size in bytes to be set
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnUSMMemset = context.urDdiTable.Enqueue.pfnUSMMemset;

    if (nullptr == pfnUSMMemset) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_usm_memset_params_t params = {&hQueue, &ptr, &value, &count, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(107, "urEnqueueUSMMemset", &params);

    ur_result_t result = pfnUSMMemset(hQueue, ptr, value, count, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(107, "urEnqueueUSMMemset", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMMemcpy
__urdlllocal ur_result_t UR_APICALL
urEnqueueUSMMemcpy(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    bool blocking,                            ///< [in] blocking or non-blocking copy
    void *pDst,                               ///< [in] pointer to the destination USM memory object
    const void *pSrc,                         ///< [in] pointer to the source USM memory object
    size_t size,                              ///< [in] size in bytes to be copied
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnUSMMemcpy = context.urDdiTable.Enqueue.pfnUSMMemcpy;

    if (nullptr == pfnUSMMemcpy) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_usm_memcpy_params_t params = {&hQueue, &blocking, &pDst, &pSrc, &size, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(108, "urEnqueueUSMMemcpy", &params);

    ur_result_t result = pfnUSMMemcpy(hQueue, blocking, pDst, pSrc, size, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(108, "urEnqueueUSMMemcpy", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMPrefetch
__urdlllocal ur_result_t UR_APICALL
urEnqueueUSMPrefetch(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    const void *pMem,                         ///< [in] pointer to the USM memory object
    size_t size,                              ///< [in] size in bytes to be fetched
    ur_usm_migration_flags_t flags,           ///< [in] USM prefetch flags
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular command instance.
) {
    auto pfnUSMPrefetch = context.urDdiTable.Enqueue.pfnUSMPrefetch;

    if (nullptr == pfnUSMPrefetch) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_usm_prefetch_params_t params = {&hQueue, &pMem, &size, &flags, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(109, "urEnqueueUSMPrefetch", &params);

    ur_result_t result = pfnUSMPrefetch(hQueue, pMem, size, flags, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(109, "urEnqueueUSMPrefetch", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMMemAdvise
__urdlllocal ur_result_t UR_APICALL
urEnqueueUSMMemAdvise(
    ur_queue_handle_t hQueue,  ///< [in] handle of the queue object
    const void *pMem,          ///< [in] pointer to the USM memory object
    size_t size,               ///< [in] size in bytes to be advised
    ur_mem_advice_t advice,    ///< [in] USM memory advice
    ur_event_handle_t *phEvent ///< [in,out][optional] return an event object that identifies this
                               ///< particular command instance.
) {
    auto pfnUSMMemAdvise = context.urDdiTable.Enqueue.pfnUSMMemAdvise;

    if (nullptr == pfnUSMMemAdvise) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_usm_mem_advise_params_t params = {&hQueue, &pMem, &size, &advice, &phEvent};
    uint64_t instance = context.notify_begin(110, "urEnqueueUSMMemAdvise", &params);

    ur_result_t result = pfnUSMMemAdvise(hQueue, pMem, size, advice, phEvent);

    context.notify_end(110, "urEnqueueUSMMemAdvise", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMFill2D
__urdlllocal ur_result_t UR_APICALL
urEnqueueUSMFill2D(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue to submit to.
    void *pMem,                               ///< [in] pointer to memory to be filled.
    size_t pitch,                             ///< [in] the total width of the destination memory including padding.
    size_t patternSize,                       ///< [in] the size in bytes of the pattern.
    const void *pPattern,                     ///< [in] pointer with the bytes of the pattern to set.
    size_t width,                             ///< [in] the width in bytes of each row to fill.
    size_t height,                            ///< [in] the height of the columns to fill.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular kernel execution instance.
) {
    auto pfnUSMFill2D = context.urDdiTable.Enqueue.pfnUSMFill2D;

    if (nullptr == pfnUSMFill2D) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_usm_fill2_d_params_t params = {&hQueue, &pMem, &pitch, &patternSize, &pPattern, &width, &height, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(111, "urEnqueueUSMFill2D", &params);

    ur_result_t result = pfnUSMFill2D(hQueue, pMem, pitch, patternSize, pPattern, width, height, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(111, "urEnqueueUSMFill2D", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMMemset2D
__urdlllocal ur_result_t UR_APICALL
urEnqueueUSMMemset2D(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue to submit to.
    void *pMem,                               ///< [in] pointer to memory to be filled.
    size_t pitch,                             ///< [in] the total width of the destination memory including padding.
    int value,                                ///< [in] the value to fill into the region in pMem. It is interpreted as
                                              ///< an 8-bit value and the upper 24 bits are ignored
    size_t width,                             ///< [in] the width in bytes of each row to set.
    size_t height,                            ///< [in] the height of the columns to set.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular kernel execution instance.
) {
    auto pfnUSMMemset2D = context.urDdiTable.Enqueue.pfnUSMMemset2D;

    if (nullptr == pfnUSMMemset2D) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_usm_memset2_d_params_t params = {&hQueue, &pMem, &pitch, &value, &width, &height, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(112, "urEnqueueUSMMemset2D", &params);

    ur_result_t result = pfnUSMMemset2D(hQueue, pMem, pitch, value, width, height, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(112, "urEnqueueUSMMemset2D", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMMemcpy2D
__urdlllocal ur_result_t UR_APICALL
urEnqueueUSMMemcpy2D(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue to submit to.
    bool blocking,                            ///< [in] indicates if this operation should block the host.
    void *pDst,                               ///< [in] pointer to memory where data will be copied.
    size_t dstPitch,                          ///< [in] the total width of the source memory including padding.
    const void *pSrc,                         ///< [in] pointer to memory to be copied.
    size_t srcPitch,                          ///< [in] the total width of the source memory including padding.
    size_t width,                             ///< [in] the width in bytes of each row to be copied.
    size_t height,                            ///< [in] the height of columns to be copied.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular kernel execution instance.
) {
    auto pfnUSMMemcpy2D = context.urDdiTable.Enqueue.pfnUSMMemcpy2D;

    if (nullptr == pfnUSMMemcpy2D) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_usm_memcpy2_d_params_t params = {&hQueue, &blocking, &pDst, &dstPitch, &pSrc, &srcPitch, &width, &height, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(113, "urEnqueueUSMMemcpy2D", &params);

    ur_result_t result = pfnUSMMemcpy2D(hQueue, blocking, pDst, dstPitch, pSrc, srcPitch, width, height, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(113, "urEnqueueUSMMemcpy2D", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueDeviceGlobalVariableWrite
__urdlllocal ur_result_t UR_APICALL
urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue to submit to.
    ur_program_handle_t hProgram,             ///< [in] handle of the program containing the device global variable.
    const char *name,                         ///< [in] the unique identifier for the device global variable.
    bool blockingWrite,                       ///< [in] indicates if this operation should block.
    size_t count,                             ///< [in] the number of bytes to copy.
    size_t offset,                            ///< [in] the byte offset into the device global variable to start copying.
    const void *pSrc,                         ///< [in] pointer to where the data must be copied from.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list.
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular kernel execution instance.
) {
    auto pfnDeviceGlobalVariableWrite = context.urDdiTable.Enqueue.pfnDeviceGlobalVariableWrite;

    if (nullptr == pfnDeviceGlobalVariableWrite) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_device_global_variable_write_params_t params = {&hQueue, &hProgram, &name, &blockingWrite, &count, &offset, &pSrc, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(114, "urEnqueueDeviceGlobalVariableWrite", &params);

    ur_result_t result = pfnDeviceGlobalVariableWrite(hQueue, hProgram, name, blockingWrite, count, offset, pSrc, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(114, "urEnqueueDeviceGlobalVariableWrite", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueDeviceGlobalVariableRead
__urdlllocal ur_result_t UR_APICALL
urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue to submit to.
    ur_program_handle_t hProgram,             ///< [in] handle of the program containing the device global variable.
    const char *name,                         ///< [in] the unique identifier for the device global variable.
    bool blockingRead,                        ///< [in] indicates if this operation should block.
    size_t count,                             ///< [in] the number of bytes to copy.
    size_t offset,                            ///< [in] the byte offset into the device global variable to start copying.
    void *pDst,                               ///< [in] pointer to where the data must be copied to.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list.
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [in,out][optional] return an event object that identifies this
                                              ///< particular kernel execution instance.
) {
    auto pfnDeviceGlobalVariableRead = context.urDdiTable.Enqueue.pfnDeviceGlobalVariableRead;

    if (nullptr == pfnDeviceGlobalVariableRead) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_enqueue_device_global_variable_read_params_t params = {&hQueue, &hProgram, &name, &blockingRead, &count, &offset, &pDst, &numEventsInWaitList, &phEventWaitList, &phEvent};
    uint64_t instance = context.notify_begin(115, "urEnqueueDeviceGlobalVariableRead", &params);

    ur_result_t result = pfnDeviceGlobalVariableRead(hQueue, hProgram, name, blockingRead, count, offset, pDst, numEventsInWaitList, phEventWaitList, phEvent);

    context.notify_end(115, "urEnqueueDeviceGlobalVariableRead", &params, &result, instance);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Global table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL
urGetGlobalProcAddrTable(
    ur_api_version_t version,       ///< [in] API version requested
    ur_global_dditable_t *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = tracing_layer::context.urDdiTable.Global;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(tracing_layer::context.version) != UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(tracing_layer::context.version) > UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnInit = pDdiTable->pfnInit;
    pDdiTable->pfnInit = tracing_layer::urInit;

    dditable.pfnGetLastResult = pDdiTable->pfnGetLastResult;
    pDdiTable->pfnGetLastResult = tracing_layer::urGetLastResult;

    dditable.pfnTearDown = pDdiTable->pfnTearDown;
    pDdiTable->pfnTearDown = tracing_layer::urTearDown;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Context table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL
urGetContextProcAddrTable(
    ur_api_version_t version,        ///< [in] API version requested
    ur_context_dditable_t *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = tracing_layer::context.urDdiTable.Context;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(tracing_layer::context.version) != UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(tracing_layer::context.version) > UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = tracing_layer::urContextCreate;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = tracing_layer::urContextRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = tracing_layer::urContextRelease;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = tracing_layer::urContextGetInfo;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle = tracing_layer::urContextGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle = tracing_layer::urContextCreateWithNativeHandle;

    dditable.pfnSetExtendedDeleter = pDdiTable->pfnSetExtendedDeleter;
    pDdiTable->pfnSetExtendedDeleter = tracing_layer::urContextSetExtendedDeleter;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Enqueue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL
urGetEnqueueProcAddrTable(
    ur_api_version_t version,        ///< [in] API version requested
    ur_enqueue_dditable_t *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = tracing_layer::context.urDdiTable.Enqueue;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(tracing_layer::context.version) != UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(tracing_layer::context.version) > UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnKernelLaunch = pDdiTable->pfnKernelLaunch;
    pDdiTable->pfnKernelLaunch = tracing_layer::urEnqueueKernelLaunch;

    dditable.pfnEventsWait = pDdiTable->pfnEventsWait;
    pDdiTable->pfnEventsWait = tracing_layer::urEnqueueEventsWait;

    dditable.pfnEventsWaitWithBarrier = pDdiTable->pfnEventsWaitWithBarrier;
    pDdiTable->pfnEventsWaitWithBarrier = tracing_layer::urEnqueueEventsWaitWithBarrier;

    dditable.pfnMemBufferRead = pDdiTable->pfnMemBufferRead;
    pDdiTable->pfnMemBufferRead = tracing_layer::urEnqueueMemBufferRead;

    dditable.pfnMemBufferWrite = pDdiTable->pfnMemBufferWrite;
    pDdiTable->pfnMemBufferWrite = tracing_layer::urEnqueueMemBufferWrite;

    dditable.pfnMemBufferReadRect = pDdiTable->pfnMemBufferReadRect;
    pDdiTable->pfnMemBufferReadRect = tracing_layer::urEnqueueMemBufferReadRect;

    dditable.pfnMemBufferWriteRect = pDdiTable->pfnMemBufferWriteRect;
    pDdiTable->pfnMemBufferWriteRect = tracing_layer::urEnqueueMemBufferWriteRect;

    dditable.pfnMemBufferCopy = pDdiTable->pfnMemBufferCopy;
    pDdiTable->pfnMemBufferCopy = tracing_layer::urEnqueueMemBufferCopy;

    dditable.pfnMemBufferCopyRect = pDdiTable->pfnMemBufferCopyRect;
    pDdiTable->pfnMemBufferCopyRect = tracing_layer::urEnqueueMemBufferCopyRect;

    dditable.pfnMemBufferFill = pDdiTable->pfnMemBufferFill;
    pDdiTable->pfnMemBufferFill = tracing_layer::urEnqueueMemBufferFill;

    dditable.pfnMemImageRead = pDdiTable->pfnMemImageRead;
    pDdiTable->pfnMemImageRead = tracing_layer::urEnqueueMemImageRead;

    dditable.pfnMemImageWrite = pDdiTable->pfnMemImageWrite;
    pDdiTable->pfnMemImageWrite = tracing_layer::urEnqueueMemImageWrite;

    dditable.pfnMemImageCopy = pDdiTable->pfnMemImageCopy;
    pDdiTable->pfnMemImageCopy = tracing_layer::urEnqueueMemImageCopy;

    dditable.pfnMemBufferMap = pDdiTable->pfnMemBufferMap;
    pDdiTable->pfnMemBufferMap = tracing_layer::urEnqueueMemBufferMap;

    dditable.pfnMemUnmap = pDdiTable->pfnMemUnmap;
    pDdiTable->pfnMemUnmap = tracing_layer::urEnqueueMemUnmap;

    dditable.pfnUSMMemset = pDdiTable->pfnUSMMemset;
    pDdiTable->pfnUSMMemset = tracing_layer::urEnqueueUSMMemset;

    dditable.pfnUSMMemcpy = pDdiTable->pfnUSMMemcpy;
    pDdiTable->pfnUSMMemcpy = tracing_layer::urEnqueueUSMMemcpy;

    dditable.pfnUSMPrefetch = pDdiTable->pfnUSMPrefetch;
    pDdiTable->pfnUSMPrefetch = tracing_layer::urEnqueueUSMPrefetch;

    dditable.pfnUSMMemAdvise = pDdiTable->pfnUSMMemAdvise;
    pDdiTable->pfnUSMMemAdvise = tracing_layer::urEnqueueUSMMemAdvise;

    dditable.pfnUSMFill2D = pDdiTable->pfnUSMFill2D;
    pDdiTable->pfnUSMFill2D = tracing_layer::urEnqueueUSMFill2D;

    dditable.pfnUSMMemset2D = pDdiTable->pfnUSMMemset2D;
    pDdiTable->pfnUSMMemset2D = tracing_layer::urEnqueueUSMMemset2D;

    dditable.pfnUSMMemcpy2D = pDdiTable->pfnUSMMemcpy2D;
    pDdiTable->pfnUSMMemcpy2D = tracing_layer::urEnqueueUSMMemcpy2D;

    dditable.pfnDeviceGlobalVariableWrite = pDdiTable->pfnDeviceGlobalVariableWrite;
    pDdiTable->pfnDeviceGlobalVariableWrite = tracing_layer::urEnqueueDeviceGlobalVariableWrite;

    dditable.pfnDeviceGlobalVariableRead = pDdiTable->pfnDeviceGlobalVariableRead;
    pDdiTable->pfnDeviceGlobalVariableRead = tracing_layer::urEnqueueDeviceGlobalVariableRead;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Event table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL
urGetEventProcAddrTable(
    ur_api_version_t version,      ///< [in] API version requested
    ur_event_dditable_t *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = tracing_layer::context.urDdiTable.Event;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(tracing_layer::context.version) != UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(tracing_layer::context.version) > UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = tracing_layer::urEventGetInfo;

    dditable.pfnGetProfilingInfo = pDdiTable->pfnGetProfilingInfo;
    pDdiTable->pfnGetProfilingInfo = tracing_layer::urEventGetProfilingInfo;

    dditable.pfnWait = pDdiTable->pfnWait;
    pDdiTable->pfnWait = tracing_layer::urEventWait;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = tracing_layer::urEventRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = tracing_layer::urEventRelease;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle = tracing_layer::urEventGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle = tracing_layer::urEventCreateWithNativeHandle;

    dditable.pfnSetCallback = pDdiTable->pfnSetCallback;
    pDdiTable->pfnSetCallback = tracing_layer::urEventSetCallback;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Kernel table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL
urGetKernelProcAddrTable(
    ur_api_version_t version,       ///< [in] API version requested
    ur_kernel_dditable_t *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = tracing_layer::context.urDdiTable.Kernel;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(tracing_layer::context.version) != UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(tracing_layer::context.version) > UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = tracing_layer::urKernelCreate;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = tracing_layer::urKernelGetInfo;

    dditable.pfnGetGroupInfo = pDdiTable->pfnGetGroupInfo;
    pDdiTable->pfnGetGroupInfo = tracing_layer::urKernelGetGroupInfo;

    dditable.pfnGetSubGroupInfo = pDdiTable->pfnGetSubGroupInfo;
    pDdiTable->pfnGetSubGroupInfo = tracing_layer::urKernelGetSubGroupInfo;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = tracing_layer::urKernelRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = tracing_layer::urKernelRelease;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle = tracing_layer::urKernelGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle = tracing_layer::urKernelCreateWithNativeHandle;

    dditable.pfnSetArgValue = pDdiTable->pfnSetArgValue;
    pDdiTable->pfnSetArgValue = tracing_layer::urKernelSetArgValue;

    dditable.pfnSetArgLocal = pDdiTable->pfnSetArgLocal;
    pDdiTable->pfnSetArgLocal = tracing_layer::urKernelSetArgLocal;

    dditable.pfnSetArgPointer = pDdiTable->pfnSetArgPointer;
    pDdiTable->pfnSetArgPointer = tracing_layer::urKernelSetArgPointer;

    dditable.pfnSetExecInfo = pDdiTable->pfnSetExecInfo;
    pDdiTable->pfnSetExecInfo = tracing_layer::urKernelSetExecInfo;

    dditable.pfnSetArgSampler = pDdiTable->pfnSetArgSampler;
    pDdiTable->pfnSetArgSampler = tracing_layer::urKernelSetArgSampler;

    dditable.pfnSetArgMemObj = pDdiTable->pfnSetArgMemObj;
    pDdiTable->pfnSetArgMemObj = tracing_layer::urKernelSetArgMemObj;

    dditable.pfnSetSpecializationConstants = pDdiTable->pfnSetSpecializationConstants;
    pDdiTable->pfnSetSpecializationConstants = tracing_layer::urKernelSetSpecializationConstants;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Mem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL
urGetMemProcAddrTable(
    ur_api_version_t version,    ///< [in] API version requested
    ur_mem_dditable_t *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = tracing_layer::context.urDdiTable.Mem;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(tracing_layer::context.version) != UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(tracing_layer::context.version) > UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnImageCreate = pDdiTable->pfnImageCreate;
    pDdiTable->pfnImageCreate = tracing_layer::urMemImageCreate;

    dditable.pfnBufferCreate = pDdiTable->pfnBufferCreate;
    pDdiTable->pfnBufferCreate = tracing_layer::urMemBufferCreate;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = tracing_layer::urMemRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = tracing_layer::urMemRelease;

    dditable.pfnBufferPartition = pDdiTable->pfnBufferPartition;
    pDdiTable->pfnBufferPartition = tracing_layer::urMemBufferPartition;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle = tracing_layer::urMemGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle = tracing_layer::urMemCreateWithNativeHandle;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = tracing_layer::urMemGetInfo;

    dditable.pfnImageGetInfo = pDdiTable->pfnImageGetInfo;
    pDdiTable->pfnImageGetInfo = tracing_layer::urMemImageGetInfo;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Module table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL
urGetModuleProcAddrTable(
    ur_api_version_t version,       ///< [in] API version requested
    ur_module_dditable_t *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = tracing_layer::context.urDdiTable.Module;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(tracing_layer::context.version) != UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(tracing_layer::context.version) > UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = tracing_layer::urModuleCreate;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = tracing_layer::urModuleRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = tracing_layer::urModuleRelease;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle = tracing_layer::urModuleGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle = tracing_layer::urModuleCreateWithNativeHandle;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Platform table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL
urGetPlatformProcAddrTable(
    ur_api_version_t version,         ///< [in] API version requested
    ur_platform_dditable_t *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = tracing_layer::context.urDdiTable.Platform;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(tracing_layer::context.version) != UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(tracing_layer::context.version) > UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnGet = pDdiTable->pfnGet;
    pDdiTable->pfnGet = tracing_layer::urPlatformGet;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = tracing_layer::urPlatformGetInfo;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle = tracing_layer::urPlatformGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle = tracing_layer::urPlatformCreateWithNativeHandle;

    dditable.pfnGetApiVersion = pDdiTable->pfnGetApiVersion;
    pDdiTable->pfnGetApiVersion = tracing_layer::urPlatformGetApiVersion;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Program table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL
urGetProgramProcAddrTable(
    ur_api_version_t version,        ///< [in] API version requested
    ur_program_dditable_t *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = tracing_layer::context.urDdiTable.Program;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(tracing_layer::context.version) != UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(tracing_layer::context.version) > UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = tracing_layer::urProgramCreate;

    dditable.pfnCreateWithBinary = pDdiTable->pfnCreateWithBinary;
    pDdiTable->pfnCreateWithBinary = tracing_layer::urProgramCreateWithBinary;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = tracing_layer::urProgramRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = tracing_layer::urProgramRelease;

    dditable.pfnGetFunctionPointer = pDdiTable->pfnGetFunctionPointer;
    pDdiTable->pfnGetFunctionPointer = tracing_layer::urProgramGetFunctionPointer;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = tracing_layer::urProgramGetInfo;

    dditable.pfnGetBuildInfo = pDdiTable->pfnGetBuildInfo;
    pDdiTable->pfnGetBuildInfo = tracing_layer::urProgramGetBuildInfo;

    dditable.pfnSetSpecializationConstants = pDdiTable->pfnSetSpecializationConstants;
    pDdiTable->pfnSetSpecializationConstants = tracing_layer::urProgramSetSpecializationConstants;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle = tracing_layer::urProgramGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle = tracing_layer::urProgramCreateWithNativeHandle;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Queue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL
urGetQueueProcAddrTable(
    ur_api_version_t version,      ///< [in] API version requested
    ur_queue_dditable_t *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = tracing_layer::context.urDdiTable.Queue;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(tracing_layer::context.version) != UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(tracing_layer::context.version) > UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = tracing_layer::urQueueGetInfo;

    dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = tracing_layer::urQueueCreate;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = tracing_layer::urQueueRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = tracing_layer::urQueueRelease;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle = tracing_layer::urQueueGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle = tracing_layer::urQueueCreateWithNativeHandle;

    dditable.pfnFinish = pDdiTable->pfnFinish;
    pDdiTable->pfnFinish = tracing_layer::urQueueFinish;

    dditable.pfnFlush = pDdiTable->pfnFlush;
    pDdiTable->pfnFlush = tracing_layer::urQueueFlush;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Sampler table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL
urGetSamplerProcAddrTable(
    ur_api_version_t version,        ///< [in] API version requested
    ur_sampler_dditable_t *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = tracing_layer::context.urDdiTable.Sampler;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(tracing_layer::context.version) != UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(tracing_layer::context.version) > UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = tracing_layer::urSamplerCreate;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = tracing_layer::urSamplerRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = tracing_layer::urSamplerRelease;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = tracing_layer::urSamplerGetInfo;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle = tracing_layer::urSamplerGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle = tracing_layer::urSamplerCreateWithNativeHandle;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USM table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL
urGetUSMProcAddrTable(
    ur_api_version_t version,    ///< [in] API version requested
    ur_usm_dditable_t *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = tracing_layer::context.urDdiTable.USM;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(tracing_layer::context.version) != UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(tracing_layer::context.version) > UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnHostAlloc = pDdiTable->pfnHostAlloc;
    pDdiTable->pfnHostAlloc = tracing_layer::urUSMHostAlloc;

    dditable.pfnDeviceAlloc = pDdiTable->pfnDeviceAlloc;
    pDdiTable->pfnDeviceAlloc = tracing_layer::urUSMDeviceAlloc;

    dditable.pfnSharedAlloc = pDdiTable->pfnSharedAlloc;
    pDdiTable->pfnSharedAlloc = tracing_layer::urUSMSharedAlloc;

    dditable.pfnFree = pDdiTable->pfnFree;
    pDdiTable->pfnFree = tracing_layer::urUSMFree;

    dditable.pfnGetMemAllocInfo = pDdiTable->pfnGetMemAllocInfo;
    pDdiTable->pfnGetMemAllocInfo = tracing_layer::urUSMGetMemAllocInfo;

    dditable.pfnPoolCreate = pDdiTable->pfnPoolCreate;
    pDdiTable->pfnPoolCreate = tracing_layer::urUSMPoolCreate;

    dditable.pfnPoolDestroy = pDdiTable->pfnPoolDestroy;
    pDdiTable->pfnPoolDestroy = tracing_layer::urUSMPoolDestroy;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Device table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL
urGetDeviceProcAddrTable(
    ur_api_version_t version,       ///< [in] API version requested
    ur_device_dditable_t *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = tracing_layer::context.urDdiTable.Device;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(tracing_layer::context.version) != UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(tracing_layer::context.version) > UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnGet = pDdiTable->pfnGet;
    pDdiTable->pfnGet = tracing_layer::urDeviceGet;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = tracing_layer::urDeviceGetInfo;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = tracing_layer::urDeviceRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = tracing_layer::urDeviceRelease;

    dditable.pfnPartition = pDdiTable->pfnPartition;
    pDdiTable->pfnPartition = tracing_layer::urDevicePartition;

    dditable.pfnSelectBinary = pDdiTable->pfnSelectBinary;
    pDdiTable->pfnSelectBinary = tracing_layer::urDeviceSelectBinary;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle = tracing_layer::urDeviceGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle = tracing_layer::urDeviceCreateWithNativeHandle;

    dditable.pfnGetGlobalTimestamps = pDdiTable->pfnGetGlobalTimestamps;
    pDdiTable->pfnGetGlobalTimestamps = tracing_layer::urDeviceGetGlobalTimestamps;

    return result;
}

ur_result_t context_t::init(ur_dditable_t *dditable) {
    ur_result_t result = UR_RESULT_SUCCESS;

    if (UR_RESULT_SUCCESS == result) {
        result = tracing_layer::urGetGlobalProcAddrTable(UR_API_VERSION_0_9, &dditable->Global);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = tracing_layer::urGetContextProcAddrTable(UR_API_VERSION_0_9, &dditable->Context);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = tracing_layer::urGetEnqueueProcAddrTable(UR_API_VERSION_0_9, &dditable->Enqueue);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = tracing_layer::urGetEventProcAddrTable(UR_API_VERSION_0_9, &dditable->Event);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = tracing_layer::urGetKernelProcAddrTable(UR_API_VERSION_0_9, &dditable->Kernel);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = tracing_layer::urGetMemProcAddrTable(UR_API_VERSION_0_9, &dditable->Mem);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = tracing_layer::urGetModuleProcAddrTable(UR_API_VERSION_0_9, &dditable->Module);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = tracing_layer::urGetPlatformProcAddrTable(UR_API_VERSION_0_9, &dditable->Platform);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = tracing_layer::urGetProgramProcAddrTable(UR_API_VERSION_0_9, &dditable->Program);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = tracing_layer::urGetQueueProcAddrTable(UR_API_VERSION_0_9, &dditable->Queue);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = tracing_layer::urGetSamplerProcAddrTable(UR_API_VERSION_0_9, &dditable->Sampler);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = tracing_layer::urGetUSMProcAddrTable(UR_API_VERSION_0_9, &dditable->USM);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = tracing_layer::urGetDeviceProcAddrTable(UR_API_VERSION_0_9, &dditable->Device);
    }

    return result;
}
} /* namespace tracing_layer */
