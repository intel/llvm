/*
 *
 * Copyright (C) 2023-2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_valddi.cpp
 *
 */
#include "ur_leak_check.hpp"
#include "ur_validation_layer.hpp"

namespace ur_validation_layer {

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterGet
__urdlllocal ur_result_t UR_APICALL urAdapterGet(
    uint32_t
        NumEntries, ///< [in] the number of adapters to be added to phAdapters.
    ///< If phAdapters is not NULL, then NumEntries should be greater than
    ///< zero, otherwise ::UR_RESULT_ERROR_INVALID_SIZE,
    ///< will be returned.
    ur_adapter_handle_t *
        phAdapters, ///< [out][optional][range(0, NumEntries)] array of handle of adapters.
    ///< If NumEntries is less than the number of adapters available, then
    ///< ::urAdapterGet shall only retrieve that number of platforms.
    uint32_t *
        pNumAdapters ///< [out][optional] returns the total number of adapters available.
) {
    auto pfnAdapterGet = context.urDdiTable.Global.pfnAdapterGet;

    if (nullptr == pfnAdapterGet) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
    }

    ur_result_t result = pfnAdapterGet(NumEntries, phAdapters, pNumAdapters);

    if (context.enableLeakChecking && phAdapters &&
        result == UR_RESULT_SUCCESS) {
        for (uint32_t i = 0; i < NumEntries; i++) {
            refCountContext.createOrIncrementRefCount(phAdapters[i], true);
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterRelease
__urdlllocal ur_result_t UR_APICALL urAdapterRelease(
    ur_adapter_handle_t hAdapter ///< [in] Adapter handle to release
) {
    auto pfnAdapterRelease = context.urDdiTable.Global.pfnAdapterRelease;

    if (nullptr == pfnAdapterRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hAdapter) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnAdapterRelease(hAdapter);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.decrementRefCount(hAdapter, true);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterRetain
__urdlllocal ur_result_t UR_APICALL urAdapterRetain(
    ur_adapter_handle_t hAdapter ///< [in] Adapter handle to retain
) {
    auto pfnAdapterRetain = context.urDdiTable.Global.pfnAdapterRetain;

    if (nullptr == pfnAdapterRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hAdapter) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnAdapterRetain(hAdapter);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.incrementRefCount(hAdapter, true);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterGetLastError
__urdlllocal ur_result_t UR_APICALL urAdapterGetLastError(
    ur_adapter_handle_t hAdapter, ///< [in] handle of the adapter instance
    const char **
        ppMessage, ///< [out] pointer to a C string where the adapter specific error message
                   ///< will be stored.
    int32_t *
        pError ///< [out] pointer to an integer where the adapter specific error code will
               ///< be stored.
) {
    auto pfnAdapterGetLastError =
        context.urDdiTable.Global.pfnAdapterGetLastError;

    if (nullptr == pfnAdapterGetLastError) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hAdapter) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == ppMessage) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pError) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hAdapter)) {
        refCountContext.logInvalidReference(hAdapter);
    }

    ur_result_t result = pfnAdapterGetLastError(hAdapter, ppMessage, pError);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterGetInfo
__urdlllocal ur_result_t UR_APICALL urAdapterGetInfo(
    ur_adapter_handle_t hAdapter, ///< [in] handle of the adapter
    ur_adapter_info_t propName,   ///< [in] type of the info to retrieve
    size_t propSize, ///< [in] the number of bytes pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If Size is not equal to or greater to the real number of bytes needed
    ///< to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    ///< returned and pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual number of bytes being queried by pPropValue.
) {
    auto pfnAdapterGetInfo = context.urDdiTable.Global.pfnAdapterGetInfo;

    if (nullptr == pfnAdapterGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hAdapter) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_ADAPTER_INFO_REFERENCE_COUNT < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hAdapter)) {
        refCountContext.logInvalidReference(hAdapter);
    }

    ur_result_t result = pfnAdapterGetInfo(hAdapter, propName, propSize,
                                           pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGet
__urdlllocal ur_result_t UR_APICALL urPlatformGet(
    ur_adapter_handle_t *
        phAdapters, ///< [in][range(0, NumAdapters)] array of adapters to query for platforms.
    uint32_t NumAdapters, ///< [in] number of adapters pointed to by phAdapters
    uint32_t
        NumEntries, ///< [in] the number of platforms to be added to phPlatforms.
    ///< If phPlatforms is not NULL, then NumEntries should be greater than
    ///< zero, otherwise ::UR_RESULT_ERROR_INVALID_SIZE,
    ///< will be returned.
    ur_platform_handle_t *
        phPlatforms, ///< [out][optional][range(0, NumEntries)] array of handle of platforms.
    ///< If NumEntries is less than the number of platforms available, then
    ///< ::urPlatformGet shall only retrieve that number of platforms.
    uint32_t *
        pNumPlatforms ///< [out][optional] returns the total number of platforms available.
) {
    auto pfnGet = context.urDdiTable.Platform.pfnGet;

    if (nullptr == pfnGet) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == phAdapters) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NumEntries == 0 && phPlatforms != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    ur_result_t result =
        pfnGet(phAdapters, NumAdapters, NumEntries, phPlatforms, pNumPlatforms);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetInfo
__urdlllocal ur_result_t UR_APICALL urPlatformGetInfo(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform
    ur_platform_info_t propName,    ///< [in] type of the info to retrieve
    size_t propSize, ///< [in] the number of bytes pointed to by pPlatformInfo.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If Size is not equal to or greater to the real number of bytes needed
    ///< to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    ///< returned and pPlatformInfo is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual number of bytes being queried by pPlatformInfo.
) {
    auto pfnGetInfo = context.urDdiTable.Platform.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hPlatform) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_PLATFORM_INFO_BACKEND < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    ur_result_t result =
        pfnGetInfo(hPlatform, propName, propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetApiVersion
__urdlllocal ur_result_t UR_APICALL urPlatformGetApiVersion(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform
    ur_api_version_t *pVersion      ///< [out] api version
) {
    auto pfnGetApiVersion = context.urDdiTable.Platform.pfnGetApiVersion;

    if (nullptr == pfnGetApiVersion) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hPlatform) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pVersion) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    ur_result_t result = pfnGetApiVersion(hPlatform, pVersion);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urPlatformGetNativeHandle(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform.
    ur_native_handle_t *
        phNativePlatform ///< [out] a pointer to the native handle of the platform.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Platform.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hPlatform) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phNativePlatform) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    ur_result_t result = pfnGetNativeHandle(hPlatform, phNativePlatform);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urPlatformCreateWithNativeHandle(
    ur_native_handle_t
        hNativePlatform, ///< [in][nocheck] the native handle of the platform.
    const ur_platform_native_properties_t *
        pProperties, ///< [in][optional] pointer to native platform properties struct.
    ur_platform_handle_t *
        phPlatform ///< [out] pointer to the handle of the platform object created.
) {
    auto pfnCreateWithNativeHandle =
        context.urDdiTable.Platform.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == phPlatform) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    ur_result_t result =
        pfnCreateWithNativeHandle(hNativePlatform, pProperties, phPlatform);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetBackendOption
__urdlllocal ur_result_t UR_APICALL urPlatformGetBackendOption(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform instance.
    const char
        *pFrontendOption, ///< [in] string containing the frontend option.
    const char **
        ppPlatformOption ///< [out] returns the correct platform specific compiler option based on
                         ///< the frontend option.
) {
    auto pfnGetBackendOption = context.urDdiTable.Platform.pfnGetBackendOption;

    if (nullptr == pfnGetBackendOption) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hPlatform) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pFrontendOption) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == ppPlatformOption) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    ur_result_t result =
        pfnGetBackendOption(hPlatform, pFrontendOption, ppPlatformOption);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGet
__urdlllocal ur_result_t UR_APICALL urDeviceGet(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform instance
    ur_device_type_t DeviceType,    ///< [in] the type of the devices.
    uint32_t
        NumEntries, ///< [in] the number of devices to be added to phDevices.
    ///< If phDevices is not NULL, then NumEntries should be greater than zero.
    ///< Otherwise ::UR_RESULT_ERROR_INVALID_SIZE
    ///< will be returned.
    ur_device_handle_t *
        phDevices, ///< [out][optional][range(0, NumEntries)] array of handle of devices.
    ///< If NumEntries is less than the number of devices available, then
    ///< platform shall only retrieve that number of devices.
    uint32_t *pNumDevices ///< [out][optional] pointer to the number of devices.
    ///< pNumDevices will be updated with the total number of devices available.
) {
    auto pfnGet = context.urDdiTable.Device.pfnGet;

    if (nullptr == pfnGet) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hPlatform) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NumEntries > 0 && phDevices == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_DEVICE_TYPE_VPU < DeviceType) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (NumEntries == 0 && phDevices != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    ur_result_t result =
        pfnGet(hPlatform, DeviceType, NumEntries, phDevices, pNumDevices);

    if (context.enableLeakChecking && phDevices &&
        result == UR_RESULT_SUCCESS) {
        for (uint32_t i = 0; i < NumEntries; i++) {
            refCountContext.createOrIncrementRefCount(phDevices[i], false);
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGetInfo
__urdlllocal ur_result_t UR_APICALL urDeviceGetInfo(
    ur_device_handle_t hDevice, ///< [in] handle of the device instance
    ur_device_info_t propName,  ///< [in] type of the info to retrieve
    size_t propSize, ///< [in] the number of bytes pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If propSize is not equal to or greater than the real number of bytes
    ///< needed to return the info
    ///< then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
) {
    auto pfnGetInfo = context.urDdiTable.Device.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_DEVICE_INFO_INTEROP_SEMAPHORE_EXPORT_SUPPORT_EXP < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result =
        pfnGetInfo(hDevice, propName, propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRetain
__urdlllocal ur_result_t UR_APICALL urDeviceRetain(
    ur_device_handle_t
        hDevice ///< [in] handle of the device to get a reference of.
) {
    auto pfnRetain = context.urDdiTable.Device.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRetain(hDevice);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.incrementRefCount(hDevice, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRelease
__urdlllocal ur_result_t UR_APICALL urDeviceRelease(
    ur_device_handle_t hDevice ///< [in] handle of the device to release.
) {
    auto pfnRelease = context.urDdiTable.Device.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRelease(hDevice);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.decrementRefCount(hDevice, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDevicePartition
__urdlllocal ur_result_t UR_APICALL urDevicePartition(
    ur_device_handle_t hDevice, ///< [in] handle of the device to partition.
    const ur_device_partition_properties_t
        *pProperties,    ///< [in] Device partition properties.
    uint32_t NumDevices, ///< [in] the number of sub-devices.
    ur_device_handle_t *
        phSubDevices, ///< [out][optional][range(0, NumDevices)] array of handle of devices.
    ///< If NumDevices is less than the number of sub-devices available, then
    ///< the function shall only retrieve that number of sub-devices.
    uint32_t *
        pNumDevicesRet ///< [out][optional] pointer to the number of sub-devices the device can be
    ///< partitioned into according to the partitioning property.
) {
    auto pfnPartition = context.urDdiTable.Device.pfnPartition;

    if (nullptr == pfnPartition) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pProperties) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pProperties->pProperties) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnPartition(hDevice, pProperties, NumDevices,
                                      phSubDevices, pNumDevicesRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceSelectBinary
__urdlllocal ur_result_t UR_APICALL urDeviceSelectBinary(
    ur_device_handle_t
        hDevice, ///< [in] handle of the device to select binary for.
    const ur_device_binary_t
        *pBinaries,       ///< [in] the array of binaries to select from.
    uint32_t NumBinaries, ///< [in] the number of binaries passed in ppBinaries.
                          ///< Must greater than or equal to zero otherwise
                          ///< ::UR_RESULT_ERROR_INVALID_VALUE is returned.
    uint32_t *
        pSelectedBinary ///< [out] the index of the selected binary in the input array of binaries.
    ///< If a suitable binary was not found the function returns ::UR_RESULT_ERROR_INVALID_BINARY.
) {
    auto pfnSelectBinary = context.urDdiTable.Device.pfnSelectBinary;

    if (nullptr == pfnSelectBinary) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pBinaries) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pSelectedBinary) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NumBinaries == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result =
        pfnSelectBinary(hDevice, pBinaries, NumBinaries, pSelectedBinary);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urDeviceGetNativeHandle(
    ur_device_handle_t hDevice, ///< [in] handle of the device.
    ur_native_handle_t
        *phNativeDevice ///< [out] a pointer to the native handle of the device.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Device.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phNativeDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnGetNativeHandle(hDevice, phNativeDevice);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urDeviceCreateWithNativeHandle(
    ur_native_handle_t
        hNativeDevice, ///< [in][nocheck] the native handle of the device.
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform instance
    const ur_device_native_properties_t *
        pProperties, ///< [in][optional] pointer to native device properties struct.
    ur_device_handle_t
        *phDevice ///< [out] pointer to the handle of the device object created.
) {
    auto pfnCreateWithNativeHandle =
        context.urDdiTable.Device.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hPlatform) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    ur_result_t result = pfnCreateWithNativeHandle(hNativeDevice, hPlatform,
                                                   pProperties, phDevice);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phDevice);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGetGlobalTimestamps
__urdlllocal ur_result_t UR_APICALL urDeviceGetGlobalTimestamps(
    ur_device_handle_t hDevice, ///< [in] handle of the device instance
    uint64_t *
        pDeviceTimestamp, ///< [out][optional] pointer to the Device's global timestamp that
                          ///< correlates with the Host's global timestamp value
    uint64_t *
        pHostTimestamp ///< [out][optional] pointer to the Host's global timestamp that
                       ///< correlates with the Device's global timestamp value
) {
    auto pfnGetGlobalTimestamps =
        context.urDdiTable.Device.pfnGetGlobalTimestamps;

    if (nullptr == pfnGetGlobalTimestamps) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result =
        pfnGetGlobalTimestamps(hDevice, pDeviceTimestamp, pHostTimestamp);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreate
__urdlllocal ur_result_t UR_APICALL urContextCreate(
    uint32_t DeviceCount, ///< [in] the number of devices given in phDevices
    const ur_device_handle_t
        *phDevices, ///< [in][range(0, DeviceCount)] array of handle of devices.
    const ur_context_properties_t *
        pProperties, ///< [in][optional] pointer to context creation properties.
    ur_context_handle_t
        *phContext ///< [out] pointer to handle of context object created
) {
    auto pfnCreate = context.urDdiTable.Context.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == phDevices) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phContext) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL != pProperties && UR_CONTEXT_FLAGS_MASK & pProperties->flags) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    ur_result_t result =
        pfnCreate(DeviceCount, phDevices, pProperties, phContext);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phContext);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRetain
__urdlllocal ur_result_t UR_APICALL urContextRetain(
    ur_context_handle_t
        hContext ///< [in] handle of the context to get a reference of.
) {
    auto pfnRetain = context.urDdiTable.Context.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRetain(hContext);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.incrementRefCount(hContext, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRelease
__urdlllocal ur_result_t UR_APICALL urContextRelease(
    ur_context_handle_t hContext ///< [in] handle of the context to release.
) {
    auto pfnRelease = context.urDdiTable.Context.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRelease(hContext);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.decrementRefCount(hContext, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextGetInfo
__urdlllocal ur_result_t UR_APICALL urContextGetInfo(
    ur_context_handle_t hContext, ///< [in] handle of the context
    ur_context_info_t propName,   ///< [in] type of the info to retrieve
    size_t
        propSize, ///< [in] the number of bytes of memory pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< if propSize is not equal to or greater than the real number of bytes
    ///< needed to return
    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
) {
    auto pfnGetInfo = context.urDdiTable.Context.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result =
        pfnGetInfo(hContext, propName, propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urContextGetNativeHandle(
    ur_context_handle_t hContext, ///< [in] handle of the context.
    ur_native_handle_t *
        phNativeContext ///< [out] a pointer to the native handle of the context.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Context.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phNativeContext) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnGetNativeHandle(hContext, phNativeContext);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urContextCreateWithNativeHandle(
    ur_native_handle_t
        hNativeContext,  ///< [in][nocheck] the native handle of the context.
    uint32_t numDevices, ///< [in] number of devices associated with the context
    const ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] list of devices associated with the context
    const ur_context_native_properties_t *
        pProperties, ///< [in][optional] pointer to native context properties struct
    ur_context_handle_t *
        phContext ///< [out] pointer to the handle of the context object created.
) {
    auto pfnCreateWithNativeHandle =
        context.urDdiTable.Context.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == phDevices) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phContext) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    ur_result_t result = pfnCreateWithNativeHandle(
        hNativeContext, numDevices, phDevices, pProperties, phContext);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phContext);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextSetExtendedDeleter
__urdlllocal ur_result_t UR_APICALL urContextSetExtendedDeleter(
    ur_context_handle_t hContext, ///< [in] handle of the context.
    ur_context_extended_deleter_t
        pfnDeleter, ///< [in] Function pointer to extended deleter.
    void *
        pUserData ///< [in][out][optional] pointer to data to be passed to callback.
) {
    auto pfnSetExtendedDeleter =
        context.urDdiTable.Context.pfnSetExtendedDeleter;

    if (nullptr == pfnSetExtendedDeleter) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pfnDeleter) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnSetExtendedDeleter(hContext, pfnDeleter, pUserData);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemImageCreate
__urdlllocal ur_result_t UR_APICALL urMemImageCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_mem_flags_t flags, ///< [in] allocation and usage information flags
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description
    void *pHost,           ///< [in][optional] pointer to the buffer data
    ur_mem_handle_t *phMem ///< [out] pointer to handle of image object created
) {
    auto pfnImageCreate = context.urDdiTable.Mem.pfnImageCreate;

    if (nullptr == pfnImageCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pImageFormat) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pImageDesc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_MEM_FLAGS_MASK & flags) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type) {
            return UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        }

        if (pHost == NULL &&
            (flags & (UR_MEM_FLAG_USE_HOST_POINTER |
                      UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) != 0) {
            return UR_RESULT_ERROR_INVALID_HOST_PTR;
        }

        if (pHost != NULL &&
            (flags & (UR_MEM_FLAG_USE_HOST_POINTER |
                      UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) == 0) {
            return UR_RESULT_ERROR_INVALID_HOST_PTR;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result =
        pfnImageCreate(hContext, flags, pImageFormat, pImageDesc, pHost, phMem);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phMem);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferCreate
__urdlllocal ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_mem_flags_t flags, ///< [in] allocation and usage information flags
    size_t size, ///< [in] size in bytes of the memory object to be allocated
    const ur_buffer_properties_t
        *pProperties, ///< [in][optional] pointer to buffer creation properties
    ur_mem_handle_t
        *phBuffer ///< [out] pointer to handle of the memory buffer created
) {
    auto pfnBufferCreate = context.urDdiTable.Mem.pfnBufferCreate;

    if (nullptr == pfnBufferCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_MEM_FLAGS_MASK & flags) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (pProperties == NULL &&
            (flags & (UR_MEM_FLAG_USE_HOST_POINTER |
                      UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) != 0) {
            return UR_RESULT_ERROR_INVALID_HOST_PTR;
        }

        if (pProperties != NULL && pProperties->pHost == NULL &&
            (flags & (UR_MEM_FLAG_USE_HOST_POINTER |
                      UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) != 0) {
            return UR_RESULT_ERROR_INVALID_HOST_PTR;
        }

        if (pProperties != NULL && pProperties->pHost != NULL &&
            (flags & (UR_MEM_FLAG_USE_HOST_POINTER |
                      UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) == 0) {
            return UR_RESULT_ERROR_INVALID_HOST_PTR;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result =
        pfnBufferCreate(hContext, flags, size, pProperties, phBuffer);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phBuffer);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRetain
__urdlllocal ur_result_t UR_APICALL urMemRetain(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to get access
) {
    auto pfnRetain = context.urDdiTable.Mem.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRetain(hMem);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.incrementRefCount(hMem, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRelease
__urdlllocal ur_result_t UR_APICALL urMemRelease(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to release
) {
    auto pfnRelease = context.urDdiTable.Mem.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRelease(hMem);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.decrementRefCount(hMem, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferPartition
__urdlllocal ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t
        hBuffer,          ///< [in] handle of the buffer object to allocate from
    ur_mem_flags_t flags, ///< [in] allocation and usage information flags
    ur_buffer_create_type_t bufferCreateType, ///< [in] buffer creation type
    const ur_buffer_region_t
        *pRegion, ///< [in] pointer to buffer create region information
    ur_mem_handle_t
        *phMem ///< [out] pointer to the handle of sub buffer created
) {
    auto pfnBufferPartition = context.urDdiTable.Mem.pfnBufferPartition;

    if (nullptr == pfnBufferPartition) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pRegion) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_MEM_FLAGS_MASK & flags) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (UR_BUFFER_CREATE_TYPE_REGION < bufferCreateType) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBuffer)) {
        refCountContext.logInvalidReference(hBuffer);
    }

    ur_result_t result =
        pfnBufferPartition(hBuffer, flags, bufferCreateType, pRegion, phMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urMemGetNativeHandle(
    ur_mem_handle_t hMem, ///< [in] handle of the mem.
    ur_device_handle_t
        hDevice, ///< [in] handle of the device that the native handle will be resident on.
    ur_native_handle_t
        *phNativeMem ///< [out] a pointer to the native handle of the mem.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Mem.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phNativeMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hMem)) {
        refCountContext.logInvalidReference(hMem);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnGetNativeHandle(hMem, hDevice, phNativeMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urMemBufferCreateWithNativeHandle(
    ur_native_handle_t
        hNativeMem, ///< [in][nocheck] the native handle to the memory.
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    const ur_mem_native_properties_t *
        pProperties, ///< [in][optional] pointer to native memory creation properties.
    ur_mem_handle_t
        *phMem ///< [out] pointer to handle of buffer memory object created.
) {
    auto pfnBufferCreateWithNativeHandle =
        context.urDdiTable.Mem.pfnBufferCreateWithNativeHandle;

    if (nullptr == pfnBufferCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnBufferCreateWithNativeHandle(hNativeMem, hContext,
                                                         pProperties, phMem);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phMem);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemImageCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urMemImageCreateWithNativeHandle(
    ur_native_handle_t
        hNativeMem, ///< [in][nocheck] the native handle to the memory.
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification.
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description.
    const ur_mem_native_properties_t *
        pProperties, ///< [in][optional] pointer to native memory creation properties.
    ur_mem_handle_t
        *phMem ///< [out] pointer to handle of image memory object created.
) {
    auto pfnImageCreateWithNativeHandle =
        context.urDdiTable.Mem.pfnImageCreateWithNativeHandle;

    if (nullptr == pfnImageCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pImageFormat) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pImageDesc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnImageCreateWithNativeHandle(
        hNativeMem, hContext, pImageFormat, pImageDesc, pProperties, phMem);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phMem);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetInfo
__urdlllocal ur_result_t UR_APICALL urMemGetInfo(
    ur_mem_handle_t
        hMemory,            ///< [in] handle to the memory object being queried.
    ur_mem_info_t propName, ///< [in] type of the info to retrieve.
    size_t
        propSize, ///< [in] the number of bytes of memory pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If propSize is less than the real number of bytes needed to return
    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
) {
    auto pfnGetInfo = context.urDdiTable.Mem.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hMemory) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_MEM_INFO_CONTEXT < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hMemory)) {
        refCountContext.logInvalidReference(hMemory);
    }

    ur_result_t result =
        pfnGetInfo(hMemory, propName, propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemImageGetInfo
__urdlllocal ur_result_t UR_APICALL urMemImageGetInfo(
    ur_mem_handle_t hMemory, ///< [in] handle to the image object being queried.
    ur_image_info_t propName, ///< [in] type of image info to retrieve.
    size_t
        propSize, ///< [in] the number of bytes of memory pointer to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If propSize is less than the real number of bytes needed to return
    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
) {
    auto pfnImageGetInfo = context.urDdiTable.Mem.pfnImageGetInfo;

    if (nullptr == pfnImageGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hMemory) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_IMAGE_INFO_DEPTH < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hMemory)) {
        refCountContext.logInvalidReference(hMemory);
    }

    ur_result_t result =
        pfnImageGetInfo(hMemory, propName, propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerCreate
__urdlllocal ur_result_t UR_APICALL urSamplerCreate(
    ur_context_handle_t hContext,   ///< [in] handle of the context object
    const ur_sampler_desc_t *pDesc, ///< [in] pointer to the sampler description
    ur_sampler_handle_t
        *phSampler ///< [out] pointer to handle of sampler object created
) {
    auto pfnCreate = context.urDdiTable.Sampler.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pDesc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phSampler) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT <
            pDesc->addressingMode) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (UR_SAMPLER_FILTER_MODE_LINEAR < pDesc->filterMode) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnCreate(hContext, pDesc, phSampler);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phSampler);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRetain
__urdlllocal ur_result_t UR_APICALL urSamplerRetain(
    ur_sampler_handle_t
        hSampler ///< [in] handle of the sampler object to get access
) {
    auto pfnRetain = context.urDdiTable.Sampler.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hSampler) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRetain(hSampler);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.incrementRefCount(hSampler, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRelease
__urdlllocal ur_result_t UR_APICALL urSamplerRelease(
    ur_sampler_handle_t
        hSampler ///< [in] handle of the sampler object to release
) {
    auto pfnRelease = context.urDdiTable.Sampler.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hSampler) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRelease(hSampler);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.decrementRefCount(hSampler, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerGetInfo
__urdlllocal ur_result_t UR_APICALL urSamplerGetInfo(
    ur_sampler_handle_t hSampler, ///< [in] handle of the sampler object
    ur_sampler_info_t propName, ///< [in] name of the sampler property to query
    size_t
        propSize, ///< [in] size in bytes of the sampler property value provided
    void *
        pPropValue, ///< [out][typename(propName, propSize)][optional] value of the sampler
                    ///< property
    size_t *
        pPropSizeRet ///< [out][optional] size in bytes returned in sampler property value
) {
    auto pfnGetInfo = context.urDdiTable.Sampler.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hSampler) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_SAMPLER_INFO_FILTER_MODE < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hSampler)) {
        refCountContext.logInvalidReference(hSampler);
    }

    ur_result_t result =
        pfnGetInfo(hSampler, propName, propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urSamplerGetNativeHandle(
    ur_sampler_handle_t hSampler, ///< [in] handle of the sampler.
    ur_native_handle_t *
        phNativeSampler ///< [out] a pointer to the native handle of the sampler.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Sampler.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hSampler) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phNativeSampler) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hSampler)) {
        refCountContext.logInvalidReference(hSampler);
    }

    ur_result_t result = pfnGetNativeHandle(hSampler, phNativeSampler);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urSamplerCreateWithNativeHandle(
    ur_native_handle_t
        hNativeSampler, ///< [in][nocheck] the native handle of the sampler.
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const ur_sampler_native_properties_t *
        pProperties, ///< [in][optional] pointer to native sampler properties struct.
    ur_sampler_handle_t *
        phSampler ///< [out] pointer to the handle of the sampler object created.
) {
    auto pfnCreateWithNativeHandle =
        context.urDdiTable.Sampler.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phSampler) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnCreateWithNativeHandle(hNativeSampler, hContext,
                                                   pProperties, phSampler);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phSampler);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMHostAlloc
__urdlllocal ur_result_t UR_APICALL urUSMHostAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const ur_usm_desc_t
        *pUSMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t
        pool, ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t
        size, ///< [in] size in bytes of the USM memory object to be allocated
    void **ppMem ///< [out] pointer to USM host memory object
) {
    auto pfnHostAlloc = context.urDdiTable.USM.pfnHostAlloc;

    if (nullptr == pfnHostAlloc) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == ppMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL != pUSMDesc && UR_USM_ADVICE_FLAGS_MASK & pUSMDesc->hints) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (pUSMDesc && pUSMDesc->align != 0 &&
            ((pUSMDesc->align & (pUSMDesc->align - 1)) != 0)) {
            return UR_RESULT_ERROR_INVALID_VALUE;
        }

        if (size == 0) {
            return UR_RESULT_ERROR_INVALID_USM_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(pool)) {
        refCountContext.logInvalidReference(pool);
    }

    ur_result_t result = pfnHostAlloc(hContext, pUSMDesc, pool, size, ppMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMDeviceAlloc
__urdlllocal ur_result_t UR_APICALL urUSMDeviceAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_usm_desc_t
        *pUSMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t
        pool, ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t
        size, ///< [in] size in bytes of the USM memory object to be allocated
    void **ppMem ///< [out] pointer to USM device memory object
) {
    auto pfnDeviceAlloc = context.urDdiTable.USM.pfnDeviceAlloc;

    if (nullptr == pfnDeviceAlloc) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == ppMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL != pUSMDesc && UR_USM_ADVICE_FLAGS_MASK & pUSMDesc->hints) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (pUSMDesc && pUSMDesc->align != 0 &&
            ((pUSMDesc->align & (pUSMDesc->align - 1)) != 0)) {
            return UR_RESULT_ERROR_INVALID_VALUE;
        }

        if (size == 0) {
            return UR_RESULT_ERROR_INVALID_USM_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(pool)) {
        refCountContext.logInvalidReference(pool);
    }

    ur_result_t result =
        pfnDeviceAlloc(hContext, hDevice, pUSMDesc, pool, size, ppMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMSharedAlloc
__urdlllocal ur_result_t UR_APICALL urUSMSharedAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_usm_desc_t *
        pUSMDesc, ///< [in][optional] Pointer to USM memory allocation descriptor.
    ur_usm_pool_handle_t
        pool, ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t
        size, ///< [in] size in bytes of the USM memory object to be allocated
    void **ppMem ///< [out] pointer to USM shared memory object
) {
    auto pfnSharedAlloc = context.urDdiTable.USM.pfnSharedAlloc;

    if (nullptr == pfnSharedAlloc) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == ppMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL != pUSMDesc && UR_USM_ADVICE_FLAGS_MASK & pUSMDesc->hints) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (pUSMDesc && pUSMDesc->align != 0 &&
            ((pUSMDesc->align & (pUSMDesc->align - 1)) != 0)) {
            return UR_RESULT_ERROR_INVALID_VALUE;
        }

        if (size == 0) {
            return UR_RESULT_ERROR_INVALID_USM_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(pool)) {
        refCountContext.logInvalidReference(pool);
    }

    ur_result_t result =
        pfnSharedAlloc(hContext, hDevice, pUSMDesc, pool, size, ppMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMFree
__urdlllocal ur_result_t UR_APICALL urUSMFree(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to USM memory object
) {
    auto pfnFree = context.urDdiTable.USM.pfnFree;

    if (nullptr == pfnFree) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnFree(hContext, pMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMGetMemAllocInfo
__urdlllocal ur_result_t UR_APICALL urUSMGetMemAllocInfo(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const void *pMem,             ///< [in] pointer to USM memory object
    ur_usm_alloc_info_t
        propName, ///< [in] the name of the USM allocation property to query
    size_t
        propSize, ///< [in] size in bytes of the USM allocation property value
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the USM
                    ///< allocation property
    size_t *
        pPropSizeRet ///< [out][optional] bytes returned in USM allocation property
) {
    auto pfnGetMemAllocInfo = context.urDdiTable.USM.pfnGetMemAllocInfo;

    if (nullptr == pfnGetMemAllocInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_USM_ALLOC_INFO_POOL < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnGetMemAllocInfo(hContext, pMem, propName, propSize,
                                            pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolCreate
__urdlllocal ur_result_t UR_APICALL urUSMPoolCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_usm_pool_desc_t *
        pPoolDesc, ///< [in] pointer to USM pool descriptor. Can be chained with
                   ///< ::ur_usm_pool_limits_desc_t
    ur_usm_pool_handle_t *ppPool ///< [out] pointer to USM memory pool
) {
    auto pfnPoolCreate = context.urDdiTable.USM.pfnPoolCreate;

    if (nullptr == pfnPoolCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pPoolDesc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == ppPool) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_USM_POOL_FLAGS_MASK & pPoolDesc->flags) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnPoolCreate(hContext, pPoolDesc, ppPool);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*ppPool);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolRetain
__urdlllocal ur_result_t UR_APICALL urUSMPoolRetain(
    ur_usm_pool_handle_t pPool ///< [in] pointer to USM memory pool
) {
    auto pfnPoolRetain = context.urDdiTable.USM.pfnPoolRetain;

    if (nullptr == pfnPoolRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == pPool) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnPoolRetain(pPool);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.incrementRefCount(pPool, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolRelease
__urdlllocal ur_result_t UR_APICALL urUSMPoolRelease(
    ur_usm_pool_handle_t pPool ///< [in] pointer to USM memory pool
) {
    auto pfnPoolRelease = context.urDdiTable.USM.pfnPoolRelease;

    if (nullptr == pfnPoolRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == pPool) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnPoolRelease(pPool);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.decrementRefCount(pPool, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolGetInfo
__urdlllocal ur_result_t UR_APICALL urUSMPoolGetInfo(
    ur_usm_pool_handle_t hPool,  ///< [in] handle of the USM memory pool
    ur_usm_pool_info_t propName, ///< [in] name of the pool property to query
    size_t propSize, ///< [in] size in bytes of the pool property value provided
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the pool
                    ///< property
    size_t *
        pPropSizeRet ///< [out][optional] size in bytes returned in pool property value
) {
    auto pfnPoolGetInfo = context.urDdiTable.USM.pfnPoolGetInfo;

    if (nullptr == pfnPoolGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hPool) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_USM_POOL_INFO_CONTEXT < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hPool)) {
        refCountContext.logInvalidReference(hPool);
    }

    ur_result_t result =
        pfnPoolGetInfo(hPool, propName, propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemGranularityGetInfo
__urdlllocal ur_result_t UR_APICALL urVirtualMemGranularityGetInfo(
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    ur_device_handle_t
        hDevice, ///< [in][optional] is the device to get the granularity from, if the
    ///< device is null then the granularity is suitable for all devices in context.
    ur_virtual_mem_granularity_info_t
        propName, ///< [in] type of the info to query.
    size_t
        propSize, ///< [in] size in bytes of the memory pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
    ///< the info. If propSize is less than the real number of bytes needed to
    ///< return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    ///< returned and pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName."
) {
    auto pfnGranularityGetInfo =
        context.urDdiTable.VirtualMem.pfnGranularityGetInfo;

    if (nullptr == pfnGranularityGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnGranularityGetInfo(
        hContext, hDevice, propName, propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemReserve
__urdlllocal ur_result_t UR_APICALL urVirtualMemReserve(
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    const void *
        pStart, ///< [in][optional] pointer to the start of the virtual memory region to
    ///< reserve, specifying a null value causes the implementation to select a
    ///< start address.
    size_t
        size, ///< [in] size in bytes of the virtual address range to reserve.
    void **
        ppStart ///< [out] pointer to the returned address at the start of reserved virtual
                ///< memory range.
) {
    auto pfnReserve = context.urDdiTable.VirtualMem.pfnReserve;

    if (nullptr == pfnReserve) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == ppStart) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnReserve(hContext, pStart, size, ppStart);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemFree
__urdlllocal ur_result_t UR_APICALL urVirtualMemFree(
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    const void *
        pStart, ///< [in] pointer to the start of the virtual memory range to free.
    size_t size ///< [in] size in bytes of the virtual memory range to free.
) {
    auto pfnFree = context.urDdiTable.VirtualMem.pfnFree;

    if (nullptr == pfnFree) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pStart) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnFree(hContext, pStart, size);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemMap
__urdlllocal ur_result_t UR_APICALL urVirtualMemMap(
    ur_context_handle_t hContext, ///< [in] handle to the context object.
    const void
        *pStart, ///< [in] pointer to the start of the virtual memory range.
    size_t size, ///< [in] size in bytes of the virtual memory range to map.
    ur_physical_mem_handle_t
        hPhysicalMem, ///< [in] handle of the physical memory to map pStart to.
    size_t
        offset, ///< [in] offset in bytes into the physical memory to map pStart to.
    ur_virtual_mem_access_flags_t
        flags ///< [in] access flags for the physical memory mapping.
) {
    auto pfnMap = context.urDdiTable.VirtualMem.pfnMap;

    if (nullptr == pfnMap) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hPhysicalMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pStart) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_VIRTUAL_MEM_ACCESS_FLAGS_MASK & flags) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hPhysicalMem)) {
        refCountContext.logInvalidReference(hPhysicalMem);
    }

    ur_result_t result =
        pfnMap(hContext, pStart, size, hPhysicalMem, offset, flags);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemUnmap
__urdlllocal ur_result_t UR_APICALL urVirtualMemUnmap(
    ur_context_handle_t hContext, ///< [in] handle to the context object.
    const void *
        pStart, ///< [in] pointer to the start of the mapped virtual memory range
    size_t size ///< [in] size in bytes of the virtual memory range.
) {
    auto pfnUnmap = context.urDdiTable.VirtualMem.pfnUnmap;

    if (nullptr == pfnUnmap) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pStart) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnUnmap(hContext, pStart, size);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemSetAccess
__urdlllocal ur_result_t UR_APICALL urVirtualMemSetAccess(
    ur_context_handle_t hContext, ///< [in] handle to the context object.
    const void
        *pStart, ///< [in] pointer to the start of the virtual memory range.
    size_t size, ///< [in] size in bytes of the virtual memory range.
    ur_virtual_mem_access_flags_t
        flags ///< [in] access flags to set for the mapped virtual memory range.
) {
    auto pfnSetAccess = context.urDdiTable.VirtualMem.pfnSetAccess;

    if (nullptr == pfnSetAccess) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pStart) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_VIRTUAL_MEM_ACCESS_FLAGS_MASK & flags) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnSetAccess(hContext, pStart, size, flags);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemGetInfo
__urdlllocal ur_result_t UR_APICALL urVirtualMemGetInfo(
    ur_context_handle_t hContext, ///< [in] handle to the context object.
    const void
        *pStart, ///< [in] pointer to the start of the virtual memory range.
    size_t size, ///< [in] size in bytes of the virtual memory range.
    ur_virtual_mem_info_t propName, ///< [in] type of the info to query.
    size_t
        propSize, ///< [in] size in bytes of the memory pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
    ///< the info. If propSize is less than the real number of bytes needed to
    ///< return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    ///< returned and pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName."
) {
    auto pfnGetInfo = context.urDdiTable.VirtualMem.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pStart) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_VIRTUAL_MEM_INFO_ACCESS_MODE < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnGetInfo(hContext, pStart, size, propName, propSize,
                                    pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemCreate
__urdlllocal ur_result_t UR_APICALL urPhysicalMemCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    ur_device_handle_t hDevice,   ///< [in] handle of the device object.
    size_t
        size, ///< [in] size in bytes of physical memory to allocate, must be a multiple
              ///< of ::UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM.
    const ur_physical_mem_properties_t *
        pProperties, ///< [in][optional] pointer to physical memory creation properties.
    ur_physical_mem_handle_t *
        phPhysicalMem ///< [out] pointer to handle of physical memory object created.
) {
    auto pfnCreate = context.urDdiTable.PhysicalMem.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phPhysicalMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL != pProperties &&
            UR_PHYSICAL_MEM_FLAGS_MASK & pProperties->flags) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result =
        pfnCreate(hContext, hDevice, size, pProperties, phPhysicalMem);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phPhysicalMem);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemRetain
__urdlllocal ur_result_t UR_APICALL urPhysicalMemRetain(
    ur_physical_mem_handle_t
        hPhysicalMem ///< [in] handle of the physical memory object to retain.
) {
    auto pfnRetain = context.urDdiTable.PhysicalMem.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hPhysicalMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRetain(hPhysicalMem);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.incrementRefCount(hPhysicalMem, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemRelease
__urdlllocal ur_result_t UR_APICALL urPhysicalMemRelease(
    ur_physical_mem_handle_t
        hPhysicalMem ///< [in] handle of the physical memory object to release.
) {
    auto pfnRelease = context.urDdiTable.PhysicalMem.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hPhysicalMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRelease(hPhysicalMem);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.decrementRefCount(hPhysicalMem, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithIL
__urdlllocal ur_result_t UR_APICALL urProgramCreateWithIL(
    ur_context_handle_t hContext, ///< [in] handle of the context instance
    const void *pIL,              ///< [in] pointer to IL binary.
    size_t length,                ///< [in] length of `pIL` in bytes.
    const ur_program_properties_t *
        pProperties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of program object created.
) {
    auto pfnCreateWithIL = context.urDdiTable.Program.pfnCreateWithIL;

    if (nullptr == pfnCreateWithIL) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pIL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL != pProperties && pProperties->count > 0 &&
            NULL == pProperties->pMetadatas) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL != pProperties && NULL != pProperties->pMetadatas &&
            pProperties->count == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (length == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result =
        pfnCreateWithIL(hContext, pIL, length, pProperties, phProgram);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phProgram);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithBinary
__urdlllocal ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, ///< [in] handle of the context instance
    ur_device_handle_t
        hDevice,            ///< [in] handle to device associated with binary.
    size_t size,            ///< [in] size in bytes.
    const uint8_t *pBinary, ///< [in] pointer to binary.
    const ur_program_properties_t *
        pProperties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of Program object created.
) {
    auto pfnCreateWithBinary = context.urDdiTable.Program.pfnCreateWithBinary;

    if (nullptr == pfnCreateWithBinary) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pBinary) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL != pProperties && pProperties->count > 0 &&
            NULL == pProperties->pMetadatas) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL != pProperties && NULL != pProperties->pMetadatas &&
            pProperties->count == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnCreateWithBinary(hContext, hDevice, size, pBinary,
                                             pProperties, phProgram);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phProgram);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramBuild
__urdlllocal ur_result_t UR_APICALL urProgramBuild(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    ur_program_handle_t hProgram, ///< [in] Handle of the program to build.
    const char *
        pOptions ///< [in][optional] pointer to build options null-terminated string.
) {
    auto pfnBuild = context.urDdiTable.Program.pfnBuild;

    if (nullptr == pfnBuild) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result = pfnBuild(hContext, hProgram, pOptions);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCompile
__urdlllocal ur_result_t UR_APICALL urProgramCompile(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    ur_program_handle_t
        hProgram, ///< [in][out] handle of the program to compile.
    const char *
        pOptions ///< [in][optional] pointer to build options null-terminated string.
) {
    auto pfnCompile = context.urDdiTable.Program.pfnCompile;

    if (nullptr == pfnCompile) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result = pfnCompile(hContext, hProgram, pOptions);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramLink
__urdlllocal ur_result_t UR_APICALL urProgramLink(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    uint32_t count, ///< [in] number of program handles in `phPrograms`.
    const ur_program_handle_t *
        phPrograms, ///< [in][range(0, count)] pointer to array of program handles.
    const char *
        pOptions, ///< [in][optional] pointer to linker options null-terminated string.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of program object created.
) {
    auto pfnLink = context.urDdiTable.Program.pfnLink;

    if (nullptr == pfnLink) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phPrograms) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (count == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result =
        pfnLink(hContext, count, phPrograms, pOptions, phProgram);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRetain
__urdlllocal ur_result_t UR_APICALL urProgramRetain(
    ur_program_handle_t hProgram ///< [in] handle for the Program to retain
) {
    auto pfnRetain = context.urDdiTable.Program.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRetain(hProgram);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.incrementRefCount(hProgram, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRelease
__urdlllocal ur_result_t UR_APICALL urProgramRelease(
    ur_program_handle_t hProgram ///< [in] handle for the Program to release
) {
    auto pfnRelease = context.urDdiTable.Program.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRelease(hProgram);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.decrementRefCount(hProgram, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetFunctionPointer
__urdlllocal ur_result_t UR_APICALL urProgramGetFunctionPointer(
    ur_device_handle_t
        hDevice, ///< [in] handle of the device to retrieve pointer for.
    ur_program_handle_t
        hProgram, ///< [in] handle of the program to search for function in.
    ///< The program must already be built to the specified device, or
    ///< otherwise ::UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE is returned.
    const char *
        pFunctionName, ///< [in] A null-terminates string denoting the mangled function name.
    void **
        ppFunctionPointer ///< [out] Returns the pointer to the function if it is found in the program.
) {
    auto pfnGetFunctionPointer =
        context.urDdiTable.Program.pfnGetFunctionPointer;

    if (nullptr == pfnGetFunctionPointer) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pFunctionName) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == ppFunctionPointer) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result = pfnGetFunctionPointer(hDevice, hProgram, pFunctionName,
                                               ppFunctionPointer);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetInfo
__urdlllocal ur_result_t UR_APICALL urProgramGetInfo(
    ur_program_handle_t hProgram, ///< [in] handle of the Program object
    ur_program_info_t propName, ///< [in] name of the Program property to query
    size_t propSize,            ///< [in] the size of the Program property.
    void *
        pPropValue, ///< [in,out][optional][typename(propName, propSize)] array of bytes of
                    ///< holding the program info property.
    ///< If propSize is not equal to or greater than the real number of bytes
    ///< needed to return
    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
) {
    auto pfnGetInfo = context.urDdiTable.Program.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_PROGRAM_INFO_KERNEL_NAMES < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result =
        pfnGetInfo(hProgram, propName, propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetBuildInfo
__urdlllocal ur_result_t UR_APICALL urProgramGetBuildInfo(
    ur_program_handle_t hProgram, ///< [in] handle of the Program object
    ur_device_handle_t hDevice,   ///< [in] handle of the Device object
    ur_program_build_info_t
        propName,    ///< [in] name of the Program build info to query
    size_t propSize, ///< [in] size of the Program build info property.
    void *
        pPropValue, ///< [in,out][optional][typename(propName, propSize)] value of the Program
                    ///< build property.
    ///< If propSize is not equal to or greater than the real number of bytes
    ///< needed to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE
    ///< error is returned and pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of data being
                     ///< queried by propName.
) {
    auto pfnGetBuildInfo = context.urDdiTable.Program.pfnGetBuildInfo;

    if (nullptr == pfnGetBuildInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (UR_PROGRAM_BUILD_INFO_BINARY_TYPE < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnGetBuildInfo(hProgram, hDevice, propName, propSize,
                                         pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramSetSpecializationConstants
__urdlllocal ur_result_t UR_APICALL urProgramSetSpecializationConstants(
    ur_program_handle_t hProgram, ///< [in] handle of the Program object
    uint32_t count, ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t *
        pSpecConstants ///< [in][range(0, count)] array of specialization constant value
                       ///< descriptions
) {
    auto pfnSetSpecializationConstants =
        context.urDdiTable.Program.pfnSetSpecializationConstants;

    if (nullptr == pfnSetSpecializationConstants) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pSpecConstants) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (count == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result =
        pfnSetSpecializationConstants(hProgram, count, pSpecConstants);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t hProgram, ///< [in] handle of the program.
    ur_native_handle_t *
        phNativeProgram ///< [out] a pointer to the native handle of the program.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Program.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phNativeProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result = pfnGetNativeHandle(hProgram, phNativeProgram);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t
        hNativeProgram, ///< [in][nocheck] the native handle of the program.
    ur_context_handle_t hContext, ///< [in] handle of the context instance
    const ur_program_native_properties_t *
        pProperties, ///< [in][optional] pointer to native program properties struct.
    ur_program_handle_t *
        phProgram ///< [out] pointer to the handle of the program object created.
) {
    auto pfnCreateWithNativeHandle =
        context.urDdiTable.Program.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnCreateWithNativeHandle(hNativeProgram, hContext,
                                                   pProperties, phProgram);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phProgram);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelCreate
__urdlllocal ur_result_t UR_APICALL urKernelCreate(
    ur_program_handle_t hProgram, ///< [in] handle of the program instance
    const char *pKernelName,      ///< [in] pointer to null-terminated string.
    ur_kernel_handle_t
        *phKernel ///< [out] pointer to handle of kernel object created.
) {
    auto pfnCreate = context.urDdiTable.Kernel.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pKernelName) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result = pfnCreate(hProgram, pKernelName, phKernel);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phKernel);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgValue
__urdlllocal ur_result_t UR_APICALL urKernelSetArgValue(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    size_t argSize,    ///< [in] size of argument type
    const ur_kernel_arg_value_properties_t
        *pProperties, ///< [in][optional] pointer to value properties.
    const void
        *pArgValue ///< [in] argument value represented as matching arg type.
) {
    auto pfnSetArgValue = context.urDdiTable.Kernel.pfnSetArgValue;

    if (nullptr == pfnSetArgValue) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pArgValue) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    ur_result_t result =
        pfnSetArgValue(hKernel, argIndex, argSize, pProperties, pArgValue);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgLocal
__urdlllocal ur_result_t UR_APICALL urKernelSetArgLocal(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    size_t
        argSize, ///< [in] size of the local buffer to be allocated by the runtime
    const ur_kernel_arg_local_properties_t
        *pProperties ///< [in][optional] pointer to local buffer properties.
) {
    auto pfnSetArgLocal = context.urDdiTable.Kernel.pfnSetArgLocal;

    if (nullptr == pfnSetArgLocal) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    ur_result_t result =
        pfnSetArgLocal(hKernel, argIndex, argSize, pProperties);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetInfo
__urdlllocal ur_result_t UR_APICALL urKernelGetInfo(
    ur_kernel_handle_t hKernel, ///< [in] handle of the Kernel object
    ur_kernel_info_t propName,  ///< [in] name of the Kernel property to query
    size_t propSize,            ///< [in] the size of the Kernel property value.
    void *
        pPropValue, ///< [in,out][optional][typename(propName, propSize)] array of bytes
                    ///< holding the kernel info property.
    ///< If propSize is not equal to or greater than the real number of bytes
    ///< needed to return
    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of data being
                     ///< queried by propName.
) {
    auto pfnGetInfo = context.urDdiTable.Kernel.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_KERNEL_INFO_NUM_REGS < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    ur_result_t result =
        pfnGetInfo(hKernel, propName, propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetGroupInfo
__urdlllocal ur_result_t UR_APICALL urKernelGetGroupInfo(
    ur_kernel_handle_t hKernel, ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice, ///< [in] handle of the Device object
    ur_kernel_group_info_t
        propName,    ///< [in] name of the work Group property to query
    size_t propSize, ///< [in] size of the Kernel Work Group property value
    void *
        pPropValue, ///< [in,out][optional][typename(propName, propSize)] value of the Kernel
                    ///< Work Group property.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of data being
                     ///< queried by propName.
) {
    auto pfnGetGroupInfo = context.urDdiTable.Kernel.pfnGetGroupInfo;

    if (nullptr == pfnGetGroupInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnGetGroupInfo(hKernel, hDevice, propName, propSize,
                                         pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetSubGroupInfo
__urdlllocal ur_result_t UR_APICALL urKernelGetSubGroupInfo(
    ur_kernel_handle_t hKernel, ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice, ///< [in] handle of the Device object
    ur_kernel_sub_group_info_t
        propName,    ///< [in] name of the SubGroup property to query
    size_t propSize, ///< [in] size of the Kernel SubGroup property value
    void *
        pPropValue, ///< [in,out][optional][typename(propName, propSize)] value of the Kernel
                    ///< SubGroup property.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of data being
                     ///< queried by propName.
) {
    auto pfnGetSubGroupInfo = context.urDdiTable.Kernel.pfnGetSubGroupInfo;

    if (nullptr == pfnGetSubGroupInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnGetSubGroupInfo(hKernel, hDevice, propName,
                                            propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRetain
__urdlllocal ur_result_t UR_APICALL urKernelRetain(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to retain
) {
    auto pfnRetain = context.urDdiTable.Kernel.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRetain(hKernel);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.incrementRefCount(hKernel, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRelease
__urdlllocal ur_result_t UR_APICALL urKernelRelease(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to release
) {
    auto pfnRelease = context.urDdiTable.Kernel.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRelease(hKernel);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.decrementRefCount(hKernel, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgPointer
__urdlllocal ur_result_t UR_APICALL urKernelSetArgPointer(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_pointer_properties_t
        *pProperties, ///< [in][optional] pointer to USM pointer properties.
    const void *
        pArgValue ///< [in][optional] USM pointer to memory location holding the argument
                  ///< value. If null then argument value is considered null.
) {
    auto pfnSetArgPointer = context.urDdiTable.Kernel.pfnSetArgPointer;

    if (nullptr == pfnSetArgPointer) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    ur_result_t result =
        pfnSetArgPointer(hKernel, argIndex, pProperties, pArgValue);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetExecInfo
__urdlllocal ur_result_t UR_APICALL urKernelSetExecInfo(
    ur_kernel_handle_t hKernel,     ///< [in] handle of the kernel object
    ur_kernel_exec_info_t propName, ///< [in] name of the execution attribute
    size_t propSize,                ///< [in] size in byte the attribute value
    const ur_kernel_exec_info_properties_t
        *pProperties, ///< [in][optional] pointer to execution info properties.
    const void *
        pPropValue ///< [in][typename(propName, propSize)] pointer to memory location holding
                   ///< the property value.
) {
    auto pfnSetExecInfo = context.urDdiTable.Kernel.pfnSetExecInfo;

    if (nullptr == pfnSetExecInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pPropValue) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_KERNEL_EXEC_INFO_CACHE_CONFIG < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    ur_result_t result =
        pfnSetExecInfo(hKernel, propName, propSize, pProperties, pPropValue);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgSampler
__urdlllocal ur_result_t UR_APICALL urKernelSetArgSampler(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_sampler_properties_t
        *pProperties, ///< [in][optional] pointer to sampler properties.
    ur_sampler_handle_t hArgValue ///< [in] handle of Sampler object.
) {
    auto pfnSetArgSampler = context.urDdiTable.Kernel.pfnSetArgSampler;

    if (nullptr == pfnSetArgSampler) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hArgValue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hArgValue)) {
        refCountContext.logInvalidReference(hArgValue);
    }

    ur_result_t result =
        pfnSetArgSampler(hKernel, argIndex, pProperties, hArgValue);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgMemObj
__urdlllocal ur_result_t UR_APICALL urKernelSetArgMemObj(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_mem_obj_properties_t
        *pProperties, ///< [in][optional] pointer to Memory object properties.
    ur_mem_handle_t hArgValue ///< [in][optional] handle of Memory object.
) {
    auto pfnSetArgMemObj = context.urDdiTable.Kernel.pfnSetArgMemObj;

    if (nullptr == pfnSetArgMemObj) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL != pProperties &&
            UR_MEM_FLAGS_MASK & pProperties->memoryAccess) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hArgValue)) {
        refCountContext.logInvalidReference(hArgValue);
    }

    ur_result_t result =
        pfnSetArgMemObj(hKernel, argIndex, pProperties, hArgValue);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetSpecializationConstants
__urdlllocal ur_result_t UR_APICALL urKernelSetSpecializationConstants(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t count, ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t *
        pSpecConstants ///< [in] array of specialization constant value descriptions
) {
    auto pfnSetSpecializationConstants =
        context.urDdiTable.Kernel.pfnSetSpecializationConstants;

    if (nullptr == pfnSetSpecializationConstants) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pSpecConstants) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (count == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    ur_result_t result =
        pfnSetSpecializationConstants(hKernel, count, pSpecConstants);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urKernelGetNativeHandle(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel.
    ur_native_handle_t
        *phNativeKernel ///< [out] a pointer to the native handle of the kernel.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Kernel.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phNativeKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    ur_result_t result = pfnGetNativeHandle(hKernel, phNativeKernel);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urKernelCreateWithNativeHandle(
    ur_native_handle_t
        hNativeKernel, ///< [in][nocheck] the native handle of the kernel.
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_program_handle_t
        hProgram, ///< [in] handle of the program associated with the kernel
    const ur_kernel_native_properties_t *
        pProperties, ///< [in][optional] pointer to native kernel properties struct
    ur_kernel_handle_t
        *phKernel ///< [out] pointer to the handle of the kernel object created.
) {
    auto pfnCreateWithNativeHandle =
        context.urDdiTable.Kernel.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result = pfnCreateWithNativeHandle(
        hNativeKernel, hContext, hProgram, pProperties, phKernel);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phKernel);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueGetInfo
__urdlllocal ur_result_t UR_APICALL urQueueGetInfo(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_queue_info_t propName, ///< [in] name of the queue property to query
    size_t
        propSize, ///< [in] size in bytes of the queue property value provided
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the queue
                    ///< property
    size_t *
        pPropSizeRet ///< [out][optional] size in bytes returned in queue property value
) {
    auto pfnGetInfo = context.urDdiTable.Queue.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_QUEUE_INFO_EMPTY < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result =
        pfnGetInfo(hQueue, propName, propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueCreate
__urdlllocal ur_result_t UR_APICALL urQueueCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_queue_properties_t
        *pProperties, ///< [in][optional] pointer to queue creation properties.
    ur_queue_handle_t
        *phQueue ///< [out] pointer to handle of queue object created
) {
    auto pfnCreate = context.urDdiTable.Queue.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL != pProperties && UR_QUEUE_FLAGS_MASK & pProperties->flags) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (pProperties != NULL &&
            pProperties->flags & UR_QUEUE_FLAG_PRIORITY_HIGH &&
            pProperties->flags & UR_QUEUE_FLAG_PRIORITY_LOW) {
            return UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES;
        }

        if (pProperties != NULL &&
            pProperties->flags & UR_QUEUE_FLAG_SUBMISSION_BATCHED &&
            pProperties->flags & UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE) {
            return UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnCreate(hContext, hDevice, pProperties, phQueue);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phQueue);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRetain
__urdlllocal ur_result_t UR_APICALL urQueueRetain(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to get access
) {
    auto pfnRetain = context.urDdiTable.Queue.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRetain(hQueue);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.incrementRefCount(hQueue, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRelease
__urdlllocal ur_result_t UR_APICALL urQueueRelease(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to release
) {
    auto pfnRelease = context.urDdiTable.Queue.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRelease(hQueue);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.decrementRefCount(hQueue, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urQueueGetNativeHandle(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue.
    ur_queue_native_desc_t
        *pDesc, ///< [in][optional] pointer to native descriptor
    ur_native_handle_t
        *phNativeQueue ///< [out] a pointer to the native handle of the queue.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Queue.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phNativeQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result = pfnGetNativeHandle(hQueue, pDesc, phNativeQueue);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    ur_native_handle_t
        hNativeQueue, ///< [in][nocheck] the native handle of the queue.
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_queue_native_properties_t *
        pProperties, ///< [in][optional] pointer to native queue properties struct
    ur_queue_handle_t
        *phQueue ///< [out] pointer to the handle of the queue object created.
) {
    auto pfnCreateWithNativeHandle =
        context.urDdiTable.Queue.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnCreateWithNativeHandle(
        hNativeQueue, hContext, hDevice, pProperties, phQueue);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phQueue);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFinish
__urdlllocal ur_result_t UR_APICALL urQueueFinish(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be finished.
) {
    auto pfnFinish = context.urDdiTable.Queue.pfnFinish;

    if (nullptr == pfnFinish) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result = pfnFinish(hQueue);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFlush
__urdlllocal ur_result_t UR_APICALL urQueueFlush(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be flushed.
) {
    auto pfnFlush = context.urDdiTable.Queue.pfnFlush;

    if (nullptr == pfnFlush) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result = pfnFlush(hQueue);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventGetInfo
__urdlllocal ur_result_t UR_APICALL urEventGetInfo(
    ur_event_handle_t hEvent, ///< [in] handle of the event object
    ur_event_info_t propName, ///< [in] the name of the event property to query
    size_t propSize, ///< [in] size in bytes of the event property value
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the event
                    ///< property
    size_t *pPropSizeRet ///< [out][optional] bytes returned in event property
) {
    auto pfnGetInfo = context.urDdiTable.Event.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hEvent) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_EVENT_INFO_REFERENCE_COUNT < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hEvent)) {
        refCountContext.logInvalidReference(hEvent);
    }

    ur_result_t result =
        pfnGetInfo(hEvent, propName, propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventGetProfilingInfo
__urdlllocal ur_result_t UR_APICALL urEventGetProfilingInfo(
    ur_event_handle_t hEvent, ///< [in] handle of the event object
    ur_profiling_info_t
        propName,    ///< [in] the name of the profiling property to query
    size_t propSize, ///< [in] size in bytes of the profiling property value
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the profiling
                    ///< property
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes returned in
                     ///< propValue
) {
    auto pfnGetProfilingInfo = context.urDdiTable.Event.pfnGetProfilingInfo;

    if (nullptr == pfnGetProfilingInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hEvent) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (UR_PROFILING_INFO_COMMAND_COMPLETE < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (pPropValue && propSize == 0) {
            return UR_RESULT_ERROR_INVALID_VALUE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hEvent)) {
        refCountContext.logInvalidReference(hEvent);
    }

    ur_result_t result = pfnGetProfilingInfo(hEvent, propName, propSize,
                                             pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventWait
__urdlllocal ur_result_t UR_APICALL urEventWait(
    uint32_t numEvents, ///< [in] number of events in the event list
    const ur_event_handle_t *
        phEventWaitList ///< [in][range(0, numEvents)] pointer to a list of events to wait for
                        ///< completion
) {
    auto pfnWait = context.urDdiTable.Event.pfnWait;

    if (nullptr == pfnWait) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == phEventWaitList) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (numEvents == 0) {
            return UR_RESULT_ERROR_INVALID_VALUE;
        }
    }

    ur_result_t result = pfnWait(numEvents, phEventWaitList);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRetain
__urdlllocal ur_result_t UR_APICALL urEventRetain(
    ur_event_handle_t hEvent ///< [in] handle of the event object
) {
    auto pfnRetain = context.urDdiTable.Event.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hEvent) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRetain(hEvent);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.incrementRefCount(hEvent, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRelease
__urdlllocal ur_result_t UR_APICALL urEventRelease(
    ur_event_handle_t hEvent ///< [in] handle of the event object
) {
    auto pfnRelease = context.urDdiTable.Event.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hEvent) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRelease(hEvent);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.decrementRefCount(hEvent, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urEventGetNativeHandle(
    ur_event_handle_t hEvent, ///< [in] handle of the event.
    ur_native_handle_t
        *phNativeEvent ///< [out] a pointer to the native handle of the event.
) {
    auto pfnGetNativeHandle = context.urDdiTable.Event.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hEvent) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phNativeEvent) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hEvent)) {
        refCountContext.logInvalidReference(hEvent);
    }

    ur_result_t result = pfnGetNativeHandle(hEvent, phNativeEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urEventCreateWithNativeHandle(
    ur_native_handle_t
        hNativeEvent, ///< [in][nocheck] the native handle of the event.
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const ur_event_native_properties_t *
        pProperties, ///< [in][optional] pointer to native event properties struct
    ur_event_handle_t
        *phEvent ///< [out] pointer to the handle of the event object created.
) {
    auto pfnCreateWithNativeHandle =
        context.urDdiTable.Event.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phEvent) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result =
        pfnCreateWithNativeHandle(hNativeEvent, hContext, pProperties, phEvent);

    if (context.enableLeakChecking && result == UR_RESULT_SUCCESS) {
        refCountContext.createRefCount(*phEvent);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventSetCallback
__urdlllocal ur_result_t UR_APICALL urEventSetCallback(
    ur_event_handle_t hEvent,       ///< [in] handle of the event object
    ur_execution_info_t execStatus, ///< [in] execution status of the event
    ur_event_callback_t pfnNotify,  ///< [in] execution status of the event
    void *
        pUserData ///< [in][out][optional] pointer to data to be passed to callback.
) {
    auto pfnSetCallback = context.urDdiTable.Event.pfnSetCallback;

    if (nullptr == pfnSetCallback) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hEvent) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pfnNotify) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_EXECUTION_INFO_QUEUED < execStatus) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (execStatus == UR_EXECUTION_INFO_QUEUED) {
            return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hEvent)) {
        refCountContext.logInvalidReference(hEvent);
    }

    ur_result_t result =
        pfnSetCallback(hEvent, execStatus, pfnNotify, pUserData);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueKernelLaunch
__urdlllocal ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue,   ///< [in] handle of the queue object
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t
        workDim, ///< [in] number of dimensions, from 1 to 3, to specify the global and
                 ///< work-group work-items
    const size_t *
        pGlobalWorkOffset, ///< [in] pointer to an array of workDim unsigned values that specify the
    ///< offset used to calculate the global ID of a work-item
    const size_t *
        pGlobalWorkSize, ///< [in] pointer to an array of workDim unsigned values that specify the
    ///< number of global work-items in workDim that will execute the kernel
    ///< function
    const size_t *
        pLocalWorkSize, ///< [in][optional] pointer to an array of workDim unsigned values that
    ///< specify the number of local work-items forming a work-group that will
    ///< execute the kernel function.
    ///< If nullptr, the runtime implementation will choose the work-group
    ///< size.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the kernel execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    ///< event.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< kernel execution instance.
) {
    auto pfnKernelLaunch = context.urDdiTable.Enqueue.pfnKernelLaunch;

    if (nullptr == pfnKernelLaunch) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pGlobalWorkOffset) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pGlobalWorkSize) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    ur_result_t result = pfnKernelLaunch(
        hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
        pLocalWorkSize, numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueEventsWait
__urdlllocal ur_result_t UR_APICALL urEnqueueEventsWait(
    ur_queue_handle_t hQueue,     ///< [in] handle of the queue object
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
    ///< previously enqueued commands
    ///< must be complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnEventsWait = context.urDdiTable.Enqueue.pfnEventsWait;

    if (nullptr == pfnEventsWait) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result =
        pfnEventsWait(hQueue, numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueEventsWaitWithBarrier
__urdlllocal ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue,     ///< [in] handle of the queue object
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
    ///< previously enqueued commands
    ///< must be complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnEventsWaitWithBarrier =
        context.urDdiTable.Enqueue.pfnEventsWaitWithBarrier;

    if (nullptr == pfnEventsWaitWithBarrier) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result = pfnEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                                  phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferRead
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(offset, size)] handle of the buffer object
    bool blockingRead, ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,     ///< [in] offset in bytes in the buffer object
    size_t size,       ///< [in] size in bytes of data being read
    void *pDst, ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferRead = context.urDdiTable.Enqueue.pfnMemBufferRead;

    if (nullptr == pfnMemBufferRead) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pDst) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (auto boundsError = bounds(hBuffer, offset, size);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBuffer)) {
        refCountContext.logInvalidReference(hBuffer);
    }

    ur_result_t result =
        pfnMemBufferRead(hQueue, hBuffer, blockingRead, offset, size, pDst,
                         numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferWrite
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(offset, size)] handle of the buffer object
    bool
        blockingWrite, ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,     ///< [in] offset in bytes in the buffer object
    size_t size,       ///< [in] size in bytes of data being written
    const void
        *pSrc, ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferWrite = context.urDdiTable.Enqueue.pfnMemBufferWrite;

    if (nullptr == pfnMemBufferWrite) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (auto boundsError = bounds(hBuffer, offset, size);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBuffer)) {
        refCountContext.logInvalidReference(hBuffer);
    }

    ur_result_t result =
        pfnMemBufferWrite(hQueue, hBuffer, blockingWrite, offset, size, pSrc,
                          numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferReadRect
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(bufferOrigin, region)] handle of the buffer object
    bool blockingRead, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOrigin, ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOrigin,   ///< [in] 3D offset in the host region
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t
        bufferRowPitch, ///< [in] length of each row in bytes in the buffer object
    size_t
        bufferSlicePitch, ///< [in] length of each 2D slice in bytes in the buffer object being read
    size_t
        hostRowPitch, ///< [in] length of each row in bytes in the host memory region pointed by
                      ///< dst
    size_t
        hostSlicePitch, ///< [in] length of each 2D slice in bytes in the host memory region
                        ///< pointed by dst
    void *pDst, ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferReadRect = context.urDdiTable.Enqueue.pfnMemBufferReadRect;

    if (nullptr == pfnMemBufferReadRect) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pDst) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (region.width == 0 || region.height == 0 || region.width == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (bufferRowPitch != 0 && bufferRowPitch < region.width) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (hostRowPitch != 0 && hostRowPitch < region.width) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (bufferSlicePitch != 0 &&
            bufferSlicePitch <
                region.height *
                    (bufferRowPitch != 0 ? bufferRowPitch : region.width)) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (bufferSlicePitch != 0 &&
            bufferSlicePitch %
                    (bufferRowPitch != 0 ? bufferRowPitch : region.width) !=
                0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (hostSlicePitch != 0 &&
            hostSlicePitch <
                region.height *
                    (hostRowPitch != 0 ? hostRowPitch : region.width)) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (hostSlicePitch != 0 &&
            hostSlicePitch %
                    (hostRowPitch != 0 ? hostRowPitch : region.width) !=
                0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (auto boundsError = bounds(hBuffer, bufferOrigin, region);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBuffer)) {
        refCountContext.logInvalidReference(hBuffer);
    }

    ur_result_t result = pfnMemBufferReadRect(
        hQueue, hBuffer, blockingRead, bufferOrigin, hostOrigin, region,
        bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst,
        numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferWriteRect
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(bufferOrigin, region)] handle of the buffer object
    bool
        blockingWrite, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOrigin, ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOrigin,   ///< [in] 3D offset in the host region
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t
        bufferRowPitch, ///< [in] length of each row in bytes in the buffer object
    size_t
        bufferSlicePitch, ///< [in] length of each 2D slice in bytes in the buffer object being
                          ///< written
    size_t
        hostRowPitch, ///< [in] length of each row in bytes in the host memory region pointed by
                      ///< src
    size_t
        hostSlicePitch, ///< [in] length of each 2D slice in bytes in the host memory region
                        ///< pointed by src
    void
        *pSrc, ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] points to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferWriteRect =
        context.urDdiTable.Enqueue.pfnMemBufferWriteRect;

    if (nullptr == pfnMemBufferWriteRect) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (region.width == 0 || region.height == 0 || region.width == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (bufferRowPitch != 0 && bufferRowPitch < region.width) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (hostRowPitch != 0 && hostRowPitch < region.width) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (bufferSlicePitch != 0 &&
            bufferSlicePitch <
                region.height *
                    (bufferRowPitch != 0 ? bufferRowPitch : region.width)) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (bufferSlicePitch != 0 &&
            bufferSlicePitch %
                    (bufferRowPitch != 0 ? bufferRowPitch : region.width) !=
                0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (hostSlicePitch != 0 &&
            hostSlicePitch <
                region.height *
                    (hostRowPitch != 0 ? hostRowPitch : region.width)) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (hostSlicePitch != 0 &&
            hostSlicePitch %
                    (hostRowPitch != 0 ? hostRowPitch : region.width) !=
                0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (auto boundsError = bounds(hBuffer, bufferOrigin, region);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBuffer)) {
        refCountContext.logInvalidReference(hBuffer);
    }

    ur_result_t result = pfnMemBufferWriteRect(
        hQueue, hBuffer, blockingWrite, bufferOrigin, hostOrigin, region,
        bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc,
        numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferCopy
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferCopy(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBufferSrc, ///< [in][bounds(srcOffset, size)] handle of the src buffer object
    ur_mem_handle_t
        hBufferDst, ///< [in][bounds(dstOffset, size)] handle of the dest buffer object
    size_t srcOffset, ///< [in] offset into hBufferSrc to begin copying from
    size_t dstOffset, ///< [in] offset info hBufferDst to begin copying into
    size_t size,      ///< [in] size in bytes of data being copied
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferCopy = context.urDdiTable.Enqueue.pfnMemBufferCopy;

    if (nullptr == pfnMemBufferCopy) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBufferSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBufferDst) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (auto boundsError = bounds(hBufferSrc, srcOffset, size);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (auto boundsError = bounds(hBufferDst, dstOffset, size);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBufferSrc)) {
        refCountContext.logInvalidReference(hBufferSrc);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBufferDst)) {
        refCountContext.logInvalidReference(hBufferDst);
    }

    ur_result_t result =
        pfnMemBufferCopy(hQueue, hBufferSrc, hBufferDst, srcOffset, dstOffset,
                         size, numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferCopyRect
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBufferSrc, ///< [in][bounds(srcOrigin, region)] handle of the source buffer object
    ur_mem_handle_t
        hBufferDst, ///< [in][bounds(dstOrigin, region)] handle of the dest buffer object
    ur_rect_offset_t srcOrigin, ///< [in] 3D offset in the source buffer
    ur_rect_offset_t dstOrigin, ///< [in] 3D offset in the destination buffer
    ur_rect_region_t
        region, ///< [in] source 3D rectangular region descriptor: width, height, depth
    size_t
        srcRowPitch, ///< [in] length of each row in bytes in the source buffer object
    size_t
        srcSlicePitch, ///< [in] length of each 2D slice in bytes in the source buffer object
    size_t
        dstRowPitch, ///< [in] length of each row in bytes in the destination buffer object
    size_t
        dstSlicePitch, ///< [in] length of each 2D slice in bytes in the destination buffer object
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferCopyRect = context.urDdiTable.Enqueue.pfnMemBufferCopyRect;

    if (nullptr == pfnMemBufferCopyRect) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBufferSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBufferDst) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (region.width == 0 || region.height == 0 || region.depth == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (srcRowPitch != 0 && srcRowPitch < region.width) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (dstRowPitch != 0 && dstRowPitch < region.width) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (srcSlicePitch != 0 &&
            srcSlicePitch < region.height * (srcRowPitch != 0 ? srcRowPitch
                                                              : region.width)) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (srcSlicePitch != 0 &&
            srcSlicePitch % (srcRowPitch != 0 ? srcRowPitch : region.width) !=
                0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (dstSlicePitch != 0 &&
            dstSlicePitch < region.height * (dstRowPitch != 0 ? dstRowPitch
                                                              : region.width)) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (dstSlicePitch != 0 &&
            dstSlicePitch % (dstRowPitch != 0 ? dstRowPitch : region.width) !=
                0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (auto boundsError = bounds(hBufferSrc, srcOrigin, region);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (auto boundsError = bounds(hBufferDst, dstOrigin, region);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBufferSrc)) {
        refCountContext.logInvalidReference(hBufferSrc);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBufferDst)) {
        refCountContext.logInvalidReference(hBufferDst);
    }

    ur_result_t result = pfnMemBufferCopyRect(
        hQueue, hBufferSrc, hBufferDst, srcOrigin, dstOrigin, region,
        srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch,
        numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferFill
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(offset, size)] handle of the buffer object
    const void *pPattern, ///< [in] pointer to the fill pattern
    size_t patternSize,   ///< [in] size in bytes of the pattern
    size_t offset,        ///< [in] offset into the buffer
    size_t size, ///< [in] fill size in bytes, must be a multiple of patternSize
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferFill = context.urDdiTable.Enqueue.pfnMemBufferFill;

    if (nullptr == pfnMemBufferFill) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pPattern) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (patternSize == 0 || size == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (patternSize > size) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if ((patternSize & (patternSize - 1)) != 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (size % patternSize != 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (offset % patternSize != 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (auto boundsError = bounds(hBuffer, offset, size);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBuffer)) {
        refCountContext.logInvalidReference(hBuffer);
    }

    ur_result_t result =
        pfnMemBufferFill(hQueue, hBuffer, pPattern, patternSize, offset, size,
                         numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemImageRead
__urdlllocal ur_result_t UR_APICALL urEnqueueMemImageRead(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hImage, ///< [in][bounds(origin, region)] handle of the image object
    bool blockingRead, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t
        origin, ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_region_t
        region, ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                ///< image
    size_t rowPitch,   ///< [in] length of each row in bytes
    size_t slicePitch, ///< [in] length of each 2D slice of the 3D image
    void *pDst, ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemImageRead = context.urDdiTable.Enqueue.pfnMemImageRead;

    if (nullptr == pfnMemImageRead) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hImage) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pDst) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (region.width == 0 || region.height == 0 || region.depth == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (auto boundsError = boundsImage(hImage, origin, region);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hImage)) {
        refCountContext.logInvalidReference(hImage);
    }

    ur_result_t result = pfnMemImageRead(
        hQueue, hImage, blockingRead, origin, region, rowPitch, slicePitch,
        pDst, numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemImageWrite
__urdlllocal ur_result_t UR_APICALL urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hImage, ///< [in][bounds(origin, region)] handle of the image object
    bool
        blockingWrite, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t
        origin, ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_region_t
        region, ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                ///< image
    size_t rowPitch,   ///< [in] length of each row in bytes
    size_t slicePitch, ///< [in] length of each 2D slice of the 3D image
    void *pSrc, ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemImageWrite = context.urDdiTable.Enqueue.pfnMemImageWrite;

    if (nullptr == pfnMemImageWrite) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hImage) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (region.width == 0 || region.height == 0 || region.depth == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (auto boundsError = boundsImage(hImage, origin, region);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hImage)) {
        refCountContext.logInvalidReference(hImage);
    }

    ur_result_t result = pfnMemImageWrite(
        hQueue, hImage, blockingWrite, origin, region, rowPitch, slicePitch,
        pSrc, numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemImageCopy
__urdlllocal ur_result_t UR_APICALL urEnqueueMemImageCopy(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hImageSrc, ///< [in][bounds(srcOrigin, region)] handle of the src image object
    ur_mem_handle_t
        hImageDst, ///< [in][bounds(dstOrigin, region)] handle of the dest image object
    ur_rect_offset_t
        srcOrigin, ///< [in] defines the (x,y,z) offset in pixels in the source 1D, 2D, or 3D
                   ///< image
    ur_rect_offset_t
        dstOrigin, ///< [in] defines the (x,y,z) offset in pixels in the destination 1D, 2D,
                   ///< or 3D image
    ur_rect_region_t
        region, ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                ///< image
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemImageCopy = context.urDdiTable.Enqueue.pfnMemImageCopy;

    if (nullptr == pfnMemImageCopy) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hImageSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hImageDst) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (region.width == 0 || region.height == 0 || region.depth == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (auto boundsError = boundsImage(hImageSrc, srcOrigin, region);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (auto boundsError = boundsImage(hImageDst, dstOrigin, region);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hImageSrc)) {
        refCountContext.logInvalidReference(hImageSrc);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hImageDst)) {
        refCountContext.logInvalidReference(hImageDst);
    }

    ur_result_t result =
        pfnMemImageCopy(hQueue, hImageSrc, hImageDst, srcOrigin, dstOrigin,
                        region, numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferMap
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(offset, size)] handle of the buffer object
    bool blockingMap, ///< [in] indicates blocking (true), non-blocking (false)
    ur_map_flags_t mapFlags, ///< [in] flags for read, write, readwrite mapping
    size_t offset, ///< [in] offset in bytes of the buffer region being mapped
    size_t size,   ///< [in] size in bytes of the buffer region being mapped
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent, ///< [out][optional] return an event object that identifies this particular
                 ///< command instance.
    void **ppRetMap ///< [out] return mapped pointer.  TODO: move it before
                    ///< numEventsInWaitList?
) {
    auto pfnMemBufferMap = context.urDdiTable.Enqueue.pfnMemBufferMap;

    if (nullptr == pfnMemBufferMap) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == ppRetMap) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_MAP_FLAGS_MASK & mapFlags) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (auto boundsError = bounds(hBuffer, offset, size);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBuffer)) {
        refCountContext.logInvalidReference(hBuffer);
    }

    ur_result_t result = pfnMemBufferMap(hQueue, hBuffer, blockingMap, mapFlags,
                                         offset, size, numEventsInWaitList,
                                         phEventWaitList, phEvent, ppRetMap);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemUnmap
__urdlllocal ur_result_t UR_APICALL urEnqueueMemUnmap(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hMem,         ///< [in] handle of the memory (buffer or image) object
    void *pMappedPtr, ///< [in] mapped host address
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemUnmap = context.urDdiTable.Enqueue.pfnMemUnmap;

    if (nullptr == pfnMemUnmap) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pMappedPtr) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hMem)) {
        refCountContext.logInvalidReference(hMem);
    }

    ur_result_t result =
        pfnMemUnmap(hQueue, hMem, pMappedPtr, numEventsInWaitList,
                    phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMFill
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMFill(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    void *pMem, ///< [in][bounds(0, size)] pointer to USM memory object
    size_t
        patternSize, ///< [in] the size in bytes of the pattern. Must be a power of 2 and less
                     ///< than or equal to width.
    const void
        *pPattern, ///< [in] pointer with the bytes of the pattern to set.
    size_t
        size, ///< [in] size in bytes to be set. Must be a multiple of patternSize.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnUSMFill = context.urDdiTable.Enqueue.pfnUSMFill;

    if (nullptr == pfnUSMFill) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pPattern) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (patternSize == 0 || size == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (patternSize > size) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if ((patternSize & (patternSize - 1)) != 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (size % patternSize != 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (auto boundsError = bounds(hQueue, pMem, 0, size);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result =
        pfnUSMFill(hQueue, pMem, patternSize, pPattern, size,
                   numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMMemcpy
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMMemcpy(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    bool blocking,            ///< [in] blocking or non-blocking copy
    void *
        pDst, ///< [in][bounds(0, size)] pointer to the destination USM memory object
    const void *
        pSrc, ///< [in][bounds(0, size)] pointer to the source USM memory object
    size_t size,                  ///< [in] size in bytes to be copied
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnUSMMemcpy = context.urDdiTable.Enqueue.pfnUSMMemcpy;

    if (nullptr == pfnUSMMemcpy) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pDst) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (size == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (auto boundsError = bounds(hQueue, pDst, 0, size);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (auto boundsError = bounds(hQueue, pSrc, 0, size);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result =
        pfnUSMMemcpy(hQueue, blocking, pDst, pSrc, size, numEventsInWaitList,
                     phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMPrefetch
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMPrefetch(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    const void
        *pMem,   ///< [in][bounds(0, size)] pointer to the USM memory object
    size_t size, ///< [in] size in bytes to be fetched
    ur_usm_migration_flags_t flags, ///< [in] USM prefetch flags
    uint32_t numEventsInWaitList,   ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnUSMPrefetch = context.urDdiTable.Enqueue.pfnUSMPrefetch;

    if (nullptr == pfnUSMPrefetch) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_USM_MIGRATION_FLAGS_MASK & flags) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (size == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (auto boundsError = bounds(hQueue, pMem, 0, size);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result =
        pfnUSMPrefetch(hQueue, pMem, size, flags, numEventsInWaitList,
                       phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMAdvise
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMAdvise(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    const void
        *pMem,   ///< [in][bounds(0, size)] pointer to the USM memory object
    size_t size, ///< [in] size in bytes to be advised
    ur_usm_advice_flags_t advice, ///< [in] USM memory advice
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnUSMAdvise = context.urDdiTable.Enqueue.pfnUSMAdvise;

    if (nullptr == pfnUSMAdvise) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_USM_ADVICE_FLAGS_MASK & advice) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (size == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (auto boundsError = bounds(hQueue, pMem, 0, size);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result = pfnUSMAdvise(hQueue, pMem, size, advice, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMFill2D
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMFill2D(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue to submit to.
    void *
        pMem, ///< [in][bounds(0, pitch * height)] pointer to memory to be filled.
    size_t
        pitch, ///< [in] the total width of the destination memory including padding.
    size_t
        patternSize, ///< [in] the size in bytes of the pattern. Must be a power of 2 and less
                     ///< than or equal to width.
    const void
        *pPattern, ///< [in] pointer with the bytes of the pattern to set.
    size_t
        width, ///< [in] the width in bytes of each row to fill. Must be a multiple of
               ///< patternSize.
    size_t height,                ///< [in] the height of the columns to fill.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the kernel execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    ///< event.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< kernel execution instance.
) {
    auto pfnUSMFill2D = context.urDdiTable.Enqueue.pfnUSMFill2D;

    if (nullptr == pfnUSMFill2D) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pPattern) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pitch == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (pitch < width) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (patternSize == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (patternSize > width * height) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (patternSize != 0 && ((patternSize & (patternSize - 1)) != 0)) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (width == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (height == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (width * height % patternSize != 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (auto boundsError = bounds(hQueue, pMem, 0, pitch * height);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result =
        pfnUSMFill2D(hQueue, pMem, pitch, patternSize, pPattern, width, height,
                     numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMMemcpy2D
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMMemcpy2D(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue to submit to.
    bool blocking, ///< [in] indicates if this operation should block the host.
    void *
        pDst, ///< [in][bounds(0, dstPitch * height)] pointer to memory where data will
              ///< be copied.
    size_t
        dstPitch, ///< [in] the total width of the source memory including padding.
    const void *
        pSrc, ///< [in][bounds(0, srcPitch * height)] pointer to memory to be copied.
    size_t
        srcPitch, ///< [in] the total width of the source memory including padding.
    size_t width,  ///< [in] the width in bytes of each row to be copied.
    size_t height, ///< [in] the height of columns to be copied.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the kernel execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    ///< event.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< kernel execution instance.
) {
    auto pfnUSMMemcpy2D = context.urDdiTable.Enqueue.pfnUSMMemcpy2D;

    if (nullptr == pfnUSMMemcpy2D) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pDst) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (srcPitch == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (dstPitch == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (srcPitch < width) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (dstPitch < width) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (height == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (auto boundsError = bounds(hQueue, pDst, 0, dstPitch * height);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (auto boundsError = bounds(hQueue, pSrc, 0, srcPitch * height);
            boundsError != UR_RESULT_SUCCESS) {
            return boundsError;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result =
        pfnUSMMemcpy2D(hQueue, blocking, pDst, dstPitch, pSrc, srcPitch, width,
                       height, numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueDeviceGlobalVariableWrite
__urdlllocal ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue to submit to.
    ur_program_handle_t
        hProgram, ///< [in] handle of the program containing the device global variable.
    const char
        *name, ///< [in] the unique identifier for the device global variable.
    bool blockingWrite, ///< [in] indicates if this operation should block.
    size_t count,       ///< [in] the number of bytes to copy.
    size_t
        offset, ///< [in] the byte offset into the device global variable to start copying.
    const void *pSrc, ///< [in] pointer to where the data must be copied from.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list.
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the kernel execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    ///< event.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< kernel execution instance.
) {
    auto pfnDeviceGlobalVariableWrite =
        context.urDdiTable.Enqueue.pfnDeviceGlobalVariableWrite;

    if (nullptr == pfnDeviceGlobalVariableWrite) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == name) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result = pfnDeviceGlobalVariableWrite(
        hQueue, hProgram, name, blockingWrite, count, offset, pSrc,
        numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueDeviceGlobalVariableRead
__urdlllocal ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue to submit to.
    ur_program_handle_t
        hProgram, ///< [in] handle of the program containing the device global variable.
    const char
        *name, ///< [in] the unique identifier for the device global variable.
    bool blockingRead, ///< [in] indicates if this operation should block.
    size_t count,      ///< [in] the number of bytes to copy.
    size_t
        offset, ///< [in] the byte offset into the device global variable to start copying.
    void *pDst, ///< [in] pointer to where the data must be copied to.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list.
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the kernel execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    ///< event.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< kernel execution instance.
) {
    auto pfnDeviceGlobalVariableRead =
        context.urDdiTable.Enqueue.pfnDeviceGlobalVariableRead;

    if (nullptr == pfnDeviceGlobalVariableRead) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == name) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pDst) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result = pfnDeviceGlobalVariableRead(
        hQueue, hProgram, name, blockingRead, count, offset, pDst,
        numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueReadHostPipe
__urdlllocal ur_result_t UR_APICALL urEnqueueReadHostPipe(
    ur_queue_handle_t
        hQueue, ///< [in] a valid host command-queue in which the read command
    ///< will be queued. hQueue and hProgram must be created with the same
    ///< UR context.
    ur_program_handle_t
        hProgram, ///< [in] a program object with a successfully built executable.
    const char *
        pipe_symbol, ///< [in] the name of the program scope pipe global variable.
    bool
        blocking, ///< [in] indicate if the read operation is blocking or non-blocking.
    void *
        pDst, ///< [in] a pointer to buffer in host memory that will hold resulting data
              ///< from pipe.
    size_t size, ///< [in] size of the memory region to read, in bytes.
    uint32_t numEventsInWaitList, ///< [in] number of events in the wait list.
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the host pipe read.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait event.
    ur_event_handle_t *
        phEvent ///< [out][optional] returns an event object that identifies this read
                ///< command
    ///< and can be used to query or queue a wait for this command to complete.
) {
    auto pfnReadHostPipe = context.urDdiTable.Enqueue.pfnReadHostPipe;

    if (nullptr == pfnReadHostPipe) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pipe_symbol) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pDst) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result =
        pfnReadHostPipe(hQueue, hProgram, pipe_symbol, blocking, pDst, size,
                        numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueWriteHostPipe
__urdlllocal ur_result_t UR_APICALL urEnqueueWriteHostPipe(
    ur_queue_handle_t
        hQueue, ///< [in] a valid host command-queue in which the write command
    ///< will be queued. hQueue and hProgram must be created with the same
    ///< UR context.
    ur_program_handle_t
        hProgram, ///< [in] a program object with a successfully built executable.
    const char *
        pipe_symbol, ///< [in] the name of the program scope pipe global variable.
    bool
        blocking, ///< [in] indicate if the read and write operations are blocking or
                  ///< non-blocking.
    void *
        pSrc, ///< [in] a pointer to buffer in host memory that holds data to be written
              ///< to the host pipe.
    size_t size, ///< [in] size of the memory region to read or write, in bytes.
    uint32_t numEventsInWaitList, ///< [in] number of events in the wait list.
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the host pipe write.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait event.
    ur_event_handle_t *
        phEvent ///< [out][optional] returns an event object that identifies this write command
    ///< and can be used to query or queue a wait for this command to complete.
) {
    auto pfnWriteHostPipe = context.urDdiTable.Enqueue.pfnWriteHostPipe;

    if (nullptr == pfnWriteHostPipe) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pipe_symbol) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result =
        pfnWriteHostPipe(hQueue, hProgram, pipe_symbol, blocking, pSrc, size,
                         numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPitchedAllocExp
__urdlllocal ur_result_t UR_APICALL urUSMPitchedAllocExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_usm_desc_t *
        pUSMDesc, ///< [in][optional] Pointer to USM memory allocation descriptor.
    ur_usm_pool_handle_t
        pool, ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t
        widthInBytes, ///< [in] width in bytes of the USM memory object to be allocated
    size_t height, ///< [in] height of the USM memory object to be allocated
    size_t
        elementSizeBytes, ///< [in] size in bytes of an element in the allocation
    void **ppMem,         ///< [out] pointer to USM shared memory object
    size_t *pResultPitch  ///< [out] pitch of the allocation
) {
    auto pfnPitchedAllocExp = context.urDdiTable.USMExp.pfnPitchedAllocExp;

    if (nullptr == pfnPitchedAllocExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == ppMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pResultPitch) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL != pUSMDesc && UR_USM_ADVICE_FLAGS_MASK & pUSMDesc->hints) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (pUSMDesc && pUSMDesc->align != 0 &&
            ((pUSMDesc->align & (pUSMDesc->align - 1)) != 0)) {
            return UR_RESULT_ERROR_INVALID_VALUE;
        }

        if (widthInBytes == 0) {
            return UR_RESULT_ERROR_INVALID_USM_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(pool)) {
        refCountContext.logInvalidReference(pool);
    }

    ur_result_t result =
        pfnPitchedAllocExp(hContext, hDevice, pUSMDesc, pool, widthInBytes,
                           height, elementSizeBytes, ppMem, pResultPitch);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesUnsampledImageHandleDestroyExp
__urdlllocal ur_result_t UR_APICALL
urBindlessImagesUnsampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_handle_t
        hImage ///< [in] pointer to handle of image object to destroy
) {
    auto pfnUnsampledImageHandleDestroyExp =
        context.urDdiTable.BindlessImagesExp.pfnUnsampledImageHandleDestroyExp;

    if (nullptr == pfnUnsampledImageHandleDestroyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hImage) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result =
        pfnUnsampledImageHandleDestroyExp(hContext, hDevice, hImage);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesSampledImageHandleDestroyExp
__urdlllocal ur_result_t UR_APICALL
urBindlessImagesSampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_handle_t
        hImage ///< [in] pointer to handle of image object to destroy
) {
    auto pfnSampledImageHandleDestroyExp =
        context.urDdiTable.BindlessImagesExp.pfnSampledImageHandleDestroyExp;

    if (nullptr == pfnSampledImageHandleDestroyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hImage) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result =
        pfnSampledImageHandleDestroyExp(hContext, hDevice, hImage);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImageAllocateExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImageAllocateExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description
    ur_exp_image_mem_handle_t
        *phImageMem ///< [out] pointer to handle of image memory allocated
) {
    auto pfnImageAllocateExp =
        context.urDdiTable.BindlessImagesExp.pfnImageAllocateExp;

    if (nullptr == pfnImageAllocateExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pImageFormat) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pImageDesc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phImageMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type) {
            return UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnImageAllocateExp(hContext, hDevice, pImageFormat,
                                             pImageDesc, phImageMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImageFreeExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImageFreeExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_mem_handle_t
        hImageMem ///< [in] handle of image memory to be freed
) {
    auto pfnImageFreeExp = context.urDdiTable.BindlessImagesExp.pfnImageFreeExp;

    if (nullptr == pfnImageFreeExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hImageMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnImageFreeExp(hContext, hDevice, hImageMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesUnsampledImageCreateExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesUnsampledImageCreateExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_mem_handle_t
        hImageMem, ///< [in] handle to memory from which to create the image
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description
    ur_mem_handle_t *phMem, ///< [out] pointer to handle of image object created
    ur_exp_image_handle_t
        *phImage ///< [out] pointer to handle of image object created
) {
    auto pfnUnsampledImageCreateExp =
        context.urDdiTable.BindlessImagesExp.pfnUnsampledImageCreateExp;

    if (nullptr == pfnUnsampledImageCreateExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hImageMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pImageFormat) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pImageDesc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phImage) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type) {
            return UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnUnsampledImageCreateExp(
        hContext, hDevice, hImageMem, pImageFormat, pImageDesc, phMem, phImage);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesSampledImageCreateExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesSampledImageCreateExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_mem_handle_t
        hImageMem, ///< [in] handle to memory from which to create the image
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description
    ur_sampler_handle_t hSampler,      ///< [in] sampler to be used
    ur_mem_handle_t *phMem, ///< [out] pointer to handle of image object created
    ur_exp_image_handle_t
        *phImage ///< [out] pointer to handle of image object created
) {
    auto pfnSampledImageCreateExp =
        context.urDdiTable.BindlessImagesExp.pfnSampledImageCreateExp;

    if (nullptr == pfnSampledImageCreateExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hImageMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hSampler) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pImageFormat) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pImageDesc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phImage) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type) {
            return UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hSampler)) {
        refCountContext.logInvalidReference(hSampler);
    }

    ur_result_t result =
        pfnSampledImageCreateExp(hContext, hDevice, hImageMem, pImageFormat,
                                 pImageDesc, hSampler, phMem, phImage);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImageCopyExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImageCopyExp(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    void *pDst,               ///< [in] location the data will be copied to
    void *pSrc,               ///< [in] location the data will be copied from
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description
    ur_exp_image_copy_flags_t
        imageCopyFlags, ///< [in] flags describing copy direction e.g. H2D or D2H
    ur_rect_offset_t
        srcOffset, ///< [in] defines the (x,y,z) source offset in pixels in the 1D, 2D, or 3D
                   ///< image
    ur_rect_offset_t
        dstOffset, ///< [in] defines the (x,y,z) destination offset in pixels in the 1D, 2D,
                   ///< or 3D image
    ur_rect_region_t
        copyExtent, ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                    ///< region to copy
    ur_rect_region_t
        hostExtent, ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                    ///< region on the host
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
    ///< previously enqueued commands
    ///< must be complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnImageCopyExp = context.urDdiTable.BindlessImagesExp.pfnImageCopyExp;

    if (nullptr == pfnImageCopyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pDst) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pImageFormat) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pImageDesc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_EXP_IMAGE_COPY_FLAGS_MASK & imageCopyFlags) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type) {
            return UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result = pfnImageCopyExp(
        hQueue, pDst, pSrc, pImageFormat, pImageDesc, imageCopyFlags, srcOffset,
        dstOffset, copyExtent, hostExtent, numEventsInWaitList, phEventWaitList,
        phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImageGetInfoExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImageGetInfoExp(
    ur_exp_image_mem_handle_t hImageMem, ///< [in] handle to the image memory
    ur_image_info_t propName,            ///< [in] queried info name
    void *pPropValue,    ///< [out][optional] returned query value
    size_t *pPropSizeRet ///< [out][optional] returned query value size
) {
    auto pfnImageGetInfoExp =
        context.urDdiTable.BindlessImagesExp.pfnImageGetInfoExp;

    if (nullptr == pfnImageGetInfoExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hImageMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_IMAGE_INFO_DEPTH < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }
    }

    ur_result_t result =
        pfnImageGetInfoExp(hImageMem, propName, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesMipmapGetLevelExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesMipmapGetLevelExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_mem_handle_t
        hImageMem,        ///< [in] memory handle to the mipmap image
    uint32_t mipmapLevel, ///< [in] requested level of the mipmap
    ur_exp_image_mem_handle_t
        *phImageMem ///< [out] returning memory handle to the individual image
) {
    auto pfnMipmapGetLevelExp =
        context.urDdiTable.BindlessImagesExp.pfnMipmapGetLevelExp;

    if (nullptr == pfnMipmapGetLevelExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hImageMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phImageMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnMipmapGetLevelExp(hContext, hDevice, hImageMem,
                                              mipmapLevel, phImageMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesMipmapFreeExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesMipmapFreeExp(
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    ur_device_handle_t hDevice,    ///< [in] handle of the device object
    ur_exp_image_mem_handle_t hMem ///< [in] handle of image memory to be freed
) {
    auto pfnMipmapFreeExp =
        context.urDdiTable.BindlessImagesExp.pfnMipmapFreeExp;

    if (nullptr == pfnMipmapFreeExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnMipmapFreeExp(hContext, hDevice, hMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImportOpaqueFDExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImportOpaqueFDExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    size_t size,                  ///< [in] size of the external memory
    ur_exp_interop_mem_desc_t
        *pInteropMemDesc, ///< [in] the interop memory descriptor
    ur_exp_interop_mem_handle_t
        *phInteropMem ///< [out] interop memory handle to the external memory
) {
    auto pfnImportOpaqueFDExp =
        context.urDdiTable.BindlessImagesExp.pfnImportOpaqueFDExp;

    if (nullptr == pfnImportOpaqueFDExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pInteropMemDesc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phInteropMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnImportOpaqueFDExp(hContext, hDevice, size,
                                              pInteropMemDesc, phInteropMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesMapExternalArrayExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesMapExternalArrayExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description
    ur_exp_interop_mem_handle_t
        hInteropMem, ///< [in] interop memory handle to the external memory
    ur_exp_image_mem_handle_t *
        phImageMem ///< [out] image memory handle to the externally allocated memory
) {
    auto pfnMapExternalArrayExp =
        context.urDdiTable.BindlessImagesExp.pfnMapExternalArrayExp;

    if (nullptr == pfnMapExternalArrayExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hInteropMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pImageFormat) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pImageDesc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phImageMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type) {
            return UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnMapExternalArrayExp(
        hContext, hDevice, pImageFormat, pImageDesc, hInteropMem, phImageMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesReleaseInteropExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesReleaseInteropExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_interop_mem_handle_t
        hInteropMem ///< [in] handle of interop memory to be freed
) {
    auto pfnReleaseInteropExp =
        context.urDdiTable.BindlessImagesExp.pfnReleaseInteropExp;

    if (nullptr == pfnReleaseInteropExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hInteropMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnReleaseInteropExp(hContext, hDevice, hInteropMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImportExternalSemaphoreOpaqueFDExp
__urdlllocal ur_result_t UR_APICALL
urBindlessImagesImportExternalSemaphoreOpaqueFDExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_interop_semaphore_desc_t
        *pInteropSemaphoreDesc, ///< [in] the interop semaphore descriptor
    ur_exp_interop_semaphore_handle_t *
        phInteropSemaphore ///< [out] interop semaphore handle to the external semaphore
) {
    auto pfnImportExternalSemaphoreOpaqueFDExp =
        context.urDdiTable.BindlessImagesExp
            .pfnImportExternalSemaphoreOpaqueFDExp;

    if (nullptr == pfnImportExternalSemaphoreOpaqueFDExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pInteropSemaphoreDesc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phInteropSemaphore) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result = pfnImportExternalSemaphoreOpaqueFDExp(
        hContext, hDevice, pInteropSemaphoreDesc, phInteropSemaphore);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesDestroyExternalSemaphoreExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesDestroyExternalSemaphoreExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_interop_semaphore_handle_t
        hInteropSemaphore ///< [in] handle of interop semaphore to be destroyed
) {
    auto pfnDestroyExternalSemaphoreExp =
        context.urDdiTable.BindlessImagesExp.pfnDestroyExternalSemaphoreExp;

    if (nullptr == pfnDestroyExternalSemaphoreExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hInteropSemaphore) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result =
        pfnDestroyExternalSemaphoreExp(hContext, hDevice, hInteropSemaphore);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesWaitExternalSemaphoreExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesWaitExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_exp_interop_semaphore_handle_t
        hSemaphore,               ///< [in] interop semaphore handle
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
    ///< previously enqueued commands
    ///< must be complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnWaitExternalSemaphoreExp =
        context.urDdiTable.BindlessImagesExp.pfnWaitExternalSemaphoreExp;

    if (nullptr == pfnWaitExternalSemaphoreExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hSemaphore) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result = pfnWaitExternalSemaphoreExp(
        hQueue, hSemaphore, numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesSignalExternalSemaphoreExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesSignalExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_exp_interop_semaphore_handle_t
        hSemaphore,               ///< [in] interop semaphore handle
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
    ///< previously enqueued commands
    ///< must be complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnSignalExternalSemaphoreExp =
        context.urDdiTable.BindlessImagesExp.pfnSignalExternalSemaphoreExp;

    if (nullptr == pfnSignalExternalSemaphoreExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hSemaphore) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result = pfnSignalExternalSemaphoreExp(
        hQueue, hSemaphore, numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferCreateExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferCreateExp(
    ur_context_handle_t hContext, ///< [in] Handle of the context object.
    ur_device_handle_t hDevice,   ///< [in] Handle of the device object.
    const ur_exp_command_buffer_desc_t
        *pCommandBufferDesc, ///< [in][optional] command-buffer descriptor.
    ur_exp_command_buffer_handle_t
        *phCommandBuffer ///< [out] Pointer to command-Buffer handle.
) {
    auto pfnCreateExp = context.urDdiTable.CommandBufferExp.pfnCreateExp;

    if (nullptr == pfnCreateExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDevice)) {
        refCountContext.logInvalidReference(hDevice);
    }

    ur_result_t result =
        pfnCreateExp(hContext, hDevice, pCommandBufferDesc, phCommandBuffer);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferRetainExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferRetainExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer ///< [in] Handle of the command-buffer object.
) {
    auto pfnRetainExp = context.urDdiTable.CommandBufferExp.pfnRetainExp;

    if (nullptr == pfnRetainExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRetainExp(hCommandBuffer);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferReleaseExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferReleaseExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer ///< [in] Handle of the command-buffer object.
) {
    auto pfnReleaseExp = context.urDdiTable.CommandBufferExp.pfnReleaseExp;

    if (nullptr == pfnReleaseExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnReleaseExp(hCommandBuffer);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferFinalizeExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferFinalizeExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer ///< [in] Handle of the command-buffer object.
) {
    auto pfnFinalizeExp = context.urDdiTable.CommandBufferExp.pfnFinalizeExp;

    if (nullptr == pfnFinalizeExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnFinalizeExp(hCommandBuffer);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendKernelLaunchExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,         ///< [in] Handle of the command-buffer object.
    ur_kernel_handle_t hKernel, ///< [in] Kernel to append.
    uint32_t workDim,           ///< [in] Dimension of the kernel execution.
    const size_t
        *pGlobalWorkOffset, ///< [in] Offset to use when executing kernel.
    const size_t *
        pGlobalWorkSize, ///< [in] Global work size to use when executing kernel.
    const size_t
        *pLocalWorkSize, ///< [in] Local work size to use when executing kernel.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint, ///< [out][optional] Sync point associated with this command.
    ur_exp_command_buffer_command_handle_t
        *phCommand ///< [out][optional] Handle to this command.
) {
    auto pfnAppendKernelLaunchExp =
        context.urDdiTable.CommandBufferExp.pfnAppendKernelLaunchExp;

    if (nullptr == pfnAppendKernelLaunchExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pGlobalWorkOffset) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pGlobalWorkSize) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pLocalWorkSize) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    ur_result_t result = pfnAppendKernelLaunchExp(
        hCommandBuffer, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
        pLocalWorkSize, numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint,
        phCommand);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendUSMMemcpyExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendUSMMemcpyExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer, ///< [in] Handle of the command-buffer object.
    void *pDst,         ///< [in] Location the data will be copied to.
    const void *pSrc,   ///< [in] The data to be copied.
    size_t size,        ///< [in] The number of bytes to copy.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
) {
    auto pfnAppendUSMMemcpyExp =
        context.urDdiTable.CommandBufferExp.pfnAppendUSMMemcpyExp;

    if (nullptr == pfnAppendUSMMemcpyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pDst) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (size == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }
    }

    ur_result_t result = pfnAppendUSMMemcpyExp(hCommandBuffer, pDst, pSrc, size,
                                               numSyncPointsInWaitList,
                                               pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendUSMFillExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendUSMFillExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,   ///< [in] handle of the command-buffer object.
    void *pMemory,        ///< [in] pointer to USM allocated memory to fill.
    const void *pPattern, ///< [in] pointer to the fill pattern.
    size_t patternSize,   ///< [in] size in bytes of the pattern.
    size_t
        size, ///< [in] fill size in bytes, must be a multiple of patternSize.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] sync point associated with this command.
) {
    auto pfnAppendUSMFillExp =
        context.urDdiTable.CommandBufferExp.pfnAppendUSMFillExp;

    if (nullptr == pfnAppendUSMFillExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pMemory) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pPattern) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (patternSize == 0 || size == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (patternSize > size) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if ((patternSize & (patternSize - 1)) != 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (size % patternSize != 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

        if (pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }
    }

    ur_result_t result = pfnAppendUSMFillExp(
        hCommandBuffer, pMemory, pPattern, patternSize, size,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferCopyExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hSrcMem, ///< [in] The data to be copied.
    ur_mem_handle_t hDstMem, ///< [in] The location the data will be copied to.
    size_t srcOffset,        ///< [in] Offset into the source memory.
    size_t dstOffset,        ///< [in] Offset into the destination memory
    size_t size,             ///< [in] The number of bytes to be copied.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
) {
    auto pfnAppendMemBufferCopyExp =
        context.urDdiTable.CommandBufferExp.pfnAppendMemBufferCopyExp;

    if (nullptr == pfnAppendMemBufferCopyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hSrcMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDstMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hSrcMem)) {
        refCountContext.logInvalidReference(hSrcMem);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDstMem)) {
        refCountContext.logInvalidReference(hDstMem);
    }

    ur_result_t result = pfnAppendMemBufferCopyExp(
        hCommandBuffer, hSrcMem, hDstMem, srcOffset, dstOffset, size,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferWriteExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] Handle of the buffer object.
    size_t offset,           ///< [in] Offset in bytes in the buffer object.
    size_t size,             ///< [in] Size in bytes of data being written.
    const void *
        pSrc, ///< [in] Pointer to host memory where data is to be written from.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
) {
    auto pfnAppendMemBufferWriteExp =
        context.urDdiTable.CommandBufferExp.pfnAppendMemBufferWriteExp;

    if (nullptr == pfnAppendMemBufferWriteExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBuffer)) {
        refCountContext.logInvalidReference(hBuffer);
    }

    ur_result_t result = pfnAppendMemBufferWriteExp(
        hCommandBuffer, hBuffer, offset, size, pSrc, numSyncPointsInWaitList,
        pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferReadExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] Handle of the buffer object.
    size_t offset,           ///< [in] Offset in bytes in the buffer object.
    size_t size,             ///< [in] Size in bytes of data being written.
    void *pDst, ///< [in] Pointer to host memory where data is to be written to.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
) {
    auto pfnAppendMemBufferReadExp =
        context.urDdiTable.CommandBufferExp.pfnAppendMemBufferReadExp;

    if (nullptr == pfnAppendMemBufferReadExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pDst) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBuffer)) {
        refCountContext.logInvalidReference(hBuffer);
    }

    ur_result_t result = pfnAppendMemBufferReadExp(
        hCommandBuffer, hBuffer, offset, size, pDst, numSyncPointsInWaitList,
        pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferCopyRectExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyRectExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hSrcMem, ///< [in] The data to be copied.
    ur_mem_handle_t hDstMem, ///< [in] The location the data will be copied to.
    ur_rect_offset_t
        srcOrigin, ///< [in] Origin for the region of data to be copied from the source.
    ur_rect_offset_t
        dstOrigin, ///< [in] Origin for the region of data to be copied to in the destination.
    ur_rect_region_t
        region, ///< [in] The extents describing the region to be copied.
    size_t srcRowPitch,   ///< [in] Row pitch of the source memory.
    size_t srcSlicePitch, ///< [in] Slice pitch of the source memory.
    size_t dstRowPitch,   ///< [in] Row pitch of the destination memory.
    size_t dstSlicePitch, ///< [in] Slice pitch of the destination memory.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
) {
    auto pfnAppendMemBufferCopyRectExp =
        context.urDdiTable.CommandBufferExp.pfnAppendMemBufferCopyRectExp;

    if (nullptr == pfnAppendMemBufferCopyRectExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hSrcMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hDstMem) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hSrcMem)) {
        refCountContext.logInvalidReference(hSrcMem);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hDstMem)) {
        refCountContext.logInvalidReference(hDstMem);
    }

    ur_result_t result = pfnAppendMemBufferCopyRectExp(
        hCommandBuffer, hSrcMem, hDstMem, srcOrigin, dstOrigin, region,
        srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferWriteRectExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteRectExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] Handle of the buffer object.
    ur_rect_offset_t bufferOffset, ///< [in] 3D offset in the buffer.
    ur_rect_offset_t hostOffset,   ///< [in] 3D offset in the host region.
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth.
    size_t
        bufferRowPitch, ///< [in] Length of each row in bytes in the buffer object.
    size_t
        bufferSlicePitch, ///< [in] Length of each 2D slice in bytes in the buffer object being
                          ///< written.
    size_t
        hostRowPitch, ///< [in] Length of each row in bytes in the host memory region pointed to
                      ///< by pSrc.
    size_t
        hostSlicePitch, ///< [in] Length of each 2D slice in bytes in the host memory region
                        ///< pointed to by pSrc.
    void *
        pSrc, ///< [in] Pointer to host memory where data is to be written from.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
) {
    auto pfnAppendMemBufferWriteRectExp =
        context.urDdiTable.CommandBufferExp.pfnAppendMemBufferWriteRectExp;

    if (nullptr == pfnAppendMemBufferWriteRectExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pSrc) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBuffer)) {
        refCountContext.logInvalidReference(hBuffer);
    }

    ur_result_t result = pfnAppendMemBufferWriteRectExp(
        hCommandBuffer, hBuffer, bufferOffset, hostOffset, region,
        bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferReadRectExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadRectExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] Handle of the buffer object.
    ur_rect_offset_t bufferOffset, ///< [in] 3D offset in the buffer.
    ur_rect_offset_t hostOffset,   ///< [in] 3D offset in the host region.
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth.
    size_t
        bufferRowPitch, ///< [in] Length of each row in bytes in the buffer object.
    size_t
        bufferSlicePitch, ///< [in] Length of each 2D slice in bytes in the buffer object being read.
    size_t
        hostRowPitch, ///< [in] Length of each row in bytes in the host memory region pointed to
                      ///< by pDst.
    size_t
        hostSlicePitch, ///< [in] Length of each 2D slice in bytes in the host memory region
                        ///< pointed to by pDst.
    void *pDst, ///< [in] Pointer to host memory where data is to be read into.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
) {
    auto pfnAppendMemBufferReadRectExp =
        context.urDdiTable.CommandBufferExp.pfnAppendMemBufferReadRectExp;

    if (nullptr == pfnAppendMemBufferReadRectExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pDst) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBuffer)) {
        refCountContext.logInvalidReference(hBuffer);
    }

    ur_result_t result = pfnAppendMemBufferReadRectExp(
        hCommandBuffer, hBuffer, bufferOffset, hostOffset, region,
        bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferFillExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferFillExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] handle of the buffer object.
    const void *pPattern,    ///< [in] pointer to the fill pattern.
    size_t patternSize,      ///< [in] size in bytes of the pattern.
    size_t offset,           ///< [in] offset into the buffer.
    size_t
        size, ///< [in] fill size in bytes, must be a multiple of patternSize.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] sync point associated with this command.
) {
    auto pfnAppendMemBufferFillExp =
        context.urDdiTable.CommandBufferExp.pfnAppendMemBufferFillExp;

    if (nullptr == pfnAppendMemBufferFillExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pPattern) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hBuffer)) {
        refCountContext.logInvalidReference(hBuffer);
    }

    ur_result_t result = pfnAppendMemBufferFillExp(
        hCommandBuffer, hBuffer, pPattern, patternSize, offset, size,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendUSMPrefetchExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendUSMPrefetchExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,  ///< [in] handle of the command-buffer object.
    const void *pMemory, ///< [in] pointer to USM allocated memory to prefetch.
    size_t size,         ///< [in] size in bytes to be fetched.
    ur_usm_migration_flags_t flags, ///< [in] USM prefetch flags
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] sync point associated with this command.
) {
    auto pfnAppendUSMPrefetchExp =
        context.urDdiTable.CommandBufferExp.pfnAppendUSMPrefetchExp;

    if (nullptr == pfnAppendUSMPrefetchExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pMemory) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_USM_MIGRATION_FLAGS_MASK & flags) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (size == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    ur_result_t result = pfnAppendUSMPrefetchExp(
        hCommandBuffer, pMemory, size, flags, numSyncPointsInWaitList,
        pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendUSMAdviseExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendUSMAdviseExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,           ///< [in] handle of the command-buffer object.
    const void *pMemory,          ///< [in] pointer to the USM memory object.
    size_t size,                  ///< [in] size in bytes to be advised.
    ur_usm_advice_flags_t advice, ///< [in] USM memory advice
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] sync point associated with this command.
) {
    auto pfnAppendUSMAdviseExp =
        context.urDdiTable.CommandBufferExp.pfnAppendUSMAdviseExp;

    if (nullptr == pfnAppendUSMAdviseExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pMemory) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_USM_ADVICE_FLAGS_MASK & advice) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP;
        }

        if (size == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    ur_result_t result = pfnAppendUSMAdviseExp(hCommandBuffer, pMemory, size,
                                               advice, numSyncPointsInWaitList,
                                               pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferEnqueueExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer, ///< [in] Handle of the command-buffer object.
    ur_queue_handle_t
        hQueue, ///< [in] The queue to submit this command-buffer for execution.
    uint32_t numEventsInWaitList, ///< [in] Size of the event wait list.
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the command-buffer execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating no wait events.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command-buffer execution instance.
) {
    auto pfnEnqueueExp = context.urDdiTable.CommandBufferExp.pfnEnqueueExp;

    if (nullptr == pfnEnqueueExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    ur_result_t result = pfnEnqueueExp(
        hCommandBuffer, hQueue, numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferRetainCommandExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferRetainCommandExp(
    ur_exp_command_buffer_command_handle_t
        hCommand ///< [in] Handle of the command-buffer command.
) {
    auto pfnRetainCommandExp =
        context.urDdiTable.CommandBufferExp.pfnRetainCommandExp;

    if (nullptr == pfnRetainCommandExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommand) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnRetainCommandExp(hCommand);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferReleaseCommandExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferReleaseCommandExp(
    ur_exp_command_buffer_command_handle_t
        hCommand ///< [in] Handle of the command-buffer command.
) {
    auto pfnReleaseCommandExp =
        context.urDdiTable.CommandBufferExp.pfnReleaseCommandExp;

    if (nullptr == pfnReleaseCommandExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommand) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    ur_result_t result = pfnReleaseCommandExp(hCommand);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferUpdateKernelLaunchExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferUpdateKernelLaunchExp(
    ur_exp_command_buffer_command_handle_t
        hCommand, ///< [in] Handle of the command-buffer kernel command to update.
    const ur_exp_command_buffer_update_kernel_launch_desc_t *
        pUpdateKernelLaunch ///< [in] Struct defining how the kernel command is to be updated.
) {
    auto pfnUpdateKernelLaunchExp =
        context.urDdiTable.CommandBufferExp.pfnUpdateKernelLaunchExp;

    if (nullptr == pfnUpdateKernelLaunchExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommand) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pUpdateKernelLaunch) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    ur_result_t result =
        pfnUpdateKernelLaunchExp(hCommand, pUpdateKernelLaunch);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferGetInfoExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferGetInfoExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer, ///< [in] handle of the command-buffer object
    ur_exp_command_buffer_info_t
        propName, ///< [in] the name of the command-buffer property to query
    size_t
        propSize, ///< [in] size in bytes of the command-buffer property value
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the
                    ///< command-buffer property
    size_t *
        pPropSizeRet ///< [out][optional] bytes returned in command-buffer property
) {
    auto pfnGetInfoExp = context.urDdiTable.CommandBufferExp.pfnGetInfoExp;

    if (nullptr == pfnGetInfoExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommandBuffer) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_EXP_COMMAND_BUFFER_INFO_REFERENCE_COUNT < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    ur_result_t result = pfnGetInfoExp(hCommandBuffer, propName, propSize,
                                       pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferCommandGetInfoExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferCommandGetInfoExp(
    ur_exp_command_buffer_command_handle_t
        hCommand, ///< [in] handle of the command-buffer command object
    ur_exp_command_buffer_command_info_t
        propName, ///< [in] the name of the command-buffer command property to query
    size_t
        propSize, ///< [in] size in bytes of the command-buffer command property value
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the
                    ///< command-buffer command property
    size_t *
        pPropSizeRet ///< [out][optional] bytes returned in command-buffer command property
) {
    auto pfnCommandGetInfoExp =
        context.urDdiTable.CommandBufferExp.pfnCommandGetInfoExp;

    if (nullptr == pfnCommandGetInfoExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hCommand) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_EXP_COMMAND_BUFFER_COMMAND_INFO_REFERENCE_COUNT < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    ur_result_t result = pfnCommandGetInfoExp(hCommand, propName, propSize,
                                              pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueCooperativeKernelLaunchExp
__urdlllocal ur_result_t UR_APICALL urEnqueueCooperativeKernelLaunchExp(
    ur_queue_handle_t hQueue,   ///< [in] handle of the queue object
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t
        workDim, ///< [in] number of dimensions, from 1 to 3, to specify the global and
                 ///< work-group work-items
    const size_t *
        pGlobalWorkOffset, ///< [in] pointer to an array of workDim unsigned values that specify the
    ///< offset used to calculate the global ID of a work-item
    const size_t *
        pGlobalWorkSize, ///< [in] pointer to an array of workDim unsigned values that specify the
    ///< number of global work-items in workDim that will execute the kernel
    ///< function
    const size_t *
        pLocalWorkSize, ///< [in][optional] pointer to an array of workDim unsigned values that
    ///< specify the number of local work-items forming a work-group that will
    ///< execute the kernel function.
    ///< If nullptr, the runtime implementation will choose the work-group
    ///< size.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the kernel execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    ///< event.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< kernel execution instance.
) {
    auto pfnCooperativeKernelLaunchExp =
        context.urDdiTable.EnqueueExp.pfnCooperativeKernelLaunchExp;

    if (nullptr == pfnCooperativeKernelLaunchExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hQueue) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pGlobalWorkOffset) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == pGlobalWorkSize) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (phEventWaitList == NULL && numEventsInWaitList > 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList == 0) {
            return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
        }

        if (phEventWaitList != NULL && numEventsInWaitList > 0) {
            for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                if (phEventWaitList[i] == NULL) {
                    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                }
            }
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hQueue)) {
        refCountContext.logInvalidReference(hQueue);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    ur_result_t result = pfnCooperativeKernelLaunchExp(
        hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
        pLocalWorkSize, numEventsInWaitList, phEventWaitList, phEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSuggestMaxCooperativeGroupCountExp
__urdlllocal ur_result_t UR_APICALL urKernelSuggestMaxCooperativeGroupCountExp(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    size_t
        localWorkSize, ///< [in] number of local work-items that will form a work-group when the
                       ///< kernel is launched
    size_t
        dynamicSharedMemorySize, ///< [in] size of dynamic shared memory, for each work-group, in bytes,
    ///< that will be used when the kernel is launched
    uint32_t *pGroupCountRet ///< [out] pointer to maximum number of groups
) {
    auto pfnSuggestMaxCooperativeGroupCountExp =
        context.urDdiTable.KernelExp.pfnSuggestMaxCooperativeGroupCountExp;

    if (nullptr == pfnSuggestMaxCooperativeGroupCountExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hKernel) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pGroupCountRet) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hKernel)) {
        refCountContext.logInvalidReference(hKernel);
    }

    ur_result_t result = pfnSuggestMaxCooperativeGroupCountExp(
        hKernel, localWorkSize, dynamicSharedMemorySize, pGroupCountRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramBuildExp
__urdlllocal ur_result_t UR_APICALL urProgramBuildExp(
    ur_program_handle_t hProgram, ///< [in] Handle of the program to build.
    uint32_t numDevices,          ///< [in] number of devices
    ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] pointer to array of device handles
    const char *
        pOptions ///< [in][optional] pointer to build options null-terminated string.
) {
    auto pfnBuildExp = context.urDdiTable.ProgramExp.pfnBuildExp;

    if (nullptr == pfnBuildExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phDevices) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result = pfnBuildExp(hProgram, numDevices, phDevices, pOptions);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCompileExp
__urdlllocal ur_result_t UR_APICALL urProgramCompileExp(
    ur_program_handle_t
        hProgram,        ///< [in][out] handle of the program to compile.
    uint32_t numDevices, ///< [in] number of devices
    ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] pointer to array of device handles
    const char *
        pOptions ///< [in][optional] pointer to build options null-terminated string.
) {
    auto pfnCompileExp = context.urDdiTable.ProgramExp.pfnCompileExp;

    if (nullptr == pfnCompileExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phDevices) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hProgram)) {
        refCountContext.logInvalidReference(hProgram);
    }

    ur_result_t result =
        pfnCompileExp(hProgram, numDevices, phDevices, pOptions);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramLinkExp
__urdlllocal ur_result_t UR_APICALL urProgramLinkExp(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    uint32_t numDevices,          ///< [in] number of devices
    ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] pointer to array of device handles
    uint32_t count, ///< [in] number of program handles in `phPrograms`.
    const ur_program_handle_t *
        phPrograms, ///< [in][range(0, count)] pointer to array of program handles.
    const char *
        pOptions, ///< [in][optional] pointer to linker options null-terminated string.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of program object created.
) {
    auto pfnLinkExp = context.urDdiTable.ProgramExp.pfnLinkExp;

    if (nullptr == pfnLinkExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == phDevices) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phPrograms) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (NULL == phProgram) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (count == 0) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnLinkExp(hContext, numDevices, phDevices, count,
                                    phPrograms, pOptions, phProgram);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMImportExp
__urdlllocal ur_result_t UR_APICALL urUSMImportExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem,                   ///< [in] pointer to host memory object
    size_t size ///< [in] size in bytes of the host memory object to be imported
) {
    auto pfnImportExp = context.urDdiTable.USMExp.pfnImportExp;

    if (nullptr == pfnImportExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnImportExp(hContext, pMem, size);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMReleaseExp
__urdlllocal ur_result_t UR_APICALL urUSMReleaseExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to host memory object
) {
    auto pfnReleaseExp = context.urDdiTable.USMExp.pfnReleaseExp;

    if (nullptr == pfnReleaseExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == hContext) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == pMem) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(hContext)) {
        refCountContext.logInvalidReference(hContext);
    }

    ur_result_t result = pfnReleaseExp(hContext, pMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PEnablePeerAccessExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PEnablePeerAccessExp(
    ur_device_handle_t
        commandDevice,            ///< [in] handle of the command device object
    ur_device_handle_t peerDevice ///< [in] handle of the peer device object
) {
    auto pfnEnablePeerAccessExp =
        context.urDdiTable.UsmP2PExp.pfnEnablePeerAccessExp;

    if (nullptr == pfnEnablePeerAccessExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == commandDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == peerDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(commandDevice)) {
        refCountContext.logInvalidReference(commandDevice);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(peerDevice)) {
        refCountContext.logInvalidReference(peerDevice);
    }

    ur_result_t result = pfnEnablePeerAccessExp(commandDevice, peerDevice);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PDisablePeerAccessExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PDisablePeerAccessExp(
    ur_device_handle_t
        commandDevice,            ///< [in] handle of the command device object
    ur_device_handle_t peerDevice ///< [in] handle of the peer device object
) {
    auto pfnDisablePeerAccessExp =
        context.urDdiTable.UsmP2PExp.pfnDisablePeerAccessExp;

    if (nullptr == pfnDisablePeerAccessExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == commandDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == peerDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(commandDevice)) {
        refCountContext.logInvalidReference(commandDevice);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(peerDevice)) {
        refCountContext.logInvalidReference(peerDevice);
    }

    ur_result_t result = pfnDisablePeerAccessExp(commandDevice, peerDevice);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PPeerAccessGetInfoExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PPeerAccessGetInfoExp(
    ur_device_handle_t
        commandDevice,             ///< [in] handle of the command device object
    ur_device_handle_t peerDevice, ///< [in] handle of the peer device object
    ur_exp_peer_info_t propName,   ///< [in] type of the info to retrieve
    size_t propSize, ///< [in] the number of bytes pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If propSize is not equal to or greater than the real number of bytes
    ///< needed to return the info
    ///< then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
) {
    auto pfnPeerAccessGetInfoExp =
        context.urDdiTable.UsmP2PExp.pfnPeerAccessGetInfoExp;

    if (nullptr == pfnPeerAccessGetInfoExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    if (context.enableParameterValidation) {
        if (NULL == commandDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (NULL == peerDevice) {
            return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
        }

        if (propSize != 0 && pPropValue == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (pPropValue == NULL && pPropSizeRet == NULL) {
            return UR_RESULT_ERROR_INVALID_NULL_POINTER;
        }

        if (UR_EXP_PEER_INFO_UR_PEER_ATOMICS_SUPPORTED < propName) {
            return UR_RESULT_ERROR_INVALID_ENUMERATION;
        }

        if (propSize == 0 && pPropValue != NULL) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(commandDevice)) {
        refCountContext.logInvalidReference(commandDevice);
    }

    if (context.enableLifetimeValidation &&
        !refCountContext.isReferenceValid(peerDevice)) {
        refCountContext.logInvalidReference(peerDevice);
    }

    ur_result_t result =
        pfnPeerAccessGetInfoExp(commandDevice, peerDevice, propName, propSize,
                                pPropValue, pPropSizeRet);

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
UR_DLLEXPORT ur_result_t UR_APICALL urGetGlobalProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_global_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.Global;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnAdapterGet = pDdiTable->pfnAdapterGet;
    pDdiTable->pfnAdapterGet = ur_validation_layer::urAdapterGet;

    dditable.pfnAdapterRelease = pDdiTable->pfnAdapterRelease;
    pDdiTable->pfnAdapterRelease = ur_validation_layer::urAdapterRelease;

    dditable.pfnAdapterRetain = pDdiTable->pfnAdapterRetain;
    pDdiTable->pfnAdapterRetain = ur_validation_layer::urAdapterRetain;

    dditable.pfnAdapterGetLastError = pDdiTable->pfnAdapterGetLastError;
    pDdiTable->pfnAdapterGetLastError =
        ur_validation_layer::urAdapterGetLastError;

    dditable.pfnAdapterGetInfo = pDdiTable->pfnAdapterGetInfo;
    pDdiTable->pfnAdapterGetInfo = ur_validation_layer::urAdapterGetInfo;

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's BindlessImagesExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetBindlessImagesExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_bindless_images_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.BindlessImagesExp;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnUnsampledImageHandleDestroyExp =
        pDdiTable->pfnUnsampledImageHandleDestroyExp;
    pDdiTable->pfnUnsampledImageHandleDestroyExp =
        ur_validation_layer::urBindlessImagesUnsampledImageHandleDestroyExp;

    dditable.pfnSampledImageHandleDestroyExp =
        pDdiTable->pfnSampledImageHandleDestroyExp;
    pDdiTable->pfnSampledImageHandleDestroyExp =
        ur_validation_layer::urBindlessImagesSampledImageHandleDestroyExp;

    dditable.pfnImageAllocateExp = pDdiTable->pfnImageAllocateExp;
    pDdiTable->pfnImageAllocateExp =
        ur_validation_layer::urBindlessImagesImageAllocateExp;

    dditable.pfnImageFreeExp = pDdiTable->pfnImageFreeExp;
    pDdiTable->pfnImageFreeExp =
        ur_validation_layer::urBindlessImagesImageFreeExp;

    dditable.pfnUnsampledImageCreateExp = pDdiTable->pfnUnsampledImageCreateExp;
    pDdiTable->pfnUnsampledImageCreateExp =
        ur_validation_layer::urBindlessImagesUnsampledImageCreateExp;

    dditable.pfnSampledImageCreateExp = pDdiTable->pfnSampledImageCreateExp;
    pDdiTable->pfnSampledImageCreateExp =
        ur_validation_layer::urBindlessImagesSampledImageCreateExp;

    dditable.pfnImageCopyExp = pDdiTable->pfnImageCopyExp;
    pDdiTable->pfnImageCopyExp =
        ur_validation_layer::urBindlessImagesImageCopyExp;

    dditable.pfnImageGetInfoExp = pDdiTable->pfnImageGetInfoExp;
    pDdiTable->pfnImageGetInfoExp =
        ur_validation_layer::urBindlessImagesImageGetInfoExp;

    dditable.pfnMipmapGetLevelExp = pDdiTable->pfnMipmapGetLevelExp;
    pDdiTable->pfnMipmapGetLevelExp =
        ur_validation_layer::urBindlessImagesMipmapGetLevelExp;

    dditable.pfnMipmapFreeExp = pDdiTable->pfnMipmapFreeExp;
    pDdiTable->pfnMipmapFreeExp =
        ur_validation_layer::urBindlessImagesMipmapFreeExp;

    dditable.pfnImportOpaqueFDExp = pDdiTable->pfnImportOpaqueFDExp;
    pDdiTable->pfnImportOpaqueFDExp =
        ur_validation_layer::urBindlessImagesImportOpaqueFDExp;

    dditable.pfnMapExternalArrayExp = pDdiTable->pfnMapExternalArrayExp;
    pDdiTable->pfnMapExternalArrayExp =
        ur_validation_layer::urBindlessImagesMapExternalArrayExp;

    dditable.pfnReleaseInteropExp = pDdiTable->pfnReleaseInteropExp;
    pDdiTable->pfnReleaseInteropExp =
        ur_validation_layer::urBindlessImagesReleaseInteropExp;

    dditable.pfnImportExternalSemaphoreOpaqueFDExp =
        pDdiTable->pfnImportExternalSemaphoreOpaqueFDExp;
    pDdiTable->pfnImportExternalSemaphoreOpaqueFDExp =
        ur_validation_layer::urBindlessImagesImportExternalSemaphoreOpaqueFDExp;

    dditable.pfnDestroyExternalSemaphoreExp =
        pDdiTable->pfnDestroyExternalSemaphoreExp;
    pDdiTable->pfnDestroyExternalSemaphoreExp =
        ur_validation_layer::urBindlessImagesDestroyExternalSemaphoreExp;

    dditable.pfnWaitExternalSemaphoreExp =
        pDdiTable->pfnWaitExternalSemaphoreExp;
    pDdiTable->pfnWaitExternalSemaphoreExp =
        ur_validation_layer::urBindlessImagesWaitExternalSemaphoreExp;

    dditable.pfnSignalExternalSemaphoreExp =
        pDdiTable->pfnSignalExternalSemaphoreExp;
    pDdiTable->pfnSignalExternalSemaphoreExp =
        ur_validation_layer::urBindlessImagesSignalExternalSemaphoreExp;

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's CommandBufferExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetCommandBufferExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_command_buffer_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.CommandBufferExp;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnCreateExp = pDdiTable->pfnCreateExp;
    pDdiTable->pfnCreateExp = ur_validation_layer::urCommandBufferCreateExp;

    dditable.pfnRetainExp = pDdiTable->pfnRetainExp;
    pDdiTable->pfnRetainExp = ur_validation_layer::urCommandBufferRetainExp;

    dditable.pfnReleaseExp = pDdiTable->pfnReleaseExp;
    pDdiTable->pfnReleaseExp = ur_validation_layer::urCommandBufferReleaseExp;

    dditable.pfnFinalizeExp = pDdiTable->pfnFinalizeExp;
    pDdiTable->pfnFinalizeExp = ur_validation_layer::urCommandBufferFinalizeExp;

    dditable.pfnAppendKernelLaunchExp = pDdiTable->pfnAppendKernelLaunchExp;
    pDdiTable->pfnAppendKernelLaunchExp =
        ur_validation_layer::urCommandBufferAppendKernelLaunchExp;

    dditable.pfnAppendUSMMemcpyExp = pDdiTable->pfnAppendUSMMemcpyExp;
    pDdiTable->pfnAppendUSMMemcpyExp =
        ur_validation_layer::urCommandBufferAppendUSMMemcpyExp;

    dditable.pfnAppendUSMFillExp = pDdiTable->pfnAppendUSMFillExp;
    pDdiTable->pfnAppendUSMFillExp =
        ur_validation_layer::urCommandBufferAppendUSMFillExp;

    dditable.pfnAppendMemBufferCopyExp = pDdiTable->pfnAppendMemBufferCopyExp;
    pDdiTable->pfnAppendMemBufferCopyExp =
        ur_validation_layer::urCommandBufferAppendMemBufferCopyExp;

    dditable.pfnAppendMemBufferWriteExp = pDdiTable->pfnAppendMemBufferWriteExp;
    pDdiTable->pfnAppendMemBufferWriteExp =
        ur_validation_layer::urCommandBufferAppendMemBufferWriteExp;

    dditable.pfnAppendMemBufferReadExp = pDdiTable->pfnAppendMemBufferReadExp;
    pDdiTable->pfnAppendMemBufferReadExp =
        ur_validation_layer::urCommandBufferAppendMemBufferReadExp;

    dditable.pfnAppendMemBufferCopyRectExp =
        pDdiTable->pfnAppendMemBufferCopyRectExp;
    pDdiTable->pfnAppendMemBufferCopyRectExp =
        ur_validation_layer::urCommandBufferAppendMemBufferCopyRectExp;

    dditable.pfnAppendMemBufferWriteRectExp =
        pDdiTable->pfnAppendMemBufferWriteRectExp;
    pDdiTable->pfnAppendMemBufferWriteRectExp =
        ur_validation_layer::urCommandBufferAppendMemBufferWriteRectExp;

    dditable.pfnAppendMemBufferReadRectExp =
        pDdiTable->pfnAppendMemBufferReadRectExp;
    pDdiTable->pfnAppendMemBufferReadRectExp =
        ur_validation_layer::urCommandBufferAppendMemBufferReadRectExp;

    dditable.pfnAppendMemBufferFillExp = pDdiTable->pfnAppendMemBufferFillExp;
    pDdiTable->pfnAppendMemBufferFillExp =
        ur_validation_layer::urCommandBufferAppendMemBufferFillExp;

    dditable.pfnAppendUSMPrefetchExp = pDdiTable->pfnAppendUSMPrefetchExp;
    pDdiTable->pfnAppendUSMPrefetchExp =
        ur_validation_layer::urCommandBufferAppendUSMPrefetchExp;

    dditable.pfnAppendUSMAdviseExp = pDdiTable->pfnAppendUSMAdviseExp;
    pDdiTable->pfnAppendUSMAdviseExp =
        ur_validation_layer::urCommandBufferAppendUSMAdviseExp;

    dditable.pfnEnqueueExp = pDdiTable->pfnEnqueueExp;
    pDdiTable->pfnEnqueueExp = ur_validation_layer::urCommandBufferEnqueueExp;

    dditable.pfnRetainCommandExp = pDdiTable->pfnRetainCommandExp;
    pDdiTable->pfnRetainCommandExp =
        ur_validation_layer::urCommandBufferRetainCommandExp;

    dditable.pfnReleaseCommandExp = pDdiTable->pfnReleaseCommandExp;
    pDdiTable->pfnReleaseCommandExp =
        ur_validation_layer::urCommandBufferReleaseCommandExp;

    dditable.pfnUpdateKernelLaunchExp = pDdiTable->pfnUpdateKernelLaunchExp;
    pDdiTable->pfnUpdateKernelLaunchExp =
        ur_validation_layer::urCommandBufferUpdateKernelLaunchExp;

    dditable.pfnGetInfoExp = pDdiTable->pfnGetInfoExp;
    pDdiTable->pfnGetInfoExp = ur_validation_layer::urCommandBufferGetInfoExp;

    dditable.pfnCommandGetInfoExp = pDdiTable->pfnCommandGetInfoExp;
    pDdiTable->pfnCommandGetInfoExp =
        ur_validation_layer::urCommandBufferCommandGetInfoExp;

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
UR_DLLEXPORT ur_result_t UR_APICALL urGetContextProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_context_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.Context;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = ur_validation_layer::urContextCreate;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = ur_validation_layer::urContextRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = ur_validation_layer::urContextRelease;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = ur_validation_layer::urContextGetInfo;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle =
        ur_validation_layer::urContextGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle =
        ur_validation_layer::urContextCreateWithNativeHandle;

    dditable.pfnSetExtendedDeleter = pDdiTable->pfnSetExtendedDeleter;
    pDdiTable->pfnSetExtendedDeleter =
        ur_validation_layer::urContextSetExtendedDeleter;

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
UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_enqueue_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.Enqueue;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnKernelLaunch = pDdiTable->pfnKernelLaunch;
    pDdiTable->pfnKernelLaunch = ur_validation_layer::urEnqueueKernelLaunch;

    dditable.pfnEventsWait = pDdiTable->pfnEventsWait;
    pDdiTable->pfnEventsWait = ur_validation_layer::urEnqueueEventsWait;

    dditable.pfnEventsWaitWithBarrier = pDdiTable->pfnEventsWaitWithBarrier;
    pDdiTable->pfnEventsWaitWithBarrier =
        ur_validation_layer::urEnqueueEventsWaitWithBarrier;

    dditable.pfnMemBufferRead = pDdiTable->pfnMemBufferRead;
    pDdiTable->pfnMemBufferRead = ur_validation_layer::urEnqueueMemBufferRead;

    dditable.pfnMemBufferWrite = pDdiTable->pfnMemBufferWrite;
    pDdiTable->pfnMemBufferWrite = ur_validation_layer::urEnqueueMemBufferWrite;

    dditable.pfnMemBufferReadRect = pDdiTable->pfnMemBufferReadRect;
    pDdiTable->pfnMemBufferReadRect =
        ur_validation_layer::urEnqueueMemBufferReadRect;

    dditable.pfnMemBufferWriteRect = pDdiTable->pfnMemBufferWriteRect;
    pDdiTable->pfnMemBufferWriteRect =
        ur_validation_layer::urEnqueueMemBufferWriteRect;

    dditable.pfnMemBufferCopy = pDdiTable->pfnMemBufferCopy;
    pDdiTable->pfnMemBufferCopy = ur_validation_layer::urEnqueueMemBufferCopy;

    dditable.pfnMemBufferCopyRect = pDdiTable->pfnMemBufferCopyRect;
    pDdiTable->pfnMemBufferCopyRect =
        ur_validation_layer::urEnqueueMemBufferCopyRect;

    dditable.pfnMemBufferFill = pDdiTable->pfnMemBufferFill;
    pDdiTable->pfnMemBufferFill = ur_validation_layer::urEnqueueMemBufferFill;

    dditable.pfnMemImageRead = pDdiTable->pfnMemImageRead;
    pDdiTable->pfnMemImageRead = ur_validation_layer::urEnqueueMemImageRead;

    dditable.pfnMemImageWrite = pDdiTable->pfnMemImageWrite;
    pDdiTable->pfnMemImageWrite = ur_validation_layer::urEnqueueMemImageWrite;

    dditable.pfnMemImageCopy = pDdiTable->pfnMemImageCopy;
    pDdiTable->pfnMemImageCopy = ur_validation_layer::urEnqueueMemImageCopy;

    dditable.pfnMemBufferMap = pDdiTable->pfnMemBufferMap;
    pDdiTable->pfnMemBufferMap = ur_validation_layer::urEnqueueMemBufferMap;

    dditable.pfnMemUnmap = pDdiTable->pfnMemUnmap;
    pDdiTable->pfnMemUnmap = ur_validation_layer::urEnqueueMemUnmap;

    dditable.pfnUSMFill = pDdiTable->pfnUSMFill;
    pDdiTable->pfnUSMFill = ur_validation_layer::urEnqueueUSMFill;

    dditable.pfnUSMMemcpy = pDdiTable->pfnUSMMemcpy;
    pDdiTable->pfnUSMMemcpy = ur_validation_layer::urEnqueueUSMMemcpy;

    dditable.pfnUSMPrefetch = pDdiTable->pfnUSMPrefetch;
    pDdiTable->pfnUSMPrefetch = ur_validation_layer::urEnqueueUSMPrefetch;

    dditable.pfnUSMAdvise = pDdiTable->pfnUSMAdvise;
    pDdiTable->pfnUSMAdvise = ur_validation_layer::urEnqueueUSMAdvise;

    dditable.pfnUSMFill2D = pDdiTable->pfnUSMFill2D;
    pDdiTable->pfnUSMFill2D = ur_validation_layer::urEnqueueUSMFill2D;

    dditable.pfnUSMMemcpy2D = pDdiTable->pfnUSMMemcpy2D;
    pDdiTable->pfnUSMMemcpy2D = ur_validation_layer::urEnqueueUSMMemcpy2D;

    dditable.pfnDeviceGlobalVariableWrite =
        pDdiTable->pfnDeviceGlobalVariableWrite;
    pDdiTable->pfnDeviceGlobalVariableWrite =
        ur_validation_layer::urEnqueueDeviceGlobalVariableWrite;

    dditable.pfnDeviceGlobalVariableRead =
        pDdiTable->pfnDeviceGlobalVariableRead;
    pDdiTable->pfnDeviceGlobalVariableRead =
        ur_validation_layer::urEnqueueDeviceGlobalVariableRead;

    dditable.pfnReadHostPipe = pDdiTable->pfnReadHostPipe;
    pDdiTable->pfnReadHostPipe = ur_validation_layer::urEnqueueReadHostPipe;

    dditable.pfnWriteHostPipe = pDdiTable->pfnWriteHostPipe;
    pDdiTable->pfnWriteHostPipe = ur_validation_layer::urEnqueueWriteHostPipe;

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's EnqueueExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_enqueue_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.EnqueueExp;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnCooperativeKernelLaunchExp =
        pDdiTable->pfnCooperativeKernelLaunchExp;
    pDdiTable->pfnCooperativeKernelLaunchExp =
        ur_validation_layer::urEnqueueCooperativeKernelLaunchExp;

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
UR_DLLEXPORT ur_result_t UR_APICALL urGetEventProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_event_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.Event;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = ur_validation_layer::urEventGetInfo;

    dditable.pfnGetProfilingInfo = pDdiTable->pfnGetProfilingInfo;
    pDdiTable->pfnGetProfilingInfo =
        ur_validation_layer::urEventGetProfilingInfo;

    dditable.pfnWait = pDdiTable->pfnWait;
    pDdiTable->pfnWait = ur_validation_layer::urEventWait;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = ur_validation_layer::urEventRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = ur_validation_layer::urEventRelease;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle = ur_validation_layer::urEventGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle =
        ur_validation_layer::urEventCreateWithNativeHandle;

    dditable.pfnSetCallback = pDdiTable->pfnSetCallback;
    pDdiTable->pfnSetCallback = ur_validation_layer::urEventSetCallback;

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
UR_DLLEXPORT ur_result_t UR_APICALL urGetKernelProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_kernel_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.Kernel;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = ur_validation_layer::urKernelCreate;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = ur_validation_layer::urKernelGetInfo;

    dditable.pfnGetGroupInfo = pDdiTable->pfnGetGroupInfo;
    pDdiTable->pfnGetGroupInfo = ur_validation_layer::urKernelGetGroupInfo;

    dditable.pfnGetSubGroupInfo = pDdiTable->pfnGetSubGroupInfo;
    pDdiTable->pfnGetSubGroupInfo =
        ur_validation_layer::urKernelGetSubGroupInfo;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = ur_validation_layer::urKernelRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = ur_validation_layer::urKernelRelease;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle =
        ur_validation_layer::urKernelGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle =
        ur_validation_layer::urKernelCreateWithNativeHandle;

    dditable.pfnSetArgValue = pDdiTable->pfnSetArgValue;
    pDdiTable->pfnSetArgValue = ur_validation_layer::urKernelSetArgValue;

    dditable.pfnSetArgLocal = pDdiTable->pfnSetArgLocal;
    pDdiTable->pfnSetArgLocal = ur_validation_layer::urKernelSetArgLocal;

    dditable.pfnSetArgPointer = pDdiTable->pfnSetArgPointer;
    pDdiTable->pfnSetArgPointer = ur_validation_layer::urKernelSetArgPointer;

    dditable.pfnSetExecInfo = pDdiTable->pfnSetExecInfo;
    pDdiTable->pfnSetExecInfo = ur_validation_layer::urKernelSetExecInfo;

    dditable.pfnSetArgSampler = pDdiTable->pfnSetArgSampler;
    pDdiTable->pfnSetArgSampler = ur_validation_layer::urKernelSetArgSampler;

    dditable.pfnSetArgMemObj = pDdiTable->pfnSetArgMemObj;
    pDdiTable->pfnSetArgMemObj = ur_validation_layer::urKernelSetArgMemObj;

    dditable.pfnSetSpecializationConstants =
        pDdiTable->pfnSetSpecializationConstants;
    pDdiTable->pfnSetSpecializationConstants =
        ur_validation_layer::urKernelSetSpecializationConstants;

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's KernelExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetKernelExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_kernel_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.KernelExp;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnSuggestMaxCooperativeGroupCountExp =
        pDdiTable->pfnSuggestMaxCooperativeGroupCountExp;
    pDdiTable->pfnSuggestMaxCooperativeGroupCountExp =
        ur_validation_layer::urKernelSuggestMaxCooperativeGroupCountExp;

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
UR_DLLEXPORT ur_result_t UR_APICALL urGetMemProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_mem_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.Mem;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnImageCreate = pDdiTable->pfnImageCreate;
    pDdiTable->pfnImageCreate = ur_validation_layer::urMemImageCreate;

    dditable.pfnBufferCreate = pDdiTable->pfnBufferCreate;
    pDdiTable->pfnBufferCreate = ur_validation_layer::urMemBufferCreate;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = ur_validation_layer::urMemRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = ur_validation_layer::urMemRelease;

    dditable.pfnBufferPartition = pDdiTable->pfnBufferPartition;
    pDdiTable->pfnBufferPartition = ur_validation_layer::urMemBufferPartition;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle = ur_validation_layer::urMemGetNativeHandle;

    dditable.pfnBufferCreateWithNativeHandle =
        pDdiTable->pfnBufferCreateWithNativeHandle;
    pDdiTable->pfnBufferCreateWithNativeHandle =
        ur_validation_layer::urMemBufferCreateWithNativeHandle;

    dditable.pfnImageCreateWithNativeHandle =
        pDdiTable->pfnImageCreateWithNativeHandle;
    pDdiTable->pfnImageCreateWithNativeHandle =
        ur_validation_layer::urMemImageCreateWithNativeHandle;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = ur_validation_layer::urMemGetInfo;

    dditable.pfnImageGetInfo = pDdiTable->pfnImageGetInfo;
    pDdiTable->pfnImageGetInfo = ur_validation_layer::urMemImageGetInfo;

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's PhysicalMem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_physical_mem_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.PhysicalMem;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = ur_validation_layer::urPhysicalMemCreate;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = ur_validation_layer::urPhysicalMemRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = ur_validation_layer::urPhysicalMemRelease;

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
UR_DLLEXPORT ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_platform_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.Platform;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnGet = pDdiTable->pfnGet;
    pDdiTable->pfnGet = ur_validation_layer::urPlatformGet;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = ur_validation_layer::urPlatformGetInfo;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle =
        ur_validation_layer::urPlatformGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle =
        ur_validation_layer::urPlatformCreateWithNativeHandle;

    dditable.pfnGetApiVersion = pDdiTable->pfnGetApiVersion;
    pDdiTable->pfnGetApiVersion = ur_validation_layer::urPlatformGetApiVersion;

    dditable.pfnGetBackendOption = pDdiTable->pfnGetBackendOption;
    pDdiTable->pfnGetBackendOption =
        ur_validation_layer::urPlatformGetBackendOption;

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
UR_DLLEXPORT ur_result_t UR_APICALL urGetProgramProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_program_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.Program;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnCreateWithIL = pDdiTable->pfnCreateWithIL;
    pDdiTable->pfnCreateWithIL = ur_validation_layer::urProgramCreateWithIL;

    dditable.pfnCreateWithBinary = pDdiTable->pfnCreateWithBinary;
    pDdiTable->pfnCreateWithBinary =
        ur_validation_layer::urProgramCreateWithBinary;

    dditable.pfnBuild = pDdiTable->pfnBuild;
    pDdiTable->pfnBuild = ur_validation_layer::urProgramBuild;

    dditable.pfnCompile = pDdiTable->pfnCompile;
    pDdiTable->pfnCompile = ur_validation_layer::urProgramCompile;

    dditable.pfnLink = pDdiTable->pfnLink;
    pDdiTable->pfnLink = ur_validation_layer::urProgramLink;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = ur_validation_layer::urProgramRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = ur_validation_layer::urProgramRelease;

    dditable.pfnGetFunctionPointer = pDdiTable->pfnGetFunctionPointer;
    pDdiTable->pfnGetFunctionPointer =
        ur_validation_layer::urProgramGetFunctionPointer;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = ur_validation_layer::urProgramGetInfo;

    dditable.pfnGetBuildInfo = pDdiTable->pfnGetBuildInfo;
    pDdiTable->pfnGetBuildInfo = ur_validation_layer::urProgramGetBuildInfo;

    dditable.pfnSetSpecializationConstants =
        pDdiTable->pfnSetSpecializationConstants;
    pDdiTable->pfnSetSpecializationConstants =
        ur_validation_layer::urProgramSetSpecializationConstants;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle =
        ur_validation_layer::urProgramGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle =
        ur_validation_layer::urProgramCreateWithNativeHandle;

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's ProgramExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetProgramExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_program_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.ProgramExp;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnBuildExp = pDdiTable->pfnBuildExp;
    pDdiTable->pfnBuildExp = ur_validation_layer::urProgramBuildExp;

    dditable.pfnCompileExp = pDdiTable->pfnCompileExp;
    pDdiTable->pfnCompileExp = ur_validation_layer::urProgramCompileExp;

    dditable.pfnLinkExp = pDdiTable->pfnLinkExp;
    pDdiTable->pfnLinkExp = ur_validation_layer::urProgramLinkExp;

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
UR_DLLEXPORT ur_result_t UR_APICALL urGetQueueProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_queue_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.Queue;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = ur_validation_layer::urQueueGetInfo;

    dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = ur_validation_layer::urQueueCreate;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = ur_validation_layer::urQueueRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = ur_validation_layer::urQueueRelease;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle = ur_validation_layer::urQueueGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle =
        ur_validation_layer::urQueueCreateWithNativeHandle;

    dditable.pfnFinish = pDdiTable->pfnFinish;
    pDdiTable->pfnFinish = ur_validation_layer::urQueueFinish;

    dditable.pfnFlush = pDdiTable->pfnFlush;
    pDdiTable->pfnFlush = ur_validation_layer::urQueueFlush;

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
UR_DLLEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_sampler_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.Sampler;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = ur_validation_layer::urSamplerCreate;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = ur_validation_layer::urSamplerRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = ur_validation_layer::urSamplerRelease;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = ur_validation_layer::urSamplerGetInfo;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle =
        ur_validation_layer::urSamplerGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle =
        ur_validation_layer::urSamplerCreateWithNativeHandle;

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
UR_DLLEXPORT ur_result_t UR_APICALL urGetUSMProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_usm_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.USM;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnHostAlloc = pDdiTable->pfnHostAlloc;
    pDdiTable->pfnHostAlloc = ur_validation_layer::urUSMHostAlloc;

    dditable.pfnDeviceAlloc = pDdiTable->pfnDeviceAlloc;
    pDdiTable->pfnDeviceAlloc = ur_validation_layer::urUSMDeviceAlloc;

    dditable.pfnSharedAlloc = pDdiTable->pfnSharedAlloc;
    pDdiTable->pfnSharedAlloc = ur_validation_layer::urUSMSharedAlloc;

    dditable.pfnFree = pDdiTable->pfnFree;
    pDdiTable->pfnFree = ur_validation_layer::urUSMFree;

    dditable.pfnGetMemAllocInfo = pDdiTable->pfnGetMemAllocInfo;
    pDdiTable->pfnGetMemAllocInfo = ur_validation_layer::urUSMGetMemAllocInfo;

    dditable.pfnPoolCreate = pDdiTable->pfnPoolCreate;
    pDdiTable->pfnPoolCreate = ur_validation_layer::urUSMPoolCreate;

    dditable.pfnPoolRetain = pDdiTable->pfnPoolRetain;
    pDdiTable->pfnPoolRetain = ur_validation_layer::urUSMPoolRetain;

    dditable.pfnPoolRelease = pDdiTable->pfnPoolRelease;
    pDdiTable->pfnPoolRelease = ur_validation_layer::urUSMPoolRelease;

    dditable.pfnPoolGetInfo = pDdiTable->pfnPoolGetInfo;
    pDdiTable->pfnPoolGetInfo = ur_validation_layer::urUSMPoolGetInfo;

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USMExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetUSMExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_usm_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.USMExp;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnPitchedAllocExp = pDdiTable->pfnPitchedAllocExp;
    pDdiTable->pfnPitchedAllocExp = ur_validation_layer::urUSMPitchedAllocExp;

    dditable.pfnImportExp = pDdiTable->pfnImportExp;
    pDdiTable->pfnImportExp = ur_validation_layer::urUSMImportExp;

    dditable.pfnReleaseExp = pDdiTable->pfnReleaseExp;
    pDdiTable->pfnReleaseExp = ur_validation_layer::urUSMReleaseExp;

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's UsmP2PExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetUsmP2PExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_usm_p2p_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.UsmP2PExp;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnEnablePeerAccessExp = pDdiTable->pfnEnablePeerAccessExp;
    pDdiTable->pfnEnablePeerAccessExp =
        ur_validation_layer::urUsmP2PEnablePeerAccessExp;

    dditable.pfnDisablePeerAccessExp = pDdiTable->pfnDisablePeerAccessExp;
    pDdiTable->pfnDisablePeerAccessExp =
        ur_validation_layer::urUsmP2PDisablePeerAccessExp;

    dditable.pfnPeerAccessGetInfoExp = pDdiTable->pfnPeerAccessGetInfoExp;
    pDdiTable->pfnPeerAccessGetInfoExp =
        ur_validation_layer::urUsmP2PPeerAccessGetInfoExp;

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's VirtualMem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetVirtualMemProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_virtual_mem_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.VirtualMem;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnGranularityGetInfo = pDdiTable->pfnGranularityGetInfo;
    pDdiTable->pfnGranularityGetInfo =
        ur_validation_layer::urVirtualMemGranularityGetInfo;

    dditable.pfnReserve = pDdiTable->pfnReserve;
    pDdiTable->pfnReserve = ur_validation_layer::urVirtualMemReserve;

    dditable.pfnFree = pDdiTable->pfnFree;
    pDdiTable->pfnFree = ur_validation_layer::urVirtualMemFree;

    dditable.pfnMap = pDdiTable->pfnMap;
    pDdiTable->pfnMap = ur_validation_layer::urVirtualMemMap;

    dditable.pfnUnmap = pDdiTable->pfnUnmap;
    pDdiTable->pfnUnmap = ur_validation_layer::urVirtualMemUnmap;

    dditable.pfnSetAccess = pDdiTable->pfnSetAccess;
    pDdiTable->pfnSetAccess = ur_validation_layer::urVirtualMemSetAccess;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = ur_validation_layer::urVirtualMemGetInfo;

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
UR_DLLEXPORT ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_device_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_validation_layer::context.urDdiTable.Device;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_validation_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_validation_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnGet = pDdiTable->pfnGet;
    pDdiTable->pfnGet = ur_validation_layer::urDeviceGet;

    dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    pDdiTable->pfnGetInfo = ur_validation_layer::urDeviceGetInfo;

    dditable.pfnRetain = pDdiTable->pfnRetain;
    pDdiTable->pfnRetain = ur_validation_layer::urDeviceRetain;

    dditable.pfnRelease = pDdiTable->pfnRelease;
    pDdiTable->pfnRelease = ur_validation_layer::urDeviceRelease;

    dditable.pfnPartition = pDdiTable->pfnPartition;
    pDdiTable->pfnPartition = ur_validation_layer::urDevicePartition;

    dditable.pfnSelectBinary = pDdiTable->pfnSelectBinary;
    pDdiTable->pfnSelectBinary = ur_validation_layer::urDeviceSelectBinary;

    dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    pDdiTable->pfnGetNativeHandle =
        ur_validation_layer::urDeviceGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle =
        ur_validation_layer::urDeviceCreateWithNativeHandle;

    dditable.pfnGetGlobalTimestamps = pDdiTable->pfnGetGlobalTimestamps;
    pDdiTable->pfnGetGlobalTimestamps =
        ur_validation_layer::urDeviceGetGlobalTimestamps;

    return result;
}

ur_result_t context_t::init(ur_dditable_t *dditable,
                            const std::set<std::string> &enabledLayerNames,
                            codeloc_data) {
    ur_result_t result = UR_RESULT_SUCCESS;

    if (enabledLayerNames.count(nameFullValidation)) {
        enableParameterValidation = true;
        enableLeakChecking = true;
        enableLifetimeValidation = true;
    } else {
        if (enabledLayerNames.count(nameParameterValidation)) {
            enableParameterValidation = true;
        }
        if (enabledLayerNames.count(nameLeakChecking)) {
            enableLeakChecking = true;
        }
        if (enabledLayerNames.count(nameLifetimeValidation)) {
            // Handle lifetime validation requires leak checking feature.
            enableLifetimeValidation = true;
            enableLeakChecking = true;
        }
    }

    if (!enableParameterValidation && !enableLeakChecking &&
        !enableLifetimeValidation) {
        return result;
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetGlobalProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Global);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetBindlessImagesExpProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->BindlessImagesExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetCommandBufferExpProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->CommandBufferExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetContextProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Context);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetEnqueueProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Enqueue);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetEnqueueExpProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->EnqueueExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetEventProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Event);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetKernelProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Kernel);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetKernelExpProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->KernelExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetMemProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Mem);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetPhysicalMemProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->PhysicalMem);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetPlatformProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Platform);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetProgramProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Program);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetProgramExpProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->ProgramExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetQueueProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Queue);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetSamplerProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Sampler);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetUSMProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->USM);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetUSMExpProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->USMExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetUsmP2PExpProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->UsmP2PExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetVirtualMemProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->VirtualMem);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_validation_layer::urGetDeviceProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Device);
    }

    return result;
}

ur_result_t context_t::tearDown() {
    ur_result_t result = UR_RESULT_SUCCESS;

    if (enableLeakChecking) {
        refCountContext.logInvalidReferences();
        refCountContext.clear();
    }
    return result;
}

} // namespace ur_validation_layer
