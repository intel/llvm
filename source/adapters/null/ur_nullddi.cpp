/*
 *
 * Copyright (C) 2019-2022 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_nullddi.cpp
 *
 */
#include "ur_null.hpp"

namespace driver {
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAdapterGet = d_context.urDdiTable.Global.pfnAdapterGet;
    if (nullptr != pfnAdapterGet) {
        result = pfnAdapterGet(NumEntries, phAdapters, pNumAdapters);
    } else {
        // generic implementation
        for (size_t i = 0; (nullptr != phAdapters) && (i < NumEntries); ++i) {
            phAdapters[i] =
                reinterpret_cast<ur_adapter_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterRelease
__urdlllocal ur_result_t UR_APICALL urAdapterRelease(
    ur_adapter_handle_t hAdapter ///< [in] Adapter handle to release
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAdapterRelease = d_context.urDdiTable.Global.pfnAdapterRelease;
    if (nullptr != pfnAdapterRelease) {
        result = pfnAdapterRelease(hAdapter);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterRetain
__urdlllocal ur_result_t UR_APICALL urAdapterRetain(
    ur_adapter_handle_t hAdapter ///< [in] Adapter handle to retain
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAdapterRetain = d_context.urDdiTable.Global.pfnAdapterRetain;
    if (nullptr != pfnAdapterRetain) {
        result = pfnAdapterRetain(hAdapter);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAdapterGetLastError =
        d_context.urDdiTable.Global.pfnAdapterGetLastError;
    if (nullptr != pfnAdapterGetLastError) {
        result = pfnAdapterGetLastError(hAdapter, ppMessage, pError);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAdapterGetInfo = d_context.urDdiTable.Global.pfnAdapterGetInfo;
    if (nullptr != pfnAdapterGetInfo) {
        result = pfnAdapterGetInfo(hAdapter, propName, propSize, pPropValue,
                                   pPropSizeRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGet = d_context.urDdiTable.Platform.pfnGet;
    if (nullptr != pfnGet) {
        result = pfnGet(phAdapters, NumAdapters, NumEntries, phPlatforms,
                        pNumPlatforms);
    } else {
        // generic implementation
        for (size_t i = 0; (nullptr != phPlatforms) && (i < NumEntries); ++i) {
            phPlatforms[i] =
                reinterpret_cast<ur_platform_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetInfo = d_context.urDdiTable.Platform.pfnGetInfo;
    if (nullptr != pfnGetInfo) {
        result =
            pfnGetInfo(hPlatform, propName, propSize, pPropValue, pPropSizeRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetApiVersion
__urdlllocal ur_result_t UR_APICALL urPlatformGetApiVersion(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform
    ur_api_version_t *pVersion      ///< [out] api version
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetApiVersion = d_context.urDdiTable.Platform.pfnGetApiVersion;
    if (nullptr != pfnGetApiVersion) {
        result = pfnGetApiVersion(hPlatform, pVersion);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urPlatformGetNativeHandle(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform.
    ur_native_handle_t *
        phNativePlatform ///< [out] a pointer to the native handle of the platform.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetNativeHandle = d_context.urDdiTable.Platform.pfnGetNativeHandle;
    if (nullptr != pfnGetNativeHandle) {
        result = pfnGetNativeHandle(hPlatform, phNativePlatform);
    } else {
        // generic implementation
        *phNativePlatform =
            reinterpret_cast<ur_native_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreateWithNativeHandle =
        d_context.urDdiTable.Platform.pfnCreateWithNativeHandle;
    if (nullptr != pfnCreateWithNativeHandle) {
        result =
            pfnCreateWithNativeHandle(hNativePlatform, pProperties, phPlatform);
    } else {
        // generic implementation
        *phPlatform = reinterpret_cast<ur_platform_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetBackendOption =
        d_context.urDdiTable.Platform.pfnGetBackendOption;
    if (nullptr != pfnGetBackendOption) {
        result =
            pfnGetBackendOption(hPlatform, pFrontendOption, ppPlatformOption);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGet = d_context.urDdiTable.Device.pfnGet;
    if (nullptr != pfnGet) {
        result =
            pfnGet(hPlatform, DeviceType, NumEntries, phDevices, pNumDevices);
    } else {
        // generic implementation
        for (size_t i = 0; (nullptr != phDevices) && (i < NumEntries); ++i) {
            phDevices[i] =
                reinterpret_cast<ur_device_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetInfo = d_context.urDdiTable.Device.pfnGetInfo;
    if (nullptr != pfnGetInfo) {
        result =
            pfnGetInfo(hDevice, propName, propSize, pPropValue, pPropSizeRet);
    } else {
        // generic implementation
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_DEVICE_INFO_PLATFORM: {
                ur_platform_handle_t *handles =
                    reinterpret_cast<ur_platform_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_platform_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_platform_handle_t>(d_context.get());
                }
            } break;
            case UR_DEVICE_INFO_PARENT_DEVICE: {
                ur_device_handle_t *handles =
                    reinterpret_cast<ur_device_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_device_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_device_handle_t>(d_context.get());
                }
            } break;
            case UR_DEVICE_INFO_COMPONENT_DEVICES: {
                ur_device_handle_t *handles =
                    reinterpret_cast<ur_device_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_device_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_device_handle_t>(d_context.get());
                }
            } break;
            case UR_DEVICE_INFO_COMPOSITE_DEVICE: {
                ur_device_handle_t *handles =
                    reinterpret_cast<ur_device_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_device_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_device_handle_t>(d_context.get());
                }
            } break;
            default: {
            } break;
            }
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRetain
__urdlllocal ur_result_t UR_APICALL urDeviceRetain(
    ur_device_handle_t
        hDevice ///< [in] handle of the device to get a reference of.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRetain = d_context.urDdiTable.Device.pfnRetain;
    if (nullptr != pfnRetain) {
        result = pfnRetain(hDevice);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRelease
__urdlllocal ur_result_t UR_APICALL urDeviceRelease(
    ur_device_handle_t hDevice ///< [in] handle of the device to release.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRelease = d_context.urDdiTable.Device.pfnRelease;
    if (nullptr != pfnRelease) {
        result = pfnRelease(hDevice);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnPartition = d_context.urDdiTable.Device.pfnPartition;
    if (nullptr != pfnPartition) {
        result = pfnPartition(hDevice, pProperties, NumDevices, phSubDevices,
                              pNumDevicesRet);
    } else {
        // generic implementation
        for (size_t i = 0; (nullptr != phSubDevices) && (i < NumDevices); ++i) {
            phSubDevices[i] =
                reinterpret_cast<ur_device_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSelectBinary = d_context.urDdiTable.Device.pfnSelectBinary;
    if (nullptr != pfnSelectBinary) {
        result =
            pfnSelectBinary(hDevice, pBinaries, NumBinaries, pSelectedBinary);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urDeviceGetNativeHandle(
    ur_device_handle_t hDevice, ///< [in] handle of the device.
    ur_native_handle_t
        *phNativeDevice ///< [out] a pointer to the native handle of the device.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetNativeHandle = d_context.urDdiTable.Device.pfnGetNativeHandle;
    if (nullptr != pfnGetNativeHandle) {
        result = pfnGetNativeHandle(hDevice, phNativeDevice);
    } else {
        // generic implementation
        *phNativeDevice = reinterpret_cast<ur_native_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreateWithNativeHandle =
        d_context.urDdiTable.Device.pfnCreateWithNativeHandle;
    if (nullptr != pfnCreateWithNativeHandle) {
        result = pfnCreateWithNativeHandle(hNativeDevice, hPlatform,
                                           pProperties, phDevice);
    } else {
        // generic implementation
        *phDevice = reinterpret_cast<ur_device_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetGlobalTimestamps =
        d_context.urDdiTable.Device.pfnGetGlobalTimestamps;
    if (nullptr != pfnGetGlobalTimestamps) {
        result =
            pfnGetGlobalTimestamps(hDevice, pDeviceTimestamp, pHostTimestamp);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreate = d_context.urDdiTable.Context.pfnCreate;
    if (nullptr != pfnCreate) {
        result = pfnCreate(DeviceCount, phDevices, pProperties, phContext);
    } else {
        // generic implementation
        *phContext = reinterpret_cast<ur_context_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRetain
__urdlllocal ur_result_t UR_APICALL urContextRetain(
    ur_context_handle_t
        hContext ///< [in] handle of the context to get a reference of.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRetain = d_context.urDdiTable.Context.pfnRetain;
    if (nullptr != pfnRetain) {
        result = pfnRetain(hContext);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRelease
__urdlllocal ur_result_t UR_APICALL urContextRelease(
    ur_context_handle_t hContext ///< [in] handle of the context to release.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRelease = d_context.urDdiTable.Context.pfnRelease;
    if (nullptr != pfnRelease) {
        result = pfnRelease(hContext);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetInfo = d_context.urDdiTable.Context.pfnGetInfo;
    if (nullptr != pfnGetInfo) {
        result =
            pfnGetInfo(hContext, propName, propSize, pPropValue, pPropSizeRet);
    } else {
        // generic implementation
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_CONTEXT_INFO_DEVICES: {
                ur_device_handle_t *handles =
                    reinterpret_cast<ur_device_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_device_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_device_handle_t>(d_context.get());
                }
            } break;
            default: {
            } break;
            }
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urContextGetNativeHandle(
    ur_context_handle_t hContext, ///< [in] handle of the context.
    ur_native_handle_t *
        phNativeContext ///< [out] a pointer to the native handle of the context.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetNativeHandle = d_context.urDdiTable.Context.pfnGetNativeHandle;
    if (nullptr != pfnGetNativeHandle) {
        result = pfnGetNativeHandle(hContext, phNativeContext);
    } else {
        // generic implementation
        *phNativeContext =
            reinterpret_cast<ur_native_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreateWithNativeHandle =
        d_context.urDdiTable.Context.pfnCreateWithNativeHandle;
    if (nullptr != pfnCreateWithNativeHandle) {
        result = pfnCreateWithNativeHandle(hNativeContext, numDevices,
                                           phDevices, pProperties, phContext);
    } else {
        // generic implementation
        *phContext = reinterpret_cast<ur_context_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextSetExtendedDeleter
__urdlllocal ur_result_t UR_APICALL urContextSetExtendedDeleter(
    ur_context_handle_t hContext, ///< [in] handle of the context.
    ur_context_extended_deleter_t
        pfnDeleter, ///< [in] Function pointer to extended deleter.
    void *
        pUserData ///< [in][out][optional] pointer to data to be passed to callback.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSetExtendedDeleter =
        d_context.urDdiTable.Context.pfnSetExtendedDeleter;
    if (nullptr != pfnSetExtendedDeleter) {
        result = pfnSetExtendedDeleter(hContext, pfnDeleter, pUserData);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnImageCreate = d_context.urDdiTable.Mem.pfnImageCreate;
    if (nullptr != pfnImageCreate) {
        result = pfnImageCreate(hContext, flags, pImageFormat, pImageDesc,
                                pHost, phMem);
    } else {
        // generic implementation
        *phMem = reinterpret_cast<ur_mem_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnBufferCreate = d_context.urDdiTable.Mem.pfnBufferCreate;
    if (nullptr != pfnBufferCreate) {
        result = pfnBufferCreate(hContext, flags, size, pProperties, phBuffer);
    } else {
        // generic implementation
        *phBuffer = reinterpret_cast<ur_mem_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRetain
__urdlllocal ur_result_t UR_APICALL urMemRetain(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to get access
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRetain = d_context.urDdiTable.Mem.pfnRetain;
    if (nullptr != pfnRetain) {
        result = pfnRetain(hMem);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRelease
__urdlllocal ur_result_t UR_APICALL urMemRelease(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to release
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRelease = d_context.urDdiTable.Mem.pfnRelease;
    if (nullptr != pfnRelease) {
        result = pfnRelease(hMem);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnBufferPartition = d_context.urDdiTable.Mem.pfnBufferPartition;
    if (nullptr != pfnBufferPartition) {
        result = pfnBufferPartition(hBuffer, flags, bufferCreateType, pRegion,
                                    phMem);
    } else {
        // generic implementation
        *phMem = reinterpret_cast<ur_mem_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urMemGetNativeHandle(
    ur_mem_handle_t hMem, ///< [in] handle of the mem.
    ur_device_handle_t
        hDevice, ///< [in] handle of the device that the native handle will be resident on.
    ur_native_handle_t
        *phNativeMem ///< [out] a pointer to the native handle of the mem.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetNativeHandle = d_context.urDdiTable.Mem.pfnGetNativeHandle;
    if (nullptr != pfnGetNativeHandle) {
        result = pfnGetNativeHandle(hMem, hDevice, phNativeMem);
    } else {
        // generic implementation
        *phNativeMem = reinterpret_cast<ur_native_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnBufferCreateWithNativeHandle =
        d_context.urDdiTable.Mem.pfnBufferCreateWithNativeHandle;
    if (nullptr != pfnBufferCreateWithNativeHandle) {
        result = pfnBufferCreateWithNativeHandle(hNativeMem, hContext,
                                                 pProperties, phMem);
    } else {
        // generic implementation
        *phMem = reinterpret_cast<ur_mem_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnImageCreateWithNativeHandle =
        d_context.urDdiTable.Mem.pfnImageCreateWithNativeHandle;
    if (nullptr != pfnImageCreateWithNativeHandle) {
        result = pfnImageCreateWithNativeHandle(
            hNativeMem, hContext, pImageFormat, pImageDesc, pProperties, phMem);
    } else {
        // generic implementation
        *phMem = reinterpret_cast<ur_mem_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetInfo = d_context.urDdiTable.Mem.pfnGetInfo;
    if (nullptr != pfnGetInfo) {
        result =
            pfnGetInfo(hMemory, propName, propSize, pPropValue, pPropSizeRet);
    } else {
        // generic implementation
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_MEM_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_context_handle_t>(d_context.get());
                }
            } break;
            default: {
            } break;
            }
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnImageGetInfo = d_context.urDdiTable.Mem.pfnImageGetInfo;
    if (nullptr != pfnImageGetInfo) {
        result = pfnImageGetInfo(hMemory, propName, propSize, pPropValue,
                                 pPropSizeRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerCreate
__urdlllocal ur_result_t UR_APICALL urSamplerCreate(
    ur_context_handle_t hContext,   ///< [in] handle of the context object
    const ur_sampler_desc_t *pDesc, ///< [in] pointer to the sampler description
    ur_sampler_handle_t
        *phSampler ///< [out] pointer to handle of sampler object created
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreate = d_context.urDdiTable.Sampler.pfnCreate;
    if (nullptr != pfnCreate) {
        result = pfnCreate(hContext, pDesc, phSampler);
    } else {
        // generic implementation
        *phSampler = reinterpret_cast<ur_sampler_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRetain
__urdlllocal ur_result_t UR_APICALL urSamplerRetain(
    ur_sampler_handle_t
        hSampler ///< [in] handle of the sampler object to get access
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRetain = d_context.urDdiTable.Sampler.pfnRetain;
    if (nullptr != pfnRetain) {
        result = pfnRetain(hSampler);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRelease
__urdlllocal ur_result_t UR_APICALL urSamplerRelease(
    ur_sampler_handle_t
        hSampler ///< [in] handle of the sampler object to release
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRelease = d_context.urDdiTable.Sampler.pfnRelease;
    if (nullptr != pfnRelease) {
        result = pfnRelease(hSampler);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetInfo = d_context.urDdiTable.Sampler.pfnGetInfo;
    if (nullptr != pfnGetInfo) {
        result =
            pfnGetInfo(hSampler, propName, propSize, pPropValue, pPropSizeRet);
    } else {
        // generic implementation
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_SAMPLER_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_context_handle_t>(d_context.get());
                }
            } break;
            default: {
            } break;
            }
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urSamplerGetNativeHandle(
    ur_sampler_handle_t hSampler, ///< [in] handle of the sampler.
    ur_native_handle_t *
        phNativeSampler ///< [out] a pointer to the native handle of the sampler.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetNativeHandle = d_context.urDdiTable.Sampler.pfnGetNativeHandle;
    if (nullptr != pfnGetNativeHandle) {
        result = pfnGetNativeHandle(hSampler, phNativeSampler);
    } else {
        // generic implementation
        *phNativeSampler =
            reinterpret_cast<ur_native_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreateWithNativeHandle =
        d_context.urDdiTable.Sampler.pfnCreateWithNativeHandle;
    if (nullptr != pfnCreateWithNativeHandle) {
        result = pfnCreateWithNativeHandle(hNativeSampler, hContext,
                                           pProperties, phSampler);
    } else {
        // generic implementation
        *phSampler = reinterpret_cast<ur_sampler_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnHostAlloc = d_context.urDdiTable.USM.pfnHostAlloc;
    if (nullptr != pfnHostAlloc) {
        result = pfnHostAlloc(hContext, pUSMDesc, pool, size, ppMem);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnDeviceAlloc = d_context.urDdiTable.USM.pfnDeviceAlloc;
    if (nullptr != pfnDeviceAlloc) {
        result = pfnDeviceAlloc(hContext, hDevice, pUSMDesc, pool, size, ppMem);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSharedAlloc = d_context.urDdiTable.USM.pfnSharedAlloc;
    if (nullptr != pfnSharedAlloc) {
        result = pfnSharedAlloc(hContext, hDevice, pUSMDesc, pool, size, ppMem);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMFree
__urdlllocal ur_result_t UR_APICALL urUSMFree(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to USM memory object
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnFree = d_context.urDdiTable.USM.pfnFree;
    if (nullptr != pfnFree) {
        result = pfnFree(hContext, pMem);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetMemAllocInfo = d_context.urDdiTable.USM.pfnGetMemAllocInfo;
    if (nullptr != pfnGetMemAllocInfo) {
        result = pfnGetMemAllocInfo(hContext, pMem, propName, propSize,
                                    pPropValue, pPropSizeRet);
    } else {
        // generic implementation
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_USM_ALLOC_INFO_DEVICE: {
                ur_device_handle_t *handles =
                    reinterpret_cast<ur_device_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_device_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_device_handle_t>(d_context.get());
                }
            } break;
            case UR_USM_ALLOC_INFO_POOL: {
                ur_usm_pool_handle_t *handles =
                    reinterpret_cast<ur_usm_pool_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_usm_pool_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_usm_pool_handle_t>(d_context.get());
                }
            } break;
            default: {
            } break;
            }
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolCreate
__urdlllocal ur_result_t UR_APICALL urUSMPoolCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_usm_pool_desc_t *
        pPoolDesc, ///< [in] pointer to USM pool descriptor. Can be chained with
                   ///< ::ur_usm_pool_limits_desc_t
    ur_usm_pool_handle_t *ppPool ///< [out] pointer to USM memory pool
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnPoolCreate = d_context.urDdiTable.USM.pfnPoolCreate;
    if (nullptr != pfnPoolCreate) {
        result = pfnPoolCreate(hContext, pPoolDesc, ppPool);
    } else {
        // generic implementation
        *ppPool = reinterpret_cast<ur_usm_pool_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolRetain
__urdlllocal ur_result_t UR_APICALL urUSMPoolRetain(
    ur_usm_pool_handle_t pPool ///< [in] pointer to USM memory pool
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnPoolRetain = d_context.urDdiTable.USM.pfnPoolRetain;
    if (nullptr != pfnPoolRetain) {
        result = pfnPoolRetain(pPool);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolRelease
__urdlllocal ur_result_t UR_APICALL urUSMPoolRelease(
    ur_usm_pool_handle_t pPool ///< [in] pointer to USM memory pool
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnPoolRelease = d_context.urDdiTable.USM.pfnPoolRelease;
    if (nullptr != pfnPoolRelease) {
        result = pfnPoolRelease(pPool);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnPoolGetInfo = d_context.urDdiTable.USM.pfnPoolGetInfo;
    if (nullptr != pfnPoolGetInfo) {
        result =
            pfnPoolGetInfo(hPool, propName, propSize, pPropValue, pPropSizeRet);
    } else {
        // generic implementation
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_USM_POOL_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_context_handle_t>(d_context.get());
                }
            } break;
            default: {
            } break;
            }
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGranularityGetInfo =
        d_context.urDdiTable.VirtualMem.pfnGranularityGetInfo;
    if (nullptr != pfnGranularityGetInfo) {
        result = pfnGranularityGetInfo(hContext, hDevice, propName, propSize,
                                       pPropValue, pPropSizeRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnReserve = d_context.urDdiTable.VirtualMem.pfnReserve;
    if (nullptr != pfnReserve) {
        result = pfnReserve(hContext, pStart, size, ppStart);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemFree
__urdlllocal ur_result_t UR_APICALL urVirtualMemFree(
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    const void *
        pStart, ///< [in] pointer to the start of the virtual memory range to free.
    size_t size ///< [in] size in bytes of the virtual memory range to free.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnFree = d_context.urDdiTable.VirtualMem.pfnFree;
    if (nullptr != pfnFree) {
        result = pfnFree(hContext, pStart, size);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMap = d_context.urDdiTable.VirtualMem.pfnMap;
    if (nullptr != pfnMap) {
        result = pfnMap(hContext, pStart, size, hPhysicalMem, offset, flags);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemUnmap
__urdlllocal ur_result_t UR_APICALL urVirtualMemUnmap(
    ur_context_handle_t hContext, ///< [in] handle to the context object.
    const void *
        pStart, ///< [in] pointer to the start of the mapped virtual memory range
    size_t size ///< [in] size in bytes of the virtual memory range.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnUnmap = d_context.urDdiTable.VirtualMem.pfnUnmap;
    if (nullptr != pfnUnmap) {
        result = pfnUnmap(hContext, pStart, size);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSetAccess = d_context.urDdiTable.VirtualMem.pfnSetAccess;
    if (nullptr != pfnSetAccess) {
        result = pfnSetAccess(hContext, pStart, size, flags);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetInfo = d_context.urDdiTable.VirtualMem.pfnGetInfo;
    if (nullptr != pfnGetInfo) {
        result = pfnGetInfo(hContext, pStart, size, propName, propSize,
                            pPropValue, pPropSizeRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreate = d_context.urDdiTable.PhysicalMem.pfnCreate;
    if (nullptr != pfnCreate) {
        result = pfnCreate(hContext, hDevice, size, pProperties, phPhysicalMem);
    } else {
        // generic implementation
        *phPhysicalMem =
            reinterpret_cast<ur_physical_mem_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemRetain
__urdlllocal ur_result_t UR_APICALL urPhysicalMemRetain(
    ur_physical_mem_handle_t
        hPhysicalMem ///< [in] handle of the physical memory object to retain.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRetain = d_context.urDdiTable.PhysicalMem.pfnRetain;
    if (nullptr != pfnRetain) {
        result = pfnRetain(hPhysicalMem);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemRelease
__urdlllocal ur_result_t UR_APICALL urPhysicalMemRelease(
    ur_physical_mem_handle_t
        hPhysicalMem ///< [in] handle of the physical memory object to release.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRelease = d_context.urDdiTable.PhysicalMem.pfnRelease;
    if (nullptr != pfnRelease) {
        result = pfnRelease(hPhysicalMem);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreateWithIL = d_context.urDdiTable.Program.pfnCreateWithIL;
    if (nullptr != pfnCreateWithIL) {
        result = pfnCreateWithIL(hContext, pIL, length, pProperties, phProgram);
    } else {
        // generic implementation
        *phProgram = reinterpret_cast<ur_program_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreateWithBinary = d_context.urDdiTable.Program.pfnCreateWithBinary;
    if (nullptr != pfnCreateWithBinary) {
        result = pfnCreateWithBinary(hContext, hDevice, size, pBinary,
                                     pProperties, phProgram);
    } else {
        // generic implementation
        *phProgram = reinterpret_cast<ur_program_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramBuild
__urdlllocal ur_result_t UR_APICALL urProgramBuild(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    ur_program_handle_t hProgram, ///< [in] Handle of the program to build.
    const char *
        pOptions ///< [in][optional] pointer to build options null-terminated string.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnBuild = d_context.urDdiTable.Program.pfnBuild;
    if (nullptr != pfnBuild) {
        result = pfnBuild(hContext, hProgram, pOptions);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCompile
__urdlllocal ur_result_t UR_APICALL urProgramCompile(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    ur_program_handle_t
        hProgram, ///< [in][out] handle of the program to compile.
    const char *
        pOptions ///< [in][optional] pointer to build options null-terminated string.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCompile = d_context.urDdiTable.Program.pfnCompile;
    if (nullptr != pfnCompile) {
        result = pfnCompile(hContext, hProgram, pOptions);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnLink = d_context.urDdiTable.Program.pfnLink;
    if (nullptr != pfnLink) {
        result = pfnLink(hContext, count, phPrograms, pOptions, phProgram);
    } else {
        // generic implementation
        *phProgram = reinterpret_cast<ur_program_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRetain
__urdlllocal ur_result_t UR_APICALL urProgramRetain(
    ur_program_handle_t hProgram ///< [in] handle for the Program to retain
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRetain = d_context.urDdiTable.Program.pfnRetain;
    if (nullptr != pfnRetain) {
        result = pfnRetain(hProgram);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRelease
__urdlllocal ur_result_t UR_APICALL urProgramRelease(
    ur_program_handle_t hProgram ///< [in] handle for the Program to release
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRelease = d_context.urDdiTable.Program.pfnRelease;
    if (nullptr != pfnRelease) {
        result = pfnRelease(hProgram);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetFunctionPointer =
        d_context.urDdiTable.Program.pfnGetFunctionPointer;
    if (nullptr != pfnGetFunctionPointer) {
        result = pfnGetFunctionPointer(hDevice, hProgram, pFunctionName,
                                       ppFunctionPointer);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetInfo = d_context.urDdiTable.Program.pfnGetInfo;
    if (nullptr != pfnGetInfo) {
        result =
            pfnGetInfo(hProgram, propName, propSize, pPropValue, pPropSizeRet);
    } else {
        // generic implementation
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_PROGRAM_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_context_handle_t>(d_context.get());
                }
            } break;
            case UR_PROGRAM_INFO_DEVICES: {
                ur_device_handle_t *handles =
                    reinterpret_cast<ur_device_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_device_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_device_handle_t>(d_context.get());
                }
            } break;
            default: {
            } break;
            }
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetBuildInfo = d_context.urDdiTable.Program.pfnGetBuildInfo;
    if (nullptr != pfnGetBuildInfo) {
        result = pfnGetBuildInfo(hProgram, hDevice, propName, propSize,
                                 pPropValue, pPropSizeRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramSetSpecializationConstants
__urdlllocal ur_result_t UR_APICALL urProgramSetSpecializationConstants(
    ur_program_handle_t hProgram, ///< [in] handle of the Program object
    uint32_t count, ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t *
        pSpecConstants ///< [in][range(0, count)] array of specialization constant value
                       ///< descriptions
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSetSpecializationConstants =
        d_context.urDdiTable.Program.pfnSetSpecializationConstants;
    if (nullptr != pfnSetSpecializationConstants) {
        result = pfnSetSpecializationConstants(hProgram, count, pSpecConstants);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t hProgram, ///< [in] handle of the program.
    ur_native_handle_t *
        phNativeProgram ///< [out] a pointer to the native handle of the program.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetNativeHandle = d_context.urDdiTable.Program.pfnGetNativeHandle;
    if (nullptr != pfnGetNativeHandle) {
        result = pfnGetNativeHandle(hProgram, phNativeProgram);
    } else {
        // generic implementation
        *phNativeProgram =
            reinterpret_cast<ur_native_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreateWithNativeHandle =
        d_context.urDdiTable.Program.pfnCreateWithNativeHandle;
    if (nullptr != pfnCreateWithNativeHandle) {
        result = pfnCreateWithNativeHandle(hNativeProgram, hContext,
                                           pProperties, phProgram);
    } else {
        // generic implementation
        *phProgram = reinterpret_cast<ur_program_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelCreate
__urdlllocal ur_result_t UR_APICALL urKernelCreate(
    ur_program_handle_t hProgram, ///< [in] handle of the program instance
    const char *pKernelName,      ///< [in] pointer to null-terminated string.
    ur_kernel_handle_t
        *phKernel ///< [out] pointer to handle of kernel object created.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreate = d_context.urDdiTable.Kernel.pfnCreate;
    if (nullptr != pfnCreate) {
        result = pfnCreate(hProgram, pKernelName, phKernel);
    } else {
        // generic implementation
        *phKernel = reinterpret_cast<ur_kernel_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSetArgValue = d_context.urDdiTable.Kernel.pfnSetArgValue;
    if (nullptr != pfnSetArgValue) {
        result =
            pfnSetArgValue(hKernel, argIndex, argSize, pProperties, pArgValue);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSetArgLocal = d_context.urDdiTable.Kernel.pfnSetArgLocal;
    if (nullptr != pfnSetArgLocal) {
        result = pfnSetArgLocal(hKernel, argIndex, argSize, pProperties);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetInfo = d_context.urDdiTable.Kernel.pfnGetInfo;
    if (nullptr != pfnGetInfo) {
        result =
            pfnGetInfo(hKernel, propName, propSize, pPropValue, pPropSizeRet);
    } else {
        // generic implementation
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_KERNEL_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_context_handle_t>(d_context.get());
                }
            } break;
            case UR_KERNEL_INFO_PROGRAM: {
                ur_program_handle_t *handles =
                    reinterpret_cast<ur_program_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_program_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_program_handle_t>(d_context.get());
                }
            } break;
            default: {
            } break;
            }
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetGroupInfo = d_context.urDdiTable.Kernel.pfnGetGroupInfo;
    if (nullptr != pfnGetGroupInfo) {
        result = pfnGetGroupInfo(hKernel, hDevice, propName, propSize,
                                 pPropValue, pPropSizeRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetSubGroupInfo = d_context.urDdiTable.Kernel.pfnGetSubGroupInfo;
    if (nullptr != pfnGetSubGroupInfo) {
        result = pfnGetSubGroupInfo(hKernel, hDevice, propName, propSize,
                                    pPropValue, pPropSizeRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRetain
__urdlllocal ur_result_t UR_APICALL urKernelRetain(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to retain
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRetain = d_context.urDdiTable.Kernel.pfnRetain;
    if (nullptr != pfnRetain) {
        result = pfnRetain(hKernel);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRelease
__urdlllocal ur_result_t UR_APICALL urKernelRelease(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to release
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRelease = d_context.urDdiTable.Kernel.pfnRelease;
    if (nullptr != pfnRelease) {
        result = pfnRelease(hKernel);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSetArgPointer = d_context.urDdiTable.Kernel.pfnSetArgPointer;
    if (nullptr != pfnSetArgPointer) {
        result = pfnSetArgPointer(hKernel, argIndex, pProperties, pArgValue);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSetExecInfo = d_context.urDdiTable.Kernel.pfnSetExecInfo;
    if (nullptr != pfnSetExecInfo) {
        result = pfnSetExecInfo(hKernel, propName, propSize, pProperties,
                                pPropValue);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgSampler
__urdlllocal ur_result_t UR_APICALL urKernelSetArgSampler(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_sampler_properties_t
        *pProperties, ///< [in][optional] pointer to sampler properties.
    ur_sampler_handle_t hArgValue ///< [in] handle of Sampler object.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSetArgSampler = d_context.urDdiTable.Kernel.pfnSetArgSampler;
    if (nullptr != pfnSetArgSampler) {
        result = pfnSetArgSampler(hKernel, argIndex, pProperties, hArgValue);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgMemObj
__urdlllocal ur_result_t UR_APICALL urKernelSetArgMemObj(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_mem_obj_properties_t
        *pProperties, ///< [in][optional] pointer to Memory object properties.
    ur_mem_handle_t hArgValue ///< [in][optional] handle of Memory object.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSetArgMemObj = d_context.urDdiTable.Kernel.pfnSetArgMemObj;
    if (nullptr != pfnSetArgMemObj) {
        result = pfnSetArgMemObj(hKernel, argIndex, pProperties, hArgValue);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetSpecializationConstants
__urdlllocal ur_result_t UR_APICALL urKernelSetSpecializationConstants(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t count, ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t *
        pSpecConstants ///< [in] array of specialization constant value descriptions
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSetSpecializationConstants =
        d_context.urDdiTable.Kernel.pfnSetSpecializationConstants;
    if (nullptr != pfnSetSpecializationConstants) {
        result = pfnSetSpecializationConstants(hKernel, count, pSpecConstants);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urKernelGetNativeHandle(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel.
    ur_native_handle_t
        *phNativeKernel ///< [out] a pointer to the native handle of the kernel.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetNativeHandle = d_context.urDdiTable.Kernel.pfnGetNativeHandle;
    if (nullptr != pfnGetNativeHandle) {
        result = pfnGetNativeHandle(hKernel, phNativeKernel);
    } else {
        // generic implementation
        *phNativeKernel = reinterpret_cast<ur_native_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreateWithNativeHandle =
        d_context.urDdiTable.Kernel.pfnCreateWithNativeHandle;
    if (nullptr != pfnCreateWithNativeHandle) {
        result = pfnCreateWithNativeHandle(hNativeKernel, hContext, hProgram,
                                           pProperties, phKernel);
    } else {
        // generic implementation
        *phKernel = reinterpret_cast<ur_kernel_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetInfo = d_context.urDdiTable.Queue.pfnGetInfo;
    if (nullptr != pfnGetInfo) {
        result =
            pfnGetInfo(hQueue, propName, propSize, pPropValue, pPropSizeRet);
    } else {
        // generic implementation
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_QUEUE_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_context_handle_t>(d_context.get());
                }
            } break;
            case UR_QUEUE_INFO_DEVICE: {
                ur_device_handle_t *handles =
                    reinterpret_cast<ur_device_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_device_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_device_handle_t>(d_context.get());
                }
            } break;
            case UR_QUEUE_INFO_DEVICE_DEFAULT: {
                ur_queue_handle_t *handles =
                    reinterpret_cast<ur_queue_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_queue_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_queue_handle_t>(d_context.get());
                }
            } break;
            default: {
            } break;
            }
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreate = d_context.urDdiTable.Queue.pfnCreate;
    if (nullptr != pfnCreate) {
        result = pfnCreate(hContext, hDevice, pProperties, phQueue);
    } else {
        // generic implementation
        *phQueue = reinterpret_cast<ur_queue_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRetain
__urdlllocal ur_result_t UR_APICALL urQueueRetain(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to get access
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRetain = d_context.urDdiTable.Queue.pfnRetain;
    if (nullptr != pfnRetain) {
        result = pfnRetain(hQueue);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRelease
__urdlllocal ur_result_t UR_APICALL urQueueRelease(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to release
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRelease = d_context.urDdiTable.Queue.pfnRelease;
    if (nullptr != pfnRelease) {
        result = pfnRelease(hQueue);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urQueueGetNativeHandle(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue.
    ur_queue_native_desc_t
        *pDesc, ///< [in][optional] pointer to native descriptor
    ur_native_handle_t
        *phNativeQueue ///< [out] a pointer to the native handle of the queue.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetNativeHandle = d_context.urDdiTable.Queue.pfnGetNativeHandle;
    if (nullptr != pfnGetNativeHandle) {
        result = pfnGetNativeHandle(hQueue, pDesc, phNativeQueue);
    } else {
        // generic implementation
        *phNativeQueue = reinterpret_cast<ur_native_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreateWithNativeHandle =
        d_context.urDdiTable.Queue.pfnCreateWithNativeHandle;
    if (nullptr != pfnCreateWithNativeHandle) {
        result = pfnCreateWithNativeHandle(hNativeQueue, hContext, hDevice,
                                           pProperties, phQueue);
    } else {
        // generic implementation
        *phQueue = reinterpret_cast<ur_queue_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFinish
__urdlllocal ur_result_t UR_APICALL urQueueFinish(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be finished.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnFinish = d_context.urDdiTable.Queue.pfnFinish;
    if (nullptr != pfnFinish) {
        result = pfnFinish(hQueue);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFlush
__urdlllocal ur_result_t UR_APICALL urQueueFlush(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be flushed.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnFlush = d_context.urDdiTable.Queue.pfnFlush;
    if (nullptr != pfnFlush) {
        result = pfnFlush(hQueue);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetInfo = d_context.urDdiTable.Event.pfnGetInfo;
    if (nullptr != pfnGetInfo) {
        result =
            pfnGetInfo(hEvent, propName, propSize, pPropValue, pPropSizeRet);
    } else {
        // generic implementation
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_EVENT_INFO_COMMAND_QUEUE: {
                ur_queue_handle_t *handles =
                    reinterpret_cast<ur_queue_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_queue_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_queue_handle_t>(d_context.get());
                }
            } break;
            case UR_EVENT_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = propSize / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    handles[i] =
                        reinterpret_cast<ur_context_handle_t>(d_context.get());
                }
            } break;
            default: {
            } break;
            }
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetProfilingInfo = d_context.urDdiTable.Event.pfnGetProfilingInfo;
    if (nullptr != pfnGetProfilingInfo) {
        result = pfnGetProfilingInfo(hEvent, propName, propSize, pPropValue,
                                     pPropSizeRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventWait
__urdlllocal ur_result_t UR_APICALL urEventWait(
    uint32_t numEvents, ///< [in] number of events in the event list
    const ur_event_handle_t *
        phEventWaitList ///< [in][range(0, numEvents)] pointer to a list of events to wait for
                        ///< completion
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnWait = d_context.urDdiTable.Event.pfnWait;
    if (nullptr != pfnWait) {
        result = pfnWait(numEvents, phEventWaitList);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRetain
__urdlllocal ur_result_t UR_APICALL urEventRetain(
    ur_event_handle_t hEvent ///< [in] handle of the event object
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRetain = d_context.urDdiTable.Event.pfnRetain;
    if (nullptr != pfnRetain) {
        result = pfnRetain(hEvent);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRelease
__urdlllocal ur_result_t UR_APICALL urEventRelease(
    ur_event_handle_t hEvent ///< [in] handle of the event object
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRelease = d_context.urDdiTable.Event.pfnRelease;
    if (nullptr != pfnRelease) {
        result = pfnRelease(hEvent);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urEventGetNativeHandle(
    ur_event_handle_t hEvent, ///< [in] handle of the event.
    ur_native_handle_t
        *phNativeEvent ///< [out] a pointer to the native handle of the event.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetNativeHandle = d_context.urDdiTable.Event.pfnGetNativeHandle;
    if (nullptr != pfnGetNativeHandle) {
        result = pfnGetNativeHandle(hEvent, phNativeEvent);
    } else {
        // generic implementation
        *phNativeEvent = reinterpret_cast<ur_native_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreateWithNativeHandle =
        d_context.urDdiTable.Event.pfnCreateWithNativeHandle;
    if (nullptr != pfnCreateWithNativeHandle) {
        result = pfnCreateWithNativeHandle(hNativeEvent, hContext, pProperties,
                                           phEvent);
    } else {
        // generic implementation
        *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventSetCallback
__urdlllocal ur_result_t UR_APICALL urEventSetCallback(
    ur_event_handle_t hEvent,       ///< [in] handle of the event object
    ur_execution_info_t execStatus, ///< [in] execution status of the event
    ur_event_callback_t pfnNotify,  ///< [in] execution status of the event
    void *
        pUserData ///< [in][out][optional] pointer to data to be passed to callback.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSetCallback = d_context.urDdiTable.Event.pfnSetCallback;
    if (nullptr != pfnSetCallback) {
        result = pfnSetCallback(hEvent, execStatus, pfnNotify, pUserData);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnKernelLaunch = d_context.urDdiTable.Enqueue.pfnKernelLaunch;
    if (nullptr != pfnKernelLaunch) {
        result = pfnKernelLaunch(hQueue, hKernel, workDim, pGlobalWorkOffset,
                                 pGlobalWorkSize, pLocalWorkSize,
                                 numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnEventsWait = d_context.urDdiTable.Enqueue.pfnEventsWait;
    if (nullptr != pfnEventsWait) {
        result = pfnEventsWait(hQueue, numEventsInWaitList, phEventWaitList,
                               phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnEventsWaitWithBarrier =
        d_context.urDdiTable.Enqueue.pfnEventsWaitWithBarrier;
    if (nullptr != pfnEventsWaitWithBarrier) {
        result = pfnEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                          phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMemBufferRead = d_context.urDdiTable.Enqueue.pfnMemBufferRead;
    if (nullptr != pfnMemBufferRead) {
        result =
            pfnMemBufferRead(hQueue, hBuffer, blockingRead, offset, size, pDst,
                             numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMemBufferWrite = d_context.urDdiTable.Enqueue.pfnMemBufferWrite;
    if (nullptr != pfnMemBufferWrite) {
        result = pfnMemBufferWrite(hQueue, hBuffer, blockingWrite, offset, size,
                                   pSrc, numEventsInWaitList, phEventWaitList,
                                   phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMemBufferReadRect =
        d_context.urDdiTable.Enqueue.pfnMemBufferReadRect;
    if (nullptr != pfnMemBufferReadRect) {
        result = pfnMemBufferReadRect(
            hQueue, hBuffer, blockingRead, bufferOrigin, hostOrigin, region,
            bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch,
            pDst, numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMemBufferWriteRect =
        d_context.urDdiTable.Enqueue.pfnMemBufferWriteRect;
    if (nullptr != pfnMemBufferWriteRect) {
        result = pfnMemBufferWriteRect(
            hQueue, hBuffer, blockingWrite, bufferOrigin, hostOrigin, region,
            bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch,
            pSrc, numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMemBufferCopy = d_context.urDdiTable.Enqueue.pfnMemBufferCopy;
    if (nullptr != pfnMemBufferCopy) {
        result = pfnMemBufferCopy(hQueue, hBufferSrc, hBufferDst, srcOffset,
                                  dstOffset, size, numEventsInWaitList,
                                  phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMemBufferCopyRect =
        d_context.urDdiTable.Enqueue.pfnMemBufferCopyRect;
    if (nullptr != pfnMemBufferCopyRect) {
        result = pfnMemBufferCopyRect(
            hQueue, hBufferSrc, hBufferDst, srcOrigin, dstOrigin, region,
            srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch,
            numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMemBufferFill = d_context.urDdiTable.Enqueue.pfnMemBufferFill;
    if (nullptr != pfnMemBufferFill) {
        result = pfnMemBufferFill(hQueue, hBuffer, pPattern, patternSize,
                                  offset, size, numEventsInWaitList,
                                  phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMemImageRead = d_context.urDdiTable.Enqueue.pfnMemImageRead;
    if (nullptr != pfnMemImageRead) {
        result = pfnMemImageRead(hQueue, hImage, blockingRead, origin, region,
                                 rowPitch, slicePitch, pDst,
                                 numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMemImageWrite = d_context.urDdiTable.Enqueue.pfnMemImageWrite;
    if (nullptr != pfnMemImageWrite) {
        result = pfnMemImageWrite(
            hQueue, hImage, blockingWrite, origin, region, rowPitch, slicePitch,
            pSrc, numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMemImageCopy = d_context.urDdiTable.Enqueue.pfnMemImageCopy;
    if (nullptr != pfnMemImageCopy) {
        result = pfnMemImageCopy(hQueue, hImageSrc, hImageDst, srcOrigin,
                                 dstOrigin, region, numEventsInWaitList,
                                 phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMemBufferMap = d_context.urDdiTable.Enqueue.pfnMemBufferMap;
    if (nullptr != pfnMemBufferMap) {
        result = pfnMemBufferMap(hQueue, hBuffer, blockingMap, mapFlags, offset,
                                 size, numEventsInWaitList, phEventWaitList,
                                 phEvent, ppRetMap);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMemUnmap = d_context.urDdiTable.Enqueue.pfnMemUnmap;
    if (nullptr != pfnMemUnmap) {
        result = pfnMemUnmap(hQueue, hMem, pMappedPtr, numEventsInWaitList,
                             phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnUSMFill = d_context.urDdiTable.Enqueue.pfnUSMFill;
    if (nullptr != pfnUSMFill) {
        result = pfnUSMFill(hQueue, pMem, patternSize, pPattern, size,
                            numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnUSMMemcpy = d_context.urDdiTable.Enqueue.pfnUSMMemcpy;
    if (nullptr != pfnUSMMemcpy) {
        result = pfnUSMMemcpy(hQueue, blocking, pDst, pSrc, size,
                              numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnUSMPrefetch = d_context.urDdiTable.Enqueue.pfnUSMPrefetch;
    if (nullptr != pfnUSMPrefetch) {
        result = pfnUSMPrefetch(hQueue, pMem, size, flags, numEventsInWaitList,
                                phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnUSMAdvise = d_context.urDdiTable.Enqueue.pfnUSMAdvise;
    if (nullptr != pfnUSMAdvise) {
        result = pfnUSMAdvise(hQueue, pMem, size, advice, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnUSMFill2D = d_context.urDdiTable.Enqueue.pfnUSMFill2D;
    if (nullptr != pfnUSMFill2D) {
        result =
            pfnUSMFill2D(hQueue, pMem, pitch, patternSize, pPattern, width,
                         height, numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnUSMMemcpy2D = d_context.urDdiTable.Enqueue.pfnUSMMemcpy2D;
    if (nullptr != pfnUSMMemcpy2D) {
        result = pfnUSMMemcpy2D(hQueue, blocking, pDst, dstPitch, pSrc,
                                srcPitch, width, height, numEventsInWaitList,
                                phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnDeviceGlobalVariableWrite =
        d_context.urDdiTable.Enqueue.pfnDeviceGlobalVariableWrite;
    if (nullptr != pfnDeviceGlobalVariableWrite) {
        result = pfnDeviceGlobalVariableWrite(
            hQueue, hProgram, name, blockingWrite, count, offset, pSrc,
            numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnDeviceGlobalVariableRead =
        d_context.urDdiTable.Enqueue.pfnDeviceGlobalVariableRead;
    if (nullptr != pfnDeviceGlobalVariableRead) {
        result = pfnDeviceGlobalVariableRead(
            hQueue, hProgram, name, blockingRead, count, offset, pDst,
            numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnReadHostPipe = d_context.urDdiTable.Enqueue.pfnReadHostPipe;
    if (nullptr != pfnReadHostPipe) {
        result =
            pfnReadHostPipe(hQueue, hProgram, pipe_symbol, blocking, pDst, size,
                            numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnWriteHostPipe = d_context.urDdiTable.Enqueue.pfnWriteHostPipe;
    if (nullptr != pfnWriteHostPipe) {
        result = pfnWriteHostPipe(hQueue, hProgram, pipe_symbol, blocking, pSrc,
                                  size, numEventsInWaitList, phEventWaitList,
                                  phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnPitchedAllocExp = d_context.urDdiTable.USMExp.pfnPitchedAllocExp;
    if (nullptr != pfnPitchedAllocExp) {
        result =
            pfnPitchedAllocExp(hContext, hDevice, pUSMDesc, pool, widthInBytes,
                               height, elementSizeBytes, ppMem, pResultPitch);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesUnsampledImageHandleDestroyExp
__urdlllocal ur_result_t UR_APICALL
urBindlessImagesUnsampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_handle_t
        hImage ///< [in] pointer to handle of image object to destroy
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnUnsampledImageHandleDestroyExp =
        d_context.urDdiTable.BindlessImagesExp
            .pfnUnsampledImageHandleDestroyExp;
    if (nullptr != pfnUnsampledImageHandleDestroyExp) {
        result = pfnUnsampledImageHandleDestroyExp(hContext, hDevice, hImage);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesSampledImageHandleDestroyExp
__urdlllocal ur_result_t UR_APICALL
urBindlessImagesSampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_handle_t
        hImage ///< [in] pointer to handle of image object to destroy
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSampledImageHandleDestroyExp =
        d_context.urDdiTable.BindlessImagesExp.pfnSampledImageHandleDestroyExp;
    if (nullptr != pfnSampledImageHandleDestroyExp) {
        result = pfnSampledImageHandleDestroyExp(hContext, hDevice, hImage);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnImageAllocateExp =
        d_context.urDdiTable.BindlessImagesExp.pfnImageAllocateExp;
    if (nullptr != pfnImageAllocateExp) {
        result = pfnImageAllocateExp(hContext, hDevice, pImageFormat,
                                     pImageDesc, phImageMem);
    } else {
        // generic implementation
        *phImageMem =
            reinterpret_cast<ur_exp_image_mem_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImageFreeExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImageFreeExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_mem_handle_t
        hImageMem ///< [in] handle of image memory to be freed
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnImageFreeExp =
        d_context.urDdiTable.BindlessImagesExp.pfnImageFreeExp;
    if (nullptr != pfnImageFreeExp) {
        result = pfnImageFreeExp(hContext, hDevice, hImageMem);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnUnsampledImageCreateExp =
        d_context.urDdiTable.BindlessImagesExp.pfnUnsampledImageCreateExp;
    if (nullptr != pfnUnsampledImageCreateExp) {
        result = pfnUnsampledImageCreateExp(hContext, hDevice, hImageMem,
                                            pImageFormat, pImageDesc, phMem,
                                            phImage);
    } else {
        // generic implementation
        *phMem = reinterpret_cast<ur_mem_handle_t>(d_context.get());

        *phImage = reinterpret_cast<ur_exp_image_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSampledImageCreateExp =
        d_context.urDdiTable.BindlessImagesExp.pfnSampledImageCreateExp;
    if (nullptr != pfnSampledImageCreateExp) {
        result =
            pfnSampledImageCreateExp(hContext, hDevice, hImageMem, pImageFormat,
                                     pImageDesc, hSampler, phMem, phImage);
    } else {
        // generic implementation
        *phMem = reinterpret_cast<ur_mem_handle_t>(d_context.get());

        *phImage = reinterpret_cast<ur_exp_image_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnImageCopyExp =
        d_context.urDdiTable.BindlessImagesExp.pfnImageCopyExp;
    if (nullptr != pfnImageCopyExp) {
        result = pfnImageCopyExp(hQueue, pDst, pSrc, pImageFormat, pImageDesc,
                                 imageCopyFlags, srcOffset, dstOffset,
                                 copyExtent, hostExtent, numEventsInWaitList,
                                 phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImageGetInfoExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImageGetInfoExp(
    ur_exp_image_mem_handle_t hImageMem, ///< [in] handle to the image memory
    ur_image_info_t propName,            ///< [in] queried info name
    void *pPropValue,    ///< [out][optional] returned query value
    size_t *pPropSizeRet ///< [out][optional] returned query value size
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnImageGetInfoExp =
        d_context.urDdiTable.BindlessImagesExp.pfnImageGetInfoExp;
    if (nullptr != pfnImageGetInfoExp) {
        result =
            pfnImageGetInfoExp(hImageMem, propName, pPropValue, pPropSizeRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMipmapGetLevelExp =
        d_context.urDdiTable.BindlessImagesExp.pfnMipmapGetLevelExp;
    if (nullptr != pfnMipmapGetLevelExp) {
        result = pfnMipmapGetLevelExp(hContext, hDevice, hImageMem, mipmapLevel,
                                      phImageMem);
    } else {
        // generic implementation
        *phImageMem =
            reinterpret_cast<ur_exp_image_mem_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesMipmapFreeExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesMipmapFreeExp(
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    ur_device_handle_t hDevice,    ///< [in] handle of the device object
    ur_exp_image_mem_handle_t hMem ///< [in] handle of image memory to be freed
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMipmapFreeExp =
        d_context.urDdiTable.BindlessImagesExp.pfnMipmapFreeExp;
    if (nullptr != pfnMipmapFreeExp) {
        result = pfnMipmapFreeExp(hContext, hDevice, hMem);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnImportOpaqueFDExp =
        d_context.urDdiTable.BindlessImagesExp.pfnImportOpaqueFDExp;
    if (nullptr != pfnImportOpaqueFDExp) {
        result = pfnImportOpaqueFDExp(hContext, hDevice, size, pInteropMemDesc,
                                      phInteropMem);
    } else {
        // generic implementation
        *phInteropMem =
            reinterpret_cast<ur_exp_interop_mem_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnMapExternalArrayExp =
        d_context.urDdiTable.BindlessImagesExp.pfnMapExternalArrayExp;
    if (nullptr != pfnMapExternalArrayExp) {
        result = pfnMapExternalArrayExp(hContext, hDevice, pImageFormat,
                                        pImageDesc, hInteropMem, phImageMem);
    } else {
        // generic implementation
        *phImageMem =
            reinterpret_cast<ur_exp_image_mem_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesReleaseInteropExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesReleaseInteropExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_interop_mem_handle_t
        hInteropMem ///< [in] handle of interop memory to be freed
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnReleaseInteropExp =
        d_context.urDdiTable.BindlessImagesExp.pfnReleaseInteropExp;
    if (nullptr != pfnReleaseInteropExp) {
        result = pfnReleaseInteropExp(hContext, hDevice, hInteropMem);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnImportExternalSemaphoreOpaqueFDExp =
        d_context.urDdiTable.BindlessImagesExp
            .pfnImportExternalSemaphoreOpaqueFDExp;
    if (nullptr != pfnImportExternalSemaphoreOpaqueFDExp) {
        result = pfnImportExternalSemaphoreOpaqueFDExp(
            hContext, hDevice, pInteropSemaphoreDesc, phInteropSemaphore);
    } else {
        // generic implementation
        *phInteropSemaphore =
            reinterpret_cast<ur_exp_interop_semaphore_handle_t>(
                d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesDestroyExternalSemaphoreExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesDestroyExternalSemaphoreExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_interop_semaphore_handle_t
        hInteropSemaphore ///< [in] handle of interop semaphore to be destroyed
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnDestroyExternalSemaphoreExp =
        d_context.urDdiTable.BindlessImagesExp.pfnDestroyExternalSemaphoreExp;
    if (nullptr != pfnDestroyExternalSemaphoreExp) {
        result = pfnDestroyExternalSemaphoreExp(hContext, hDevice,
                                                hInteropSemaphore);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnWaitExternalSemaphoreExp =
        d_context.urDdiTable.BindlessImagesExp.pfnWaitExternalSemaphoreExp;
    if (nullptr != pfnWaitExternalSemaphoreExp) {
        result = pfnWaitExternalSemaphoreExp(
            hQueue, hSemaphore, numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSignalExternalSemaphoreExp =
        d_context.urDdiTable.BindlessImagesExp.pfnSignalExternalSemaphoreExp;
    if (nullptr != pfnSignalExternalSemaphoreExp) {
        result = pfnSignalExternalSemaphoreExp(
            hQueue, hSemaphore, numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCreateExp = d_context.urDdiTable.CommandBufferExp.pfnCreateExp;
    if (nullptr != pfnCreateExp) {
        result = pfnCreateExp(hContext, hDevice, pCommandBufferDesc,
                              phCommandBuffer);
    } else {
        // generic implementation
        *phCommandBuffer =
            reinterpret_cast<ur_exp_command_buffer_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferRetainExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferRetainExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer ///< [in] Handle of the command-buffer object.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRetainExp = d_context.urDdiTable.CommandBufferExp.pfnRetainExp;
    if (nullptr != pfnRetainExp) {
        result = pfnRetainExp(hCommandBuffer);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferReleaseExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferReleaseExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer ///< [in] Handle of the command-buffer object.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnReleaseExp = d_context.urDdiTable.CommandBufferExp.pfnReleaseExp;
    if (nullptr != pfnReleaseExp) {
        result = pfnReleaseExp(hCommandBuffer);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferFinalizeExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferFinalizeExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer ///< [in] Handle of the command-buffer object.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnFinalizeExp = d_context.urDdiTable.CommandBufferExp.pfnFinalizeExp;
    if (nullptr != pfnFinalizeExp) {
        result = pfnFinalizeExp(hCommandBuffer);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAppendKernelLaunchExp =
        d_context.urDdiTable.CommandBufferExp.pfnAppendKernelLaunchExp;
    if (nullptr != pfnAppendKernelLaunchExp) {
        result = pfnAppendKernelLaunchExp(
            hCommandBuffer, hKernel, workDim, pGlobalWorkOffset,
            pGlobalWorkSize, pLocalWorkSize, numSyncPointsInWaitList,
            pSyncPointWaitList, pSyncPoint, phCommand);
    } else {
        // generic implementation
        if (nullptr != phCommand) {
            *phCommand =
                reinterpret_cast<ur_exp_command_buffer_command_handle_t>(
                    d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAppendUSMMemcpyExp =
        d_context.urDdiTable.CommandBufferExp.pfnAppendUSMMemcpyExp;
    if (nullptr != pfnAppendUSMMemcpyExp) {
        result = pfnAppendUSMMemcpyExp(hCommandBuffer, pDst, pSrc, size,
                                       numSyncPointsInWaitList,
                                       pSyncPointWaitList, pSyncPoint);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAppendUSMFillExp =
        d_context.urDdiTable.CommandBufferExp.pfnAppendUSMFillExp;
    if (nullptr != pfnAppendUSMFillExp) {
        result = pfnAppendUSMFillExp(hCommandBuffer, pMemory, pPattern,
                                     patternSize, size, numSyncPointsInWaitList,
                                     pSyncPointWaitList, pSyncPoint);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAppendMemBufferCopyExp =
        d_context.urDdiTable.CommandBufferExp.pfnAppendMemBufferCopyExp;
    if (nullptr != pfnAppendMemBufferCopyExp) {
        result = pfnAppendMemBufferCopyExp(
            hCommandBuffer, hSrcMem, hDstMem, srcOffset, dstOffset, size,
            numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAppendMemBufferWriteExp =
        d_context.urDdiTable.CommandBufferExp.pfnAppendMemBufferWriteExp;
    if (nullptr != pfnAppendMemBufferWriteExp) {
        result = pfnAppendMemBufferWriteExp(hCommandBuffer, hBuffer, offset,
                                            size, pSrc, numSyncPointsInWaitList,
                                            pSyncPointWaitList, pSyncPoint);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAppendMemBufferReadExp =
        d_context.urDdiTable.CommandBufferExp.pfnAppendMemBufferReadExp;
    if (nullptr != pfnAppendMemBufferReadExp) {
        result = pfnAppendMemBufferReadExp(hCommandBuffer, hBuffer, offset,
                                           size, pDst, numSyncPointsInWaitList,
                                           pSyncPointWaitList, pSyncPoint);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAppendMemBufferCopyRectExp =
        d_context.urDdiTable.CommandBufferExp.pfnAppendMemBufferCopyRectExp;
    if (nullptr != pfnAppendMemBufferCopyRectExp) {
        result = pfnAppendMemBufferCopyRectExp(
            hCommandBuffer, hSrcMem, hDstMem, srcOrigin, dstOrigin, region,
            srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch,
            numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAppendMemBufferWriteRectExp =
        d_context.urDdiTable.CommandBufferExp.pfnAppendMemBufferWriteRectExp;
    if (nullptr != pfnAppendMemBufferWriteRectExp) {
        result = pfnAppendMemBufferWriteRectExp(
            hCommandBuffer, hBuffer, bufferOffset, hostOffset, region,
            bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch,
            pSrc, numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAppendMemBufferReadRectExp =
        d_context.urDdiTable.CommandBufferExp.pfnAppendMemBufferReadRectExp;
    if (nullptr != pfnAppendMemBufferReadRectExp) {
        result = pfnAppendMemBufferReadRectExp(
            hCommandBuffer, hBuffer, bufferOffset, hostOffset, region,
            bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch,
            pDst, numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAppendMemBufferFillExp =
        d_context.urDdiTable.CommandBufferExp.pfnAppendMemBufferFillExp;
    if (nullptr != pfnAppendMemBufferFillExp) {
        result = pfnAppendMemBufferFillExp(
            hCommandBuffer, hBuffer, pPattern, patternSize, offset, size,
            numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAppendUSMPrefetchExp =
        d_context.urDdiTable.CommandBufferExp.pfnAppendUSMPrefetchExp;
    if (nullptr != pfnAppendUSMPrefetchExp) {
        result = pfnAppendUSMPrefetchExp(hCommandBuffer, pMemory, size, flags,
                                         numSyncPointsInWaitList,
                                         pSyncPointWaitList, pSyncPoint);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnAppendUSMAdviseExp =
        d_context.urDdiTable.CommandBufferExp.pfnAppendUSMAdviseExp;
    if (nullptr != pfnAppendUSMAdviseExp) {
        result = pfnAppendUSMAdviseExp(hCommandBuffer, pMemory, size, advice,
                                       numSyncPointsInWaitList,
                                       pSyncPointWaitList, pSyncPoint);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnEnqueueExp = d_context.urDdiTable.CommandBufferExp.pfnEnqueueExp;
    if (nullptr != pfnEnqueueExp) {
        result = pfnEnqueueExp(hCommandBuffer, hQueue, numEventsInWaitList,
                               phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferRetainCommandExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferRetainCommandExp(
    ur_exp_command_buffer_command_handle_t
        hCommand ///< [in] Handle of the command-buffer command.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnRetainCommandExp =
        d_context.urDdiTable.CommandBufferExp.pfnRetainCommandExp;
    if (nullptr != pfnRetainCommandExp) {
        result = pfnRetainCommandExp(hCommand);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferReleaseCommandExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferReleaseCommandExp(
    ur_exp_command_buffer_command_handle_t
        hCommand ///< [in] Handle of the command-buffer command.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnReleaseCommandExp =
        d_context.urDdiTable.CommandBufferExp.pfnReleaseCommandExp;
    if (nullptr != pfnReleaseCommandExp) {
        result = pfnReleaseCommandExp(hCommand);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferUpdateKernelLaunchExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferUpdateKernelLaunchExp(
    ur_exp_command_buffer_command_handle_t
        hCommand, ///< [in] Handle of the command-buffer kernel command to update.
    const ur_exp_command_buffer_update_kernel_launch_desc_t *
        pUpdateKernelLaunch ///< [in] Struct defining how the kernel command is to be updated.
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnUpdateKernelLaunchExp =
        d_context.urDdiTable.CommandBufferExp.pfnUpdateKernelLaunchExp;
    if (nullptr != pfnUpdateKernelLaunchExp) {
        result = pfnUpdateKernelLaunchExp(hCommand, pUpdateKernelLaunch);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnGetInfoExp = d_context.urDdiTable.CommandBufferExp.pfnGetInfoExp;
    if (nullptr != pfnGetInfoExp) {
        result = pfnGetInfoExp(hCommandBuffer, propName, propSize, pPropValue,
                               pPropSizeRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCommandGetInfoExp =
        d_context.urDdiTable.CommandBufferExp.pfnCommandGetInfoExp;
    if (nullptr != pfnCommandGetInfoExp) {
        result = pfnCommandGetInfoExp(hCommand, propName, propSize, pPropValue,
                                      pPropSizeRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCooperativeKernelLaunchExp =
        d_context.urDdiTable.EnqueueExp.pfnCooperativeKernelLaunchExp;
    if (nullptr != pfnCooperativeKernelLaunchExp) {
        result = pfnCooperativeKernelLaunchExp(
            hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
            pLocalWorkSize, numEventsInWaitList, phEventWaitList, phEvent);
    } else {
        // generic implementation
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(d_context.get());
        }
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnSuggestMaxCooperativeGroupCountExp =
        d_context.urDdiTable.KernelExp.pfnSuggestMaxCooperativeGroupCountExp;
    if (nullptr != pfnSuggestMaxCooperativeGroupCountExp) {
        result = pfnSuggestMaxCooperativeGroupCountExp(
            hKernel, localWorkSize, dynamicSharedMemorySize, pGroupCountRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnBuildExp = d_context.urDdiTable.ProgramExp.pfnBuildExp;
    if (nullptr != pfnBuildExp) {
        result = pfnBuildExp(hProgram, numDevices, phDevices, pOptions);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnCompileExp = d_context.urDdiTable.ProgramExp.pfnCompileExp;
    if (nullptr != pfnCompileExp) {
        result = pfnCompileExp(hProgram, numDevices, phDevices, pOptions);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnLinkExp = d_context.urDdiTable.ProgramExp.pfnLinkExp;
    if (nullptr != pfnLinkExp) {
        result = pfnLinkExp(hContext, numDevices, phDevices, count, phPrograms,
                            pOptions, phProgram);
    } else {
        // generic implementation
        *phProgram = reinterpret_cast<ur_program_handle_t>(d_context.get());
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMImportExp
__urdlllocal ur_result_t UR_APICALL urUSMImportExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem,                   ///< [in] pointer to host memory object
    size_t size ///< [in] size in bytes of the host memory object to be imported
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnImportExp = d_context.urDdiTable.USMExp.pfnImportExp;
    if (nullptr != pfnImportExp) {
        result = pfnImportExp(hContext, pMem, size);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMReleaseExp
__urdlllocal ur_result_t UR_APICALL urUSMReleaseExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to host memory object
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnReleaseExp = d_context.urDdiTable.USMExp.pfnReleaseExp;
    if (nullptr != pfnReleaseExp) {
        result = pfnReleaseExp(hContext, pMem);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PEnablePeerAccessExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PEnablePeerAccessExp(
    ur_device_handle_t
        commandDevice,            ///< [in] handle of the command device object
    ur_device_handle_t peerDevice ///< [in] handle of the peer device object
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnEnablePeerAccessExp =
        d_context.urDdiTable.UsmP2PExp.pfnEnablePeerAccessExp;
    if (nullptr != pfnEnablePeerAccessExp) {
        result = pfnEnablePeerAccessExp(commandDevice, peerDevice);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PDisablePeerAccessExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PDisablePeerAccessExp(
    ur_device_handle_t
        commandDevice,            ///< [in] handle of the command device object
    ur_device_handle_t peerDevice ///< [in] handle of the peer device object
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnDisablePeerAccessExp =
        d_context.urDdiTable.UsmP2PExp.pfnDisablePeerAccessExp;
    if (nullptr != pfnDisablePeerAccessExp) {
        result = pfnDisablePeerAccessExp(commandDevice, peerDevice);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    ur_result_t result = UR_RESULT_SUCCESS;

    // if the driver has created a custom function, then call it instead of using the generic path
    auto pfnPeerAccessGetInfoExp =
        d_context.urDdiTable.UsmP2PExp.pfnPeerAccessGetInfoExp;
    if (nullptr != pfnPeerAccessGetInfoExp) {
        result = pfnPeerAccessGetInfoExp(commandDevice, peerDevice, propName,
                                         propSize, pPropValue, pPropSizeRet);
    } else {
        // generic implementation
    }

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

} // namespace driver

#if defined(__cplusplus)
extern "C" {
#endif

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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnAdapterGet = driver::urAdapterGet;

    pDdiTable->pfnAdapterRelease = driver::urAdapterRelease;

    pDdiTable->pfnAdapterRetain = driver::urAdapterRetain;

    pDdiTable->pfnAdapterGetLastError = driver::urAdapterGetLastError;

    pDdiTable->pfnAdapterGetInfo = driver::urAdapterGetInfo;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnUnsampledImageHandleDestroyExp =
        driver::urBindlessImagesUnsampledImageHandleDestroyExp;

    pDdiTable->pfnSampledImageHandleDestroyExp =
        driver::urBindlessImagesSampledImageHandleDestroyExp;

    pDdiTable->pfnImageAllocateExp = driver::urBindlessImagesImageAllocateExp;

    pDdiTable->pfnImageFreeExp = driver::urBindlessImagesImageFreeExp;

    pDdiTable->pfnUnsampledImageCreateExp =
        driver::urBindlessImagesUnsampledImageCreateExp;

    pDdiTable->pfnSampledImageCreateExp =
        driver::urBindlessImagesSampledImageCreateExp;

    pDdiTable->pfnImageCopyExp = driver::urBindlessImagesImageCopyExp;

    pDdiTable->pfnImageGetInfoExp = driver::urBindlessImagesImageGetInfoExp;

    pDdiTable->pfnMipmapGetLevelExp = driver::urBindlessImagesMipmapGetLevelExp;

    pDdiTable->pfnMipmapFreeExp = driver::urBindlessImagesMipmapFreeExp;

    pDdiTable->pfnImportOpaqueFDExp = driver::urBindlessImagesImportOpaqueFDExp;

    pDdiTable->pfnMapExternalArrayExp =
        driver::urBindlessImagesMapExternalArrayExp;

    pDdiTable->pfnReleaseInteropExp = driver::urBindlessImagesReleaseInteropExp;

    pDdiTable->pfnImportExternalSemaphoreOpaqueFDExp =
        driver::urBindlessImagesImportExternalSemaphoreOpaqueFDExp;

    pDdiTable->pfnDestroyExternalSemaphoreExp =
        driver::urBindlessImagesDestroyExternalSemaphoreExp;

    pDdiTable->pfnWaitExternalSemaphoreExp =
        driver::urBindlessImagesWaitExternalSemaphoreExp;

    pDdiTable->pfnSignalExternalSemaphoreExp =
        driver::urBindlessImagesSignalExternalSemaphoreExp;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreateExp = driver::urCommandBufferCreateExp;

    pDdiTable->pfnRetainExp = driver::urCommandBufferRetainExp;

    pDdiTable->pfnReleaseExp = driver::urCommandBufferReleaseExp;

    pDdiTable->pfnFinalizeExp = driver::urCommandBufferFinalizeExp;

    pDdiTable->pfnAppendKernelLaunchExp =
        driver::urCommandBufferAppendKernelLaunchExp;

    pDdiTable->pfnAppendUSMMemcpyExp =
        driver::urCommandBufferAppendUSMMemcpyExp;

    pDdiTable->pfnAppendUSMFillExp = driver::urCommandBufferAppendUSMFillExp;

    pDdiTable->pfnAppendMemBufferCopyExp =
        driver::urCommandBufferAppendMemBufferCopyExp;

    pDdiTable->pfnAppendMemBufferWriteExp =
        driver::urCommandBufferAppendMemBufferWriteExp;

    pDdiTable->pfnAppendMemBufferReadExp =
        driver::urCommandBufferAppendMemBufferReadExp;

    pDdiTable->pfnAppendMemBufferCopyRectExp =
        driver::urCommandBufferAppendMemBufferCopyRectExp;

    pDdiTable->pfnAppendMemBufferWriteRectExp =
        driver::urCommandBufferAppendMemBufferWriteRectExp;

    pDdiTable->pfnAppendMemBufferReadRectExp =
        driver::urCommandBufferAppendMemBufferReadRectExp;

    pDdiTable->pfnAppendMemBufferFillExp =
        driver::urCommandBufferAppendMemBufferFillExp;

    pDdiTable->pfnAppendUSMPrefetchExp =
        driver::urCommandBufferAppendUSMPrefetchExp;

    pDdiTable->pfnAppendUSMAdviseExp =
        driver::urCommandBufferAppendUSMAdviseExp;

    pDdiTable->pfnEnqueueExp = driver::urCommandBufferEnqueueExp;

    pDdiTable->pfnRetainCommandExp = driver::urCommandBufferRetainCommandExp;

    pDdiTable->pfnReleaseCommandExp = driver::urCommandBufferReleaseCommandExp;

    pDdiTable->pfnUpdateKernelLaunchExp =
        driver::urCommandBufferUpdateKernelLaunchExp;

    pDdiTable->pfnGetInfoExp = driver::urCommandBufferGetInfoExp;

    pDdiTable->pfnCommandGetInfoExp = driver::urCommandBufferCommandGetInfoExp;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreate = driver::urContextCreate;

    pDdiTable->pfnRetain = driver::urContextRetain;

    pDdiTable->pfnRelease = driver::urContextRelease;

    pDdiTable->pfnGetInfo = driver::urContextGetInfo;

    pDdiTable->pfnGetNativeHandle = driver::urContextGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle =
        driver::urContextCreateWithNativeHandle;

    pDdiTable->pfnSetExtendedDeleter = driver::urContextSetExtendedDeleter;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnKernelLaunch = driver::urEnqueueKernelLaunch;

    pDdiTable->pfnEventsWait = driver::urEnqueueEventsWait;

    pDdiTable->pfnEventsWaitWithBarrier =
        driver::urEnqueueEventsWaitWithBarrier;

    pDdiTable->pfnMemBufferRead = driver::urEnqueueMemBufferRead;

    pDdiTable->pfnMemBufferWrite = driver::urEnqueueMemBufferWrite;

    pDdiTable->pfnMemBufferReadRect = driver::urEnqueueMemBufferReadRect;

    pDdiTable->pfnMemBufferWriteRect = driver::urEnqueueMemBufferWriteRect;

    pDdiTable->pfnMemBufferCopy = driver::urEnqueueMemBufferCopy;

    pDdiTable->pfnMemBufferCopyRect = driver::urEnqueueMemBufferCopyRect;

    pDdiTable->pfnMemBufferFill = driver::urEnqueueMemBufferFill;

    pDdiTable->pfnMemImageRead = driver::urEnqueueMemImageRead;

    pDdiTable->pfnMemImageWrite = driver::urEnqueueMemImageWrite;

    pDdiTable->pfnMemImageCopy = driver::urEnqueueMemImageCopy;

    pDdiTable->pfnMemBufferMap = driver::urEnqueueMemBufferMap;

    pDdiTable->pfnMemUnmap = driver::urEnqueueMemUnmap;

    pDdiTable->pfnUSMFill = driver::urEnqueueUSMFill;

    pDdiTable->pfnUSMMemcpy = driver::urEnqueueUSMMemcpy;

    pDdiTable->pfnUSMPrefetch = driver::urEnqueueUSMPrefetch;

    pDdiTable->pfnUSMAdvise = driver::urEnqueueUSMAdvise;

    pDdiTable->pfnUSMFill2D = driver::urEnqueueUSMFill2D;

    pDdiTable->pfnUSMMemcpy2D = driver::urEnqueueUSMMemcpy2D;

    pDdiTable->pfnDeviceGlobalVariableWrite =
        driver::urEnqueueDeviceGlobalVariableWrite;

    pDdiTable->pfnDeviceGlobalVariableRead =
        driver::urEnqueueDeviceGlobalVariableRead;

    pDdiTable->pfnReadHostPipe = driver::urEnqueueReadHostPipe;

    pDdiTable->pfnWriteHostPipe = driver::urEnqueueWriteHostPipe;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCooperativeKernelLaunchExp =
        driver::urEnqueueCooperativeKernelLaunchExp;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnGetInfo = driver::urEventGetInfo;

    pDdiTable->pfnGetProfilingInfo = driver::urEventGetProfilingInfo;

    pDdiTable->pfnWait = driver::urEventWait;

    pDdiTable->pfnRetain = driver::urEventRetain;

    pDdiTable->pfnRelease = driver::urEventRelease;

    pDdiTable->pfnGetNativeHandle = driver::urEventGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle =
        driver::urEventCreateWithNativeHandle;

    pDdiTable->pfnSetCallback = driver::urEventSetCallback;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreate = driver::urKernelCreate;

    pDdiTable->pfnGetInfo = driver::urKernelGetInfo;

    pDdiTable->pfnGetGroupInfo = driver::urKernelGetGroupInfo;

    pDdiTable->pfnGetSubGroupInfo = driver::urKernelGetSubGroupInfo;

    pDdiTable->pfnRetain = driver::urKernelRetain;

    pDdiTable->pfnRelease = driver::urKernelRelease;

    pDdiTable->pfnGetNativeHandle = driver::urKernelGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle =
        driver::urKernelCreateWithNativeHandle;

    pDdiTable->pfnSetArgValue = driver::urKernelSetArgValue;

    pDdiTable->pfnSetArgLocal = driver::urKernelSetArgLocal;

    pDdiTable->pfnSetArgPointer = driver::urKernelSetArgPointer;

    pDdiTable->pfnSetExecInfo = driver::urKernelSetExecInfo;

    pDdiTable->pfnSetArgSampler = driver::urKernelSetArgSampler;

    pDdiTable->pfnSetArgMemObj = driver::urKernelSetArgMemObj;

    pDdiTable->pfnSetSpecializationConstants =
        driver::urKernelSetSpecializationConstants;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnSuggestMaxCooperativeGroupCountExp =
        driver::urKernelSuggestMaxCooperativeGroupCountExp;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnImageCreate = driver::urMemImageCreate;

    pDdiTable->pfnBufferCreate = driver::urMemBufferCreate;

    pDdiTable->pfnRetain = driver::urMemRetain;

    pDdiTable->pfnRelease = driver::urMemRelease;

    pDdiTable->pfnBufferPartition = driver::urMemBufferPartition;

    pDdiTable->pfnGetNativeHandle = driver::urMemGetNativeHandle;

    pDdiTable->pfnBufferCreateWithNativeHandle =
        driver::urMemBufferCreateWithNativeHandle;

    pDdiTable->pfnImageCreateWithNativeHandle =
        driver::urMemImageCreateWithNativeHandle;

    pDdiTable->pfnGetInfo = driver::urMemGetInfo;

    pDdiTable->pfnImageGetInfo = driver::urMemImageGetInfo;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreate = driver::urPhysicalMemCreate;

    pDdiTable->pfnRetain = driver::urPhysicalMemRetain;

    pDdiTable->pfnRelease = driver::urPhysicalMemRelease;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnGet = driver::urPlatformGet;

    pDdiTable->pfnGetInfo = driver::urPlatformGetInfo;

    pDdiTable->pfnGetNativeHandle = driver::urPlatformGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle =
        driver::urPlatformCreateWithNativeHandle;

    pDdiTable->pfnGetApiVersion = driver::urPlatformGetApiVersion;

    pDdiTable->pfnGetBackendOption = driver::urPlatformGetBackendOption;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreateWithIL = driver::urProgramCreateWithIL;

    pDdiTable->pfnCreateWithBinary = driver::urProgramCreateWithBinary;

    pDdiTable->pfnBuild = driver::urProgramBuild;

    pDdiTable->pfnCompile = driver::urProgramCompile;

    pDdiTable->pfnLink = driver::urProgramLink;

    pDdiTable->pfnRetain = driver::urProgramRetain;

    pDdiTable->pfnRelease = driver::urProgramRelease;

    pDdiTable->pfnGetFunctionPointer = driver::urProgramGetFunctionPointer;

    pDdiTable->pfnGetInfo = driver::urProgramGetInfo;

    pDdiTable->pfnGetBuildInfo = driver::urProgramGetBuildInfo;

    pDdiTable->pfnSetSpecializationConstants =
        driver::urProgramSetSpecializationConstants;

    pDdiTable->pfnGetNativeHandle = driver::urProgramGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle =
        driver::urProgramCreateWithNativeHandle;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnBuildExp = driver::urProgramBuildExp;

    pDdiTable->pfnCompileExp = driver::urProgramCompileExp;

    pDdiTable->pfnLinkExp = driver::urProgramLinkExp;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnGetInfo = driver::urQueueGetInfo;

    pDdiTable->pfnCreate = driver::urQueueCreate;

    pDdiTable->pfnRetain = driver::urQueueRetain;

    pDdiTable->pfnRelease = driver::urQueueRelease;

    pDdiTable->pfnGetNativeHandle = driver::urQueueGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle =
        driver::urQueueCreateWithNativeHandle;

    pDdiTable->pfnFinish = driver::urQueueFinish;

    pDdiTable->pfnFlush = driver::urQueueFlush;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreate = driver::urSamplerCreate;

    pDdiTable->pfnRetain = driver::urSamplerRetain;

    pDdiTable->pfnRelease = driver::urSamplerRelease;

    pDdiTable->pfnGetInfo = driver::urSamplerGetInfo;

    pDdiTable->pfnGetNativeHandle = driver::urSamplerGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle =
        driver::urSamplerCreateWithNativeHandle;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnHostAlloc = driver::urUSMHostAlloc;

    pDdiTable->pfnDeviceAlloc = driver::urUSMDeviceAlloc;

    pDdiTable->pfnSharedAlloc = driver::urUSMSharedAlloc;

    pDdiTable->pfnFree = driver::urUSMFree;

    pDdiTable->pfnGetMemAllocInfo = driver::urUSMGetMemAllocInfo;

    pDdiTable->pfnPoolCreate = driver::urUSMPoolCreate;

    pDdiTable->pfnPoolRetain = driver::urUSMPoolRetain;

    pDdiTable->pfnPoolRelease = driver::urUSMPoolRelease;

    pDdiTable->pfnPoolGetInfo = driver::urUSMPoolGetInfo;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnPitchedAllocExp = driver::urUSMPitchedAllocExp;

    pDdiTable->pfnImportExp = driver::urUSMImportExp;

    pDdiTable->pfnReleaseExp = driver::urUSMReleaseExp;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnEnablePeerAccessExp = driver::urUsmP2PEnablePeerAccessExp;

    pDdiTable->pfnDisablePeerAccessExp = driver::urUsmP2PDisablePeerAccessExp;

    pDdiTable->pfnPeerAccessGetInfoExp = driver::urUsmP2PPeerAccessGetInfoExp;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnGranularityGetInfo = driver::urVirtualMemGranularityGetInfo;

    pDdiTable->pfnReserve = driver::urVirtualMemReserve;

    pDdiTable->pfnFree = driver::urVirtualMemFree;

    pDdiTable->pfnMap = driver::urVirtualMemMap;

    pDdiTable->pfnUnmap = driver::urVirtualMemUnmap;

    pDdiTable->pfnSetAccess = driver::urVirtualMemSetAccess;

    pDdiTable->pfnGetInfo = driver::urVirtualMemGetInfo;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
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
    ) try {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (driver::d_context.version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnGet = driver::urDeviceGet;

    pDdiTable->pfnGetInfo = driver::urDeviceGetInfo;

    pDdiTable->pfnRetain = driver::urDeviceRetain;

    pDdiTable->pfnRelease = driver::urDeviceRelease;

    pDdiTable->pfnPartition = driver::urDevicePartition;

    pDdiTable->pfnSelectBinary = driver::urDeviceSelectBinary;

    pDdiTable->pfnGetNativeHandle = driver::urDeviceGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle =
        driver::urDeviceCreateWithNativeHandle;

    pDdiTable->pfnGetGlobalTimestamps = driver::urDeviceGetGlobalTimestamps;

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

#if defined(__cplusplus)
}
#endif
