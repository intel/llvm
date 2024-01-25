/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_ldrddi.cpp
 *
 */
#include "ur_lib_loader.hpp"
#include "ur_loader.hpp"

namespace ur_loader {
///////////////////////////////////////////////////////////////////////////////
ur_adapter_factory_t ur_adapter_factory;
ur_platform_factory_t ur_platform_factory;
ur_device_factory_t ur_device_factory;
ur_context_factory_t ur_context_factory;
ur_event_factory_t ur_event_factory;
ur_program_factory_t ur_program_factory;
ur_kernel_factory_t ur_kernel_factory;
ur_queue_factory_t ur_queue_factory;
ur_native_factory_t ur_native_factory;
ur_sampler_factory_t ur_sampler_factory;
ur_mem_factory_t ur_mem_factory;
ur_physical_mem_factory_t ur_physical_mem_factory;
ur_usm_pool_factory_t ur_usm_pool_factory;
ur_exp_image_factory_t ur_exp_image_factory;
ur_exp_image_mem_factory_t ur_exp_image_mem_factory;
ur_exp_interop_mem_factory_t ur_exp_interop_mem_factory;
ur_exp_interop_semaphore_factory_t ur_exp_interop_semaphore_factory;
ur_exp_command_buffer_factory_t ur_exp_command_buffer_factory;

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
    ur_result_t result = UR_RESULT_SUCCESS;

    size_t adapterIndex = 0;
    if (nullptr != phAdapters && NumEntries != 0) {
        for (auto &platform : context->platforms) {
            platform.dditable.ur.Global.pfnAdapterGet(
                1, &phAdapters[adapterIndex], nullptr);
            try {
                phAdapters[adapterIndex] =
                    reinterpret_cast<ur_adapter_handle_t>(
                        ur_adapter_factory.getInstance(phAdapters[adapterIndex],
                                                       &platform.dditable));
            } catch (std::bad_alloc &) {
                result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
                break;
            }
            adapterIndex++;
            if (adapterIndex == NumEntries) {
                break;
            }
        }
    }

    if (pNumAdapters != nullptr) {
        *pNumAdapters = static_cast<uint32_t>(context->platforms.size());
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterRelease
__urdlllocal ur_result_t UR_APICALL urAdapterRelease(
    ur_adapter_handle_t hAdapter ///< [in] Adapter handle to release
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_adapter_object_t *>(hAdapter)->dditable;
    auto pfnAdapterRelease = dditable->ur.Global.pfnAdapterRelease;
    if (nullptr == pfnAdapterRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hAdapter = reinterpret_cast<ur_adapter_object_t *>(hAdapter)->handle;

    // forward to device-platform
    result = pfnAdapterRelease(hAdapter);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterRetain
__urdlllocal ur_result_t UR_APICALL urAdapterRetain(
    ur_adapter_handle_t hAdapter ///< [in] Adapter handle to retain
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_adapter_object_t *>(hAdapter)->dditable;
    auto pfnAdapterRetain = dditable->ur.Global.pfnAdapterRetain;
    if (nullptr == pfnAdapterRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hAdapter = reinterpret_cast<ur_adapter_object_t *>(hAdapter)->handle;

    // forward to device-platform
    result = pfnAdapterRetain(hAdapter);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_adapter_object_t *>(hAdapter)->dditable;
    auto pfnAdapterGetLastError = dditable->ur.Global.pfnAdapterGetLastError;
    if (nullptr == pfnAdapterGetLastError) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hAdapter = reinterpret_cast<ur_adapter_object_t *>(hAdapter)->handle;

    // forward to device-platform
    result = pfnAdapterGetLastError(hAdapter, ppMessage, pError);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_adapter_object_t *>(hAdapter)->dditable;
    auto pfnAdapterGetInfo = dditable->ur.Global.pfnAdapterGetInfo;
    if (nullptr == pfnAdapterGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hAdapter = reinterpret_cast<ur_adapter_object_t *>(hAdapter)->handle;

    // forward to device-platform
    result = pfnAdapterGetInfo(hAdapter, propName, propSize, pPropValue,
                               pPropSizeRet);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    uint32_t total_platform_handle_count = 0;

    for (uint32_t adapter_index = 0; adapter_index < NumAdapters;
         adapter_index++) {
        // extract adapter's function pointer table
        auto dditable =
            reinterpret_cast<ur_platform_object_t *>(phAdapters[adapter_index])
                ->dditable;

        if ((0 < NumEntries) && (NumEntries == total_platform_handle_count)) {
            break;
        }

        uint32_t library_platform_handle_count = 0;

        result = dditable->ur.Platform.pfnGet(&phAdapters[adapter_index], 1, 0,
                                              nullptr,
                                              &library_platform_handle_count);
        if (UR_RESULT_SUCCESS != result) {
            break;
        }

        if (nullptr != phPlatforms && NumEntries != 0) {
            if (total_platform_handle_count + library_platform_handle_count >
                NumEntries) {
                library_platform_handle_count =
                    NumEntries - total_platform_handle_count;
            }
            result = dditable->ur.Platform.pfnGet(
                &phAdapters[adapter_index], 1, library_platform_handle_count,
                &phPlatforms[total_platform_handle_count], nullptr);
            if (UR_RESULT_SUCCESS != result) {
                break;
            }

            try {
                for (uint32_t i = 0; i < library_platform_handle_count; ++i) {
                    uint32_t platform_index = total_platform_handle_count + i;
                    phPlatforms[platform_index] =
                        reinterpret_cast<ur_platform_handle_t>(
                            ur_platform_factory.getInstance(
                                phPlatforms[platform_index], dditable));
                }
            } catch (std::bad_alloc &) {
                result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
            }
        }

        total_platform_handle_count += library_platform_handle_count;
    }

    if (UR_RESULT_SUCCESS == result && pNumPlatforms != nullptr) {
        *pNumPlatforms = total_platform_handle_count;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_platform_object_t *>(hPlatform)->dditable;
    auto pfnGetInfo = dditable->ur.Platform.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hPlatform = reinterpret_cast<ur_platform_object_t *>(hPlatform)->handle;

    // forward to device-platform
    result =
        pfnGetInfo(hPlatform, propName, propSize, pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetApiVersion
__urdlllocal ur_result_t UR_APICALL urPlatformGetApiVersion(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform
    ur_api_version_t *pVersion      ///< [out] api version
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_platform_object_t *>(hPlatform)->dditable;
    auto pfnGetApiVersion = dditable->ur.Platform.pfnGetApiVersion;
    if (nullptr == pfnGetApiVersion) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hPlatform = reinterpret_cast<ur_platform_object_t *>(hPlatform)->handle;

    // forward to device-platform
    result = pfnGetApiVersion(hPlatform, pVersion);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urPlatformGetNativeHandle(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform.
    ur_native_handle_t *
        phNativePlatform ///< [out] a pointer to the native handle of the platform.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_platform_object_t *>(hPlatform)->dditable;
    auto pfnGetNativeHandle = dditable->ur.Platform.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hPlatform = reinterpret_cast<ur_platform_object_t *>(hPlatform)->handle;

    // forward to device-platform
    result = pfnGetNativeHandle(hPlatform, phNativePlatform);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_native_object_t *>(hNativePlatform)->dditable;
    auto pfnCreateWithNativeHandle =
        dditable->ur.Platform.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hNativePlatform =
        reinterpret_cast<ur_native_object_t *>(hNativePlatform)->handle;

    // forward to device-platform
    result =
        pfnCreateWithNativeHandle(hNativePlatform, pProperties, phPlatform);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phPlatform = reinterpret_cast<ur_platform_handle_t>(
            ur_platform_factory.getInstance(*phPlatform, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_platform_object_t *>(hPlatform)->dditable;
    auto pfnGetBackendOption = dditable->ur.Platform.pfnGetBackendOption;
    if (nullptr == pfnGetBackendOption) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hPlatform = reinterpret_cast<ur_platform_object_t *>(hPlatform)->handle;

    // forward to device-platform
    result = pfnGetBackendOption(hPlatform, pFrontendOption, ppPlatformOption);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_platform_object_t *>(hPlatform)->dditable;
    auto pfnGet = dditable->ur.Device.pfnGet;
    if (nullptr == pfnGet) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hPlatform = reinterpret_cast<ur_platform_object_t *>(hPlatform)->handle;

    // forward to device-platform
    result = pfnGet(hPlatform, DeviceType, NumEntries, phDevices, pNumDevices);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handles to loader handles
        for (size_t i = 0; (nullptr != phDevices) && (i < NumEntries); ++i) {
            phDevices[i] = reinterpret_cast<ur_device_handle_t>(
                ur_device_factory.getInstance(phDevices[i], dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_device_object_t *>(hDevice)->dditable;
    auto pfnGetInfo = dditable->ur.Device.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // this value is needed for converting adapter handles to loader handles
    size_t sizeret = 0;
    if (pPropSizeRet == NULL) {
        pPropSizeRet = &sizeret;
    }

    // forward to device-platform
    result = pfnGetInfo(hDevice, propName, propSize, pPropValue, pPropSizeRet);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_DEVICE_INFO_PLATFORM: {
                ur_platform_handle_t *handles =
                    reinterpret_cast<ur_platform_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_platform_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_platform_handle_t>(
                            ur_platform_factory.getInstance(handles[i],
                                                            dditable));
                    }
                }
            } break;
            case UR_DEVICE_INFO_PARENT_DEVICE: {
                ur_device_handle_t *handles =
                    reinterpret_cast<ur_device_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_device_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_device_handle_t>(
                            ur_device_factory.getInstance(handles[i],
                                                          dditable));
                    }
                }
            } break;
            default: {
            } break;
            }
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRetain
__urdlllocal ur_result_t UR_APICALL urDeviceRetain(
    ur_device_handle_t
        hDevice ///< [in] handle of the device to get a reference of.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_device_object_t *>(hDevice)->dditable;
    auto pfnRetain = dditable->ur.Device.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnRetain(hDevice);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRelease
__urdlllocal ur_result_t UR_APICALL urDeviceRelease(
    ur_device_handle_t hDevice ///< [in] handle of the device to release.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_device_object_t *>(hDevice)->dditable;
    auto pfnRelease = dditable->ur.Device.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnRelease(hDevice);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_device_object_t *>(hDevice)->dditable;
    auto pfnPartition = dditable->ur.Device.pfnPartition;
    if (nullptr == pfnPartition) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnPartition(hDevice, pProperties, NumDevices, phSubDevices,
                          pNumDevicesRet);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handles to loader handles
        for (size_t i = 0; (nullptr != phSubDevices) && (i < NumDevices); ++i) {
            phSubDevices[i] = reinterpret_cast<ur_device_handle_t>(
                ur_device_factory.getInstance(phSubDevices[i], dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_device_object_t *>(hDevice)->dditable;
    auto pfnSelectBinary = dditable->ur.Device.pfnSelectBinary;
    if (nullptr == pfnSelectBinary) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnSelectBinary(hDevice, pBinaries, NumBinaries, pSelectedBinary);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urDeviceGetNativeHandle(
    ur_device_handle_t hDevice, ///< [in] handle of the device.
    ur_native_handle_t
        *phNativeDevice ///< [out] a pointer to the native handle of the device.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_device_object_t *>(hDevice)->dditable;
    auto pfnGetNativeHandle = dditable->ur.Device.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnGetNativeHandle(hDevice, phNativeDevice);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_platform_object_t *>(hPlatform)->dditable;
    auto pfnCreateWithNativeHandle =
        dditable->ur.Device.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hPlatform = reinterpret_cast<ur_platform_object_t *>(hPlatform)->handle;

    // forward to device-platform
    result = pfnCreateWithNativeHandle(hNativeDevice, hPlatform, pProperties,
                                       phDevice);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phDevice = reinterpret_cast<ur_device_handle_t>(
            ur_device_factory.getInstance(*phDevice, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_device_object_t *>(hDevice)->dditable;
    auto pfnGetGlobalTimestamps = dditable->ur.Device.pfnGetGlobalTimestamps;
    if (nullptr == pfnGetGlobalTimestamps) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnGetGlobalTimestamps(hDevice, pDeviceTimestamp, pHostTimestamp);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_device_object_t *>(*phDevices)->dditable;
    auto pfnCreate = dditable->ur.Context.pfnCreate;
    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handles to platform handles
    auto phDevicesLocal = std::vector<ur_device_handle_t>(DeviceCount);
    for (size_t i = 0; i < DeviceCount; ++i) {
        phDevicesLocal[i] =
            reinterpret_cast<ur_device_object_t *>(phDevices[i])->handle;
    }

    // forward to device-platform
    result =
        pfnCreate(DeviceCount, phDevicesLocal.data(), pProperties, phContext);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phContext = reinterpret_cast<ur_context_handle_t>(
            ur_context_factory.getInstance(*phContext, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRetain
__urdlllocal ur_result_t UR_APICALL urContextRetain(
    ur_context_handle_t
        hContext ///< [in] handle of the context to get a reference of.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnRetain = dditable->ur.Context.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnRetain(hContext);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRelease
__urdlllocal ur_result_t UR_APICALL urContextRelease(
    ur_context_handle_t hContext ///< [in] handle of the context to release.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnRelease = dditable->ur.Context.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnRelease(hContext);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnGetInfo = dditable->ur.Context.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // this value is needed for converting adapter handles to loader handles
    size_t sizeret = 0;
    if (pPropSizeRet == NULL) {
        pPropSizeRet = &sizeret;
    }

    // forward to device-platform
    result = pfnGetInfo(hContext, propName, propSize, pPropValue, pPropSizeRet);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_CONTEXT_INFO_DEVICES: {
                ur_device_handle_t *handles =
                    reinterpret_cast<ur_device_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_device_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_device_handle_t>(
                            ur_device_factory.getInstance(handles[i],
                                                          dditable));
                    }
                }
            } break;
            default: {
            } break;
            }
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urContextGetNativeHandle(
    ur_context_handle_t hContext, ///< [in] handle of the context.
    ur_native_handle_t *
        phNativeContext ///< [out] a pointer to the native handle of the context.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnGetNativeHandle = dditable->ur.Context.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnGetNativeHandle(hContext, phNativeContext);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_device_object_t *>(*phDevices)->dditable;
    auto pfnCreateWithNativeHandle =
        dditable->ur.Context.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handles to platform handles
    auto phDevicesLocal = std::vector<ur_device_handle_t>(numDevices);
    for (size_t i = 0; i < numDevices; ++i) {
        phDevicesLocal[i] =
            reinterpret_cast<ur_device_object_t *>(phDevices[i])->handle;
    }

    // forward to device-platform
    result = pfnCreateWithNativeHandle(hNativeContext, numDevices,
                                       phDevicesLocal.data(), pProperties,
                                       phContext);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phContext = reinterpret_cast<ur_context_handle_t>(
            ur_context_factory.getInstance(*phContext, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnSetExtendedDeleter = dditable->ur.Context.pfnSetExtendedDeleter;
    if (nullptr == pfnSetExtendedDeleter) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnSetExtendedDeleter(hContext, pfnDeleter, pUserData);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnImageCreate = dditable->ur.Mem.pfnImageCreate;
    if (nullptr == pfnImageCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result =
        pfnImageCreate(hContext, flags, pImageFormat, pImageDesc, pHost, phMem);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phMem = reinterpret_cast<ur_mem_handle_t>(
            ur_mem_factory.getInstance(*phMem, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnBufferCreate = dditable->ur.Mem.pfnBufferCreate;
    if (nullptr == pfnBufferCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnBufferCreate(hContext, flags, size, pProperties, phBuffer);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phBuffer = reinterpret_cast<ur_mem_handle_t>(
            ur_mem_factory.getInstance(*phBuffer, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRetain
__urdlllocal ur_result_t UR_APICALL urMemRetain(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to get access
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_mem_object_t *>(hMem)->dditable;
    auto pfnRetain = dditable->ur.Mem.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hMem = reinterpret_cast<ur_mem_object_t *>(hMem)->handle;

    // forward to device-platform
    result = pfnRetain(hMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRelease
__urdlllocal ur_result_t UR_APICALL urMemRelease(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to release
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_mem_object_t *>(hMem)->dditable;
    auto pfnRelease = dditable->ur.Mem.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hMem = reinterpret_cast<ur_mem_object_t *>(hMem)->handle;

    // forward to device-platform
    result = pfnRelease(hMem);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_mem_object_t *>(hBuffer)->dditable;
    auto pfnBufferPartition = dditable->ur.Mem.pfnBufferPartition;
    if (nullptr == pfnBufferPartition) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hBuffer = reinterpret_cast<ur_mem_object_t *>(hBuffer)->handle;

    // forward to device-platform
    result =
        pfnBufferPartition(hBuffer, flags, bufferCreateType, pRegion, phMem);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phMem = reinterpret_cast<ur_mem_handle_t>(
            ur_mem_factory.getInstance(*phMem, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urMemGetNativeHandle(
    ur_mem_handle_t hMem, ///< [in] handle of the mem.
    ur_native_handle_t
        *phNativeMem ///< [out] a pointer to the native handle of the mem.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_mem_object_t *>(hMem)->dditable;
    auto pfnGetNativeHandle = dditable->ur.Mem.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hMem = reinterpret_cast<ur_mem_object_t *>(hMem)->handle;

    // forward to device-platform
    result = pfnGetNativeHandle(hMem, phNativeMem);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnBufferCreateWithNativeHandle =
        dditable->ur.Mem.pfnBufferCreateWithNativeHandle;
    if (nullptr == pfnBufferCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnBufferCreateWithNativeHandle(hNativeMem, hContext, pProperties,
                                             phMem);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phMem = reinterpret_cast<ur_mem_handle_t>(
            ur_mem_factory.getInstance(*phMem, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnImageCreateWithNativeHandle =
        dditable->ur.Mem.pfnImageCreateWithNativeHandle;
    if (nullptr == pfnImageCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnImageCreateWithNativeHandle(hNativeMem, hContext, pImageFormat,
                                            pImageDesc, pProperties, phMem);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phMem = reinterpret_cast<ur_mem_handle_t>(
            ur_mem_factory.getInstance(*phMem, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_mem_object_t *>(hMemory)->dditable;
    auto pfnGetInfo = dditable->ur.Mem.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hMemory = reinterpret_cast<ur_mem_object_t *>(hMemory)->handle;

    // this value is needed for converting adapter handles to loader handles
    size_t sizeret = 0;
    if (pPropSizeRet == NULL) {
        pPropSizeRet = &sizeret;
    }

    // forward to device-platform
    result = pfnGetInfo(hMemory, propName, propSize, pPropValue, pPropSizeRet);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_MEM_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_context_handle_t>(
                            ur_context_factory.getInstance(handles[i],
                                                           dditable));
                    }
                }
            } break;
            default: {
            } break;
            }
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_mem_object_t *>(hMemory)->dditable;
    auto pfnImageGetInfo = dditable->ur.Mem.pfnImageGetInfo;
    if (nullptr == pfnImageGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hMemory = reinterpret_cast<ur_mem_object_t *>(hMemory)->handle;

    // forward to device-platform
    result =
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnCreate = dditable->ur.Sampler.pfnCreate;
    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnCreate(hContext, pDesc, phSampler);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phSampler = reinterpret_cast<ur_sampler_handle_t>(
            ur_sampler_factory.getInstance(*phSampler, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRetain
__urdlllocal ur_result_t UR_APICALL urSamplerRetain(
    ur_sampler_handle_t
        hSampler ///< [in] handle of the sampler object to get access
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_sampler_object_t *>(hSampler)->dditable;
    auto pfnRetain = dditable->ur.Sampler.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hSampler = reinterpret_cast<ur_sampler_object_t *>(hSampler)->handle;

    // forward to device-platform
    result = pfnRetain(hSampler);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRelease
__urdlllocal ur_result_t UR_APICALL urSamplerRelease(
    ur_sampler_handle_t
        hSampler ///< [in] handle of the sampler object to release
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_sampler_object_t *>(hSampler)->dditable;
    auto pfnRelease = dditable->ur.Sampler.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hSampler = reinterpret_cast<ur_sampler_object_t *>(hSampler)->handle;

    // forward to device-platform
    result = pfnRelease(hSampler);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_sampler_object_t *>(hSampler)->dditable;
    auto pfnGetInfo = dditable->ur.Sampler.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hSampler = reinterpret_cast<ur_sampler_object_t *>(hSampler)->handle;

    // this value is needed for converting adapter handles to loader handles
    size_t sizeret = 0;
    if (pPropSizeRet == NULL) {
        pPropSizeRet = &sizeret;
    }

    // forward to device-platform
    result = pfnGetInfo(hSampler, propName, propSize, pPropValue, pPropSizeRet);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_SAMPLER_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_context_handle_t>(
                            ur_context_factory.getInstance(handles[i],
                                                           dditable));
                    }
                }
            } break;
            default: {
            } break;
            }
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urSamplerGetNativeHandle(
    ur_sampler_handle_t hSampler, ///< [in] handle of the sampler.
    ur_native_handle_t *
        phNativeSampler ///< [out] a pointer to the native handle of the sampler.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_sampler_object_t *>(hSampler)->dditable;
    auto pfnGetNativeHandle = dditable->ur.Sampler.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hSampler = reinterpret_cast<ur_sampler_object_t *>(hSampler)->handle;

    // forward to device-platform
    result = pfnGetNativeHandle(hSampler, phNativeSampler);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnCreateWithNativeHandle =
        dditable->ur.Sampler.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnCreateWithNativeHandle(hNativeSampler, hContext, pProperties,
                                       phSampler);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phSampler = reinterpret_cast<ur_sampler_handle_t>(
            ur_sampler_factory.getInstance(*phSampler, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnHostAlloc = dditable->ur.USM.pfnHostAlloc;
    if (nullptr == pfnHostAlloc) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    pool = (pool) ? reinterpret_cast<ur_usm_pool_object_t *>(pool)->handle
                  : nullptr;

    // forward to device-platform
    result = pfnHostAlloc(hContext, pUSMDesc, pool, size, ppMem);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnDeviceAlloc = dditable->ur.USM.pfnDeviceAlloc;
    if (nullptr == pfnDeviceAlloc) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    pool = (pool) ? reinterpret_cast<ur_usm_pool_object_t *>(pool)->handle
                  : nullptr;

    // forward to device-platform
    result = pfnDeviceAlloc(hContext, hDevice, pUSMDesc, pool, size, ppMem);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnSharedAlloc = dditable->ur.USM.pfnSharedAlloc;
    if (nullptr == pfnSharedAlloc) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    pool = (pool) ? reinterpret_cast<ur_usm_pool_object_t *>(pool)->handle
                  : nullptr;

    // forward to device-platform
    result = pfnSharedAlloc(hContext, hDevice, pUSMDesc, pool, size, ppMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMFree
__urdlllocal ur_result_t UR_APICALL urUSMFree(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to USM memory object
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnFree = dditable->ur.USM.pfnFree;
    if (nullptr == pfnFree) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnFree(hContext, pMem);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnGetMemAllocInfo = dditable->ur.USM.pfnGetMemAllocInfo;
    if (nullptr == pfnGetMemAllocInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // this value is needed for converting adapter handles to loader handles
    size_t sizeret = 0;
    if (pPropSizeRet == NULL) {
        pPropSizeRet = &sizeret;
    }

    // forward to device-platform
    result = pfnGetMemAllocInfo(hContext, pMem, propName, propSize, pPropValue,
                                pPropSizeRet);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_USM_ALLOC_INFO_DEVICE: {
                ur_device_handle_t *handles =
                    reinterpret_cast<ur_device_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_device_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_device_handle_t>(
                            ur_device_factory.getInstance(handles[i],
                                                          dditable));
                    }
                }
            } break;
            case UR_USM_ALLOC_INFO_POOL: {
                ur_usm_pool_handle_t *handles =
                    reinterpret_cast<ur_usm_pool_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_usm_pool_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_usm_pool_handle_t>(
                            ur_usm_pool_factory.getInstance(handles[i],
                                                            dditable));
                    }
                }
            } break;
            default: {
            } break;
            }
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnPoolCreate = dditable->ur.USM.pfnPoolCreate;
    if (nullptr == pfnPoolCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnPoolCreate(hContext, pPoolDesc, ppPool);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *ppPool = reinterpret_cast<ur_usm_pool_handle_t>(
            ur_usm_pool_factory.getInstance(*ppPool, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolRetain
__urdlllocal ur_result_t UR_APICALL urUSMPoolRetain(
    ur_usm_pool_handle_t pPool ///< [in] pointer to USM memory pool
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_usm_pool_object_t *>(pPool)->dditable;
    auto pfnPoolRetain = dditable->ur.USM.pfnPoolRetain;
    if (nullptr == pfnPoolRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    pPool = reinterpret_cast<ur_usm_pool_object_t *>(pPool)->handle;

    // forward to device-platform
    result = pfnPoolRetain(pPool);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolRelease
__urdlllocal ur_result_t UR_APICALL urUSMPoolRelease(
    ur_usm_pool_handle_t pPool ///< [in] pointer to USM memory pool
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_usm_pool_object_t *>(pPool)->dditable;
    auto pfnPoolRelease = dditable->ur.USM.pfnPoolRelease;
    if (nullptr == pfnPoolRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    pPool = reinterpret_cast<ur_usm_pool_object_t *>(pPool)->handle;

    // forward to device-platform
    result = pfnPoolRelease(pPool);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_usm_pool_object_t *>(hPool)->dditable;
    auto pfnPoolGetInfo = dditable->ur.USM.pfnPoolGetInfo;
    if (nullptr == pfnPoolGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hPool = reinterpret_cast<ur_usm_pool_object_t *>(hPool)->handle;

    // this value is needed for converting adapter handles to loader handles
    size_t sizeret = 0;
    if (pPropSizeRet == NULL) {
        pPropSizeRet = &sizeret;
    }

    // forward to device-platform
    result =
        pfnPoolGetInfo(hPool, propName, propSize, pPropValue, pPropSizeRet);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_USM_POOL_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_context_handle_t>(
                            ur_context_factory.getInstance(handles[i],
                                                           dditable));
                    }
                }
            } break;
            default: {
            } break;
            }
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnGranularityGetInfo = dditable->ur.VirtualMem.pfnGranularityGetInfo;
    if (nullptr == pfnGranularityGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = (hDevice)
                  ? reinterpret_cast<ur_device_object_t *>(hDevice)->handle
                  : nullptr;

    // forward to device-platform
    result = pfnGranularityGetInfo(hContext, hDevice, propName, propSize,
                                   pPropValue, pPropSizeRet);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnReserve = dditable->ur.VirtualMem.pfnReserve;
    if (nullptr == pfnReserve) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnReserve(hContext, pStart, size, ppStart);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnFree = dditable->ur.VirtualMem.pfnFree;
    if (nullptr == pfnFree) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnFree(hContext, pStart, size);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnMap = dditable->ur.VirtualMem.pfnMap;
    if (nullptr == pfnMap) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hPhysicalMem =
        reinterpret_cast<ur_physical_mem_object_t *>(hPhysicalMem)->handle;

    // forward to device-platform
    result = pfnMap(hContext, pStart, size, hPhysicalMem, offset, flags);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnUnmap = dditable->ur.VirtualMem.pfnUnmap;
    if (nullptr == pfnUnmap) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnUnmap(hContext, pStart, size);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnSetAccess = dditable->ur.VirtualMem.pfnSetAccess;
    if (nullptr == pfnSetAccess) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnSetAccess(hContext, pStart, size, flags);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnGetInfo = dditable->ur.VirtualMem.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnGetInfo(hContext, pStart, size, propName, propSize, pPropValue,
                        pPropSizeRet);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnCreate = dditable->ur.PhysicalMem.pfnCreate;
    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnCreate(hContext, hDevice, size, pProperties, phPhysicalMem);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phPhysicalMem = reinterpret_cast<ur_physical_mem_handle_t>(
            ur_physical_mem_factory.getInstance(*phPhysicalMem, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemRetain
__urdlllocal ur_result_t UR_APICALL urPhysicalMemRetain(
    ur_physical_mem_handle_t
        hPhysicalMem ///< [in] handle of the physical memory object to retain.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_physical_mem_object_t *>(hPhysicalMem)->dditable;
    auto pfnRetain = dditable->ur.PhysicalMem.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hPhysicalMem =
        reinterpret_cast<ur_physical_mem_object_t *>(hPhysicalMem)->handle;

    // forward to device-platform
    result = pfnRetain(hPhysicalMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemRelease
__urdlllocal ur_result_t UR_APICALL urPhysicalMemRelease(
    ur_physical_mem_handle_t
        hPhysicalMem ///< [in] handle of the physical memory object to release.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_physical_mem_object_t *>(hPhysicalMem)->dditable;
    auto pfnRelease = dditable->ur.PhysicalMem.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hPhysicalMem =
        reinterpret_cast<ur_physical_mem_object_t *>(hPhysicalMem)->handle;

    // forward to device-platform
    result = pfnRelease(hPhysicalMem);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnCreateWithIL = dditable->ur.Program.pfnCreateWithIL;
    if (nullptr == pfnCreateWithIL) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnCreateWithIL(hContext, pIL, length, pProperties, phProgram);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phProgram = reinterpret_cast<ur_program_handle_t>(
            ur_program_factory.getInstance(*phProgram, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnCreateWithBinary = dditable->ur.Program.pfnCreateWithBinary;
    if (nullptr == pfnCreateWithBinary) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnCreateWithBinary(hContext, hDevice, size, pBinary, pProperties,
                                 phProgram);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phProgram = reinterpret_cast<ur_program_handle_t>(
            ur_program_factory.getInstance(*phProgram, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnBuild = dditable->ur.Program.pfnBuild;
    if (nullptr == pfnBuild) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // forward to device-platform
    result = pfnBuild(hContext, hProgram, pOptions);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnCompile = dditable->ur.Program.pfnCompile;
    if (nullptr == pfnCompile) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // forward to device-platform
    result = pfnCompile(hContext, hProgram, pOptions);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnLink = dditable->ur.Program.pfnLink;
    if (nullptr == pfnLink) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handles to platform handles
    auto phProgramsLocal = std::vector<ur_program_handle_t>(count);
    for (size_t i = 0; i < count; ++i) {
        phProgramsLocal[i] =
            reinterpret_cast<ur_program_object_t *>(phPrograms[i])->handle;
    }

    // forward to device-platform
    result =
        pfnLink(hContext, count, phProgramsLocal.data(), pOptions, phProgram);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phProgram = reinterpret_cast<ur_program_handle_t>(
            ur_program_factory.getInstance(*phProgram, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRetain
__urdlllocal ur_result_t UR_APICALL urProgramRetain(
    ur_program_handle_t hProgram ///< [in] handle for the Program to retain
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_program_object_t *>(hProgram)->dditable;
    auto pfnRetain = dditable->ur.Program.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // forward to device-platform
    result = pfnRetain(hProgram);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRelease
__urdlllocal ur_result_t UR_APICALL urProgramRelease(
    ur_program_handle_t hProgram ///< [in] handle for the Program to release
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_program_object_t *>(hProgram)->dditable;
    auto pfnRelease = dditable->ur.Program.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // forward to device-platform
    result = pfnRelease(hProgram);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_device_object_t *>(hDevice)->dditable;
    auto pfnGetFunctionPointer = dditable->ur.Program.pfnGetFunctionPointer;
    if (nullptr == pfnGetFunctionPointer) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // forward to device-platform
    result = pfnGetFunctionPointer(hDevice, hProgram, pFunctionName,
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_program_object_t *>(hProgram)->dditable;
    auto pfnGetInfo = dditable->ur.Program.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // this value is needed for converting adapter handles to loader handles
    size_t sizeret = 0;
    if (pPropSizeRet == NULL) {
        pPropSizeRet = &sizeret;
    }

    // forward to device-platform
    result = pfnGetInfo(hProgram, propName, propSize, pPropValue, pPropSizeRet);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_PROGRAM_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_context_handle_t>(
                            ur_context_factory.getInstance(handles[i],
                                                           dditable));
                    }
                }
            } break;
            case UR_PROGRAM_INFO_DEVICES: {
                ur_device_handle_t *handles =
                    reinterpret_cast<ur_device_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_device_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_device_handle_t>(
                            ur_device_factory.getInstance(handles[i],
                                                          dditable));
                    }
                }
            } break;
            default: {
            } break;
            }
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_program_object_t *>(hProgram)->dditable;
    auto pfnGetBuildInfo = dditable->ur.Program.pfnGetBuildInfo;
    if (nullptr == pfnGetBuildInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnGetBuildInfo(hProgram, hDevice, propName, propSize, pPropValue,
                             pPropSizeRet);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_program_object_t *>(hProgram)->dditable;
    auto pfnSetSpecializationConstants =
        dditable->ur.Program.pfnSetSpecializationConstants;
    if (nullptr == pfnSetSpecializationConstants) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // forward to device-platform
    result = pfnSetSpecializationConstants(hProgram, count, pSpecConstants);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t hProgram, ///< [in] handle of the program.
    ur_native_handle_t *
        phNativeProgram ///< [out] a pointer to the native handle of the program.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_program_object_t *>(hProgram)->dditable;
    auto pfnGetNativeHandle = dditable->ur.Program.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // forward to device-platform
    result = pfnGetNativeHandle(hProgram, phNativeProgram);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnCreateWithNativeHandle =
        dditable->ur.Program.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnCreateWithNativeHandle(hNativeProgram, hContext, pProperties,
                                       phProgram);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phProgram = reinterpret_cast<ur_program_handle_t>(
            ur_program_factory.getInstance(*phProgram, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_program_object_t *>(hProgram)->dditable;
    auto pfnCreate = dditable->ur.Kernel.pfnCreate;
    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // forward to device-platform
    result = pfnCreate(hProgram, pKernelName, phKernel);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phKernel = reinterpret_cast<ur_kernel_handle_t>(
            ur_kernel_factory.getInstance(*phKernel, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnSetArgValue = dditable->ur.Kernel.pfnSetArgValue;
    if (nullptr == pfnSetArgValue) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // forward to device-platform
    result = pfnSetArgValue(hKernel, argIndex, argSize, pProperties, pArgValue);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnSetArgLocal = dditable->ur.Kernel.pfnSetArgLocal;
    if (nullptr == pfnSetArgLocal) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // forward to device-platform
    result = pfnSetArgLocal(hKernel, argIndex, argSize, pProperties);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnGetInfo = dditable->ur.Kernel.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // this value is needed for converting adapter handles to loader handles
    size_t sizeret = 0;
    if (pPropSizeRet == NULL) {
        pPropSizeRet = &sizeret;
    }

    // forward to device-platform
    result = pfnGetInfo(hKernel, propName, propSize, pPropValue, pPropSizeRet);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_KERNEL_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_context_handle_t>(
                            ur_context_factory.getInstance(handles[i],
                                                           dditable));
                    }
                }
            } break;
            case UR_KERNEL_INFO_PROGRAM: {
                ur_program_handle_t *handles =
                    reinterpret_cast<ur_program_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_program_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_program_handle_t>(
                            ur_program_factory.getInstance(handles[i],
                                                           dditable));
                    }
                }
            } break;
            default: {
            } break;
            }
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnGetGroupInfo = dditable->ur.Kernel.pfnGetGroupInfo;
    if (nullptr == pfnGetGroupInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnGetGroupInfo(hKernel, hDevice, propName, propSize, pPropValue,
                             pPropSizeRet);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnGetSubGroupInfo = dditable->ur.Kernel.pfnGetSubGroupInfo;
    if (nullptr == pfnGetSubGroupInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnGetSubGroupInfo(hKernel, hDevice, propName, propSize,
                                pPropValue, pPropSizeRet);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRetain
__urdlllocal ur_result_t UR_APICALL urKernelRetain(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to retain
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnRetain = dditable->ur.Kernel.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // forward to device-platform
    result = pfnRetain(hKernel);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRelease
__urdlllocal ur_result_t UR_APICALL urKernelRelease(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to release
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnRelease = dditable->ur.Kernel.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // forward to device-platform
    result = pfnRelease(hKernel);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnSetArgPointer = dditable->ur.Kernel.pfnSetArgPointer;
    if (nullptr == pfnSetArgPointer) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // forward to device-platform
    result = pfnSetArgPointer(hKernel, argIndex, pProperties, pArgValue);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnSetExecInfo = dditable->ur.Kernel.pfnSetExecInfo;
    if (nullptr == pfnSetExecInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // forward to device-platform
    result =
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnSetArgSampler = dditable->ur.Kernel.pfnSetArgSampler;
    if (nullptr == pfnSetArgSampler) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // convert loader handle to platform handle
    hArgValue = reinterpret_cast<ur_sampler_object_t *>(hArgValue)->handle;

    // forward to device-platform
    result = pfnSetArgSampler(hKernel, argIndex, pProperties, hArgValue);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnSetArgMemObj = dditable->ur.Kernel.pfnSetArgMemObj;
    if (nullptr == pfnSetArgMemObj) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // convert loader handle to platform handle
    hArgValue = (hArgValue)
                    ? reinterpret_cast<ur_mem_object_t *>(hArgValue)->handle
                    : nullptr;

    // forward to device-platform
    result = pfnSetArgMemObj(hKernel, argIndex, pProperties, hArgValue);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnSetSpecializationConstants =
        dditable->ur.Kernel.pfnSetSpecializationConstants;
    if (nullptr == pfnSetSpecializationConstants) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // forward to device-platform
    result = pfnSetSpecializationConstants(hKernel, count, pSpecConstants);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urKernelGetNativeHandle(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel.
    ur_native_handle_t
        *phNativeKernel ///< [out] a pointer to the native handle of the kernel.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnGetNativeHandle = dditable->ur.Kernel.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // forward to device-platform
    result = pfnGetNativeHandle(hKernel, phNativeKernel);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnCreateWithNativeHandle =
        dditable->ur.Kernel.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // forward to device-platform
    result = pfnCreateWithNativeHandle(hNativeKernel, hContext, hProgram,
                                       pProperties, phKernel);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phKernel = reinterpret_cast<ur_kernel_handle_t>(
            ur_kernel_factory.getInstance(*phKernel, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnGetInfo = dditable->ur.Queue.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // this value is needed for converting adapter handles to loader handles
    size_t sizeret = 0;
    if (pPropSizeRet == NULL) {
        pPropSizeRet = &sizeret;
    }

    // forward to device-platform
    result = pfnGetInfo(hQueue, propName, propSize, pPropValue, pPropSizeRet);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_QUEUE_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_context_handle_t>(
                            ur_context_factory.getInstance(handles[i],
                                                           dditable));
                    }
                }
            } break;
            case UR_QUEUE_INFO_DEVICE: {
                ur_device_handle_t *handles =
                    reinterpret_cast<ur_device_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_device_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_device_handle_t>(
                            ur_device_factory.getInstance(handles[i],
                                                          dditable));
                    }
                }
            } break;
            case UR_QUEUE_INFO_DEVICE_DEFAULT: {
                ur_queue_handle_t *handles =
                    reinterpret_cast<ur_queue_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_queue_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_queue_handle_t>(
                            ur_queue_factory.getInstance(handles[i], dditable));
                    }
                }
            } break;
            default: {
            } break;
            }
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnCreate = dditable->ur.Queue.pfnCreate;
    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnCreate(hContext, hDevice, pProperties, phQueue);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phQueue = reinterpret_cast<ur_queue_handle_t>(
            ur_queue_factory.getInstance(*phQueue, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRetain
__urdlllocal ur_result_t UR_APICALL urQueueRetain(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to get access
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnRetain = dditable->ur.Queue.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // forward to device-platform
    result = pfnRetain(hQueue);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRelease
__urdlllocal ur_result_t UR_APICALL urQueueRelease(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to release
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnRelease = dditable->ur.Queue.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // forward to device-platform
    result = pfnRelease(hQueue);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnGetNativeHandle = dditable->ur.Queue.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // forward to device-platform
    result = pfnGetNativeHandle(hQueue, pDesc, phNativeQueue);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnCreateWithNativeHandle =
        dditable->ur.Queue.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnCreateWithNativeHandle(hNativeQueue, hContext, hDevice,
                                       pProperties, phQueue);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phQueue = reinterpret_cast<ur_queue_handle_t>(
            ur_queue_factory.getInstance(*phQueue, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFinish
__urdlllocal ur_result_t UR_APICALL urQueueFinish(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be finished.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnFinish = dditable->ur.Queue.pfnFinish;
    if (nullptr == pfnFinish) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // forward to device-platform
    result = pfnFinish(hQueue);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFlush
__urdlllocal ur_result_t UR_APICALL urQueueFlush(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be flushed.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnFlush = dditable->ur.Queue.pfnFlush;
    if (nullptr == pfnFlush) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // forward to device-platform
    result = pfnFlush(hQueue);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_event_object_t *>(hEvent)->dditable;
    auto pfnGetInfo = dditable->ur.Event.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hEvent = reinterpret_cast<ur_event_object_t *>(hEvent)->handle;

    // this value is needed for converting adapter handles to loader handles
    size_t sizeret = 0;
    if (pPropSizeRet == NULL) {
        pPropSizeRet = &sizeret;
    }

    // forward to device-platform
    result = pfnGetInfo(hEvent, propName, propSize, pPropValue, pPropSizeRet);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        if (pPropValue != nullptr) {
            switch (propName) {
            case UR_EVENT_INFO_COMMAND_QUEUE: {
                ur_queue_handle_t *handles =
                    reinterpret_cast<ur_queue_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_queue_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_queue_handle_t>(
                            ur_queue_factory.getInstance(handles[i], dditable));
                    }
                }
            } break;
            case UR_EVENT_INFO_CONTEXT: {
                ur_context_handle_t *handles =
                    reinterpret_cast<ur_context_handle_t *>(pPropValue);
                size_t nelements = *pPropSizeRet / sizeof(ur_context_handle_t);
                for (size_t i = 0; i < nelements; ++i) {
                    if (handles[i] != nullptr) {
                        handles[i] = reinterpret_cast<ur_context_handle_t>(
                            ur_context_factory.getInstance(handles[i],
                                                           dditable));
                    }
                }
            } break;
            default: {
            } break;
            }
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_event_object_t *>(hEvent)->dditable;
    auto pfnGetProfilingInfo = dditable->ur.Event.pfnGetProfilingInfo;
    if (nullptr == pfnGetProfilingInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hEvent = reinterpret_cast<ur_event_object_t *>(hEvent)->handle;

    // forward to device-platform
    result = pfnGetProfilingInfo(hEvent, propName, propSize, pPropValue,
                                 pPropSizeRet);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_event_object_t *>(*phEventWaitList)->dditable;
    auto pfnWait = dditable->ur.Event.pfnWait;
    if (nullptr == pfnWait) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handles to platform handles
    auto phEventWaitListLocal = std::vector<ur_event_handle_t>(numEvents);
    for (size_t i = 0; i < numEvents; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnWait(numEvents, phEventWaitListLocal.data());

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRetain
__urdlllocal ur_result_t UR_APICALL urEventRetain(
    ur_event_handle_t hEvent ///< [in] handle of the event object
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_event_object_t *>(hEvent)->dditable;
    auto pfnRetain = dditable->ur.Event.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hEvent = reinterpret_cast<ur_event_object_t *>(hEvent)->handle;

    // forward to device-platform
    result = pfnRetain(hEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRelease
__urdlllocal ur_result_t UR_APICALL urEventRelease(
    ur_event_handle_t hEvent ///< [in] handle of the event object
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_event_object_t *>(hEvent)->dditable;
    auto pfnRelease = dditable->ur.Event.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hEvent = reinterpret_cast<ur_event_object_t *>(hEvent)->handle;

    // forward to device-platform
    result = pfnRelease(hEvent);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urEventGetNativeHandle(
    ur_event_handle_t hEvent, ///< [in] handle of the event.
    ur_native_handle_t
        *phNativeEvent ///< [out] a pointer to the native handle of the event.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_event_object_t *>(hEvent)->dditable;
    auto pfnGetNativeHandle = dditable->ur.Event.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hEvent = reinterpret_cast<ur_event_object_t *>(hEvent)->handle;

    // forward to device-platform
    result = pfnGetNativeHandle(hEvent, phNativeEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnCreateWithNativeHandle =
        dditable->ur.Event.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result =
        pfnCreateWithNativeHandle(hNativeEvent, hContext, pProperties, phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phEvent = reinterpret_cast<ur_event_handle_t>(
            ur_event_factory.getInstance(*phEvent, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_event_object_t *>(hEvent)->dditable;
    auto pfnSetCallback = dditable->ur.Event.pfnSetCallback;
    if (nullptr == pfnSetCallback) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hEvent = reinterpret_cast<ur_event_object_t *>(hEvent)->handle;

    // forward to device-platform
    result = pfnSetCallback(hEvent, execStatus, pfnNotify, pUserData);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnKernelLaunch = dditable->ur.Enqueue.pfnKernelLaunch;
    if (nullptr == pfnKernelLaunch) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result =
        pfnKernelLaunch(hQueue, hKernel, workDim, pGlobalWorkOffset,
                        pGlobalWorkSize, pLocalWorkSize, numEventsInWaitList,
                        phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnEventsWait = dditable->ur.Enqueue.pfnEventsWait;
    if (nullptr == pfnEventsWait) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnEventsWait(hQueue, numEventsInWaitList,
                           phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnEventsWaitWithBarrier =
        dditable->ur.Enqueue.pfnEventsWaitWithBarrier;
    if (nullptr == pfnEventsWaitWithBarrier) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                      phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnMemBufferRead = dditable->ur.Enqueue.pfnMemBufferRead;
    if (nullptr == pfnMemBufferRead) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hBuffer = reinterpret_cast<ur_mem_object_t *>(hBuffer)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnMemBufferRead(hQueue, hBuffer, blockingRead, offset, size, pDst,
                              numEventsInWaitList, phEventWaitListLocal.data(),
                              phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnMemBufferWrite = dditable->ur.Enqueue.pfnMemBufferWrite;
    if (nullptr == pfnMemBufferWrite) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hBuffer = reinterpret_cast<ur_mem_object_t *>(hBuffer)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnMemBufferWrite(hQueue, hBuffer, blockingWrite, offset, size,
                               pSrc, numEventsInWaitList,
                               phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnMemBufferReadRect = dditable->ur.Enqueue.pfnMemBufferReadRect;
    if (nullptr == pfnMemBufferReadRect) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hBuffer = reinterpret_cast<ur_mem_object_t *>(hBuffer)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnMemBufferReadRect(
        hQueue, hBuffer, blockingRead, bufferOrigin, hostOrigin, region,
        bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst,
        numEventsInWaitList, phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnMemBufferWriteRect = dditable->ur.Enqueue.pfnMemBufferWriteRect;
    if (nullptr == pfnMemBufferWriteRect) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hBuffer = reinterpret_cast<ur_mem_object_t *>(hBuffer)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnMemBufferWriteRect(
        hQueue, hBuffer, blockingWrite, bufferOrigin, hostOrigin, region,
        bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc,
        numEventsInWaitList, phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnMemBufferCopy = dditable->ur.Enqueue.pfnMemBufferCopy;
    if (nullptr == pfnMemBufferCopy) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hBufferSrc = reinterpret_cast<ur_mem_object_t *>(hBufferSrc)->handle;

    // convert loader handle to platform handle
    hBufferDst = reinterpret_cast<ur_mem_object_t *>(hBufferDst)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnMemBufferCopy(hQueue, hBufferSrc, hBufferDst, srcOffset,
                              dstOffset, size, numEventsInWaitList,
                              phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnMemBufferCopyRect = dditable->ur.Enqueue.pfnMemBufferCopyRect;
    if (nullptr == pfnMemBufferCopyRect) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hBufferSrc = reinterpret_cast<ur_mem_object_t *>(hBufferSrc)->handle;

    // convert loader handle to platform handle
    hBufferDst = reinterpret_cast<ur_mem_object_t *>(hBufferDst)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnMemBufferCopyRect(
        hQueue, hBufferSrc, hBufferDst, srcOrigin, dstOrigin, region,
        srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch,
        numEventsInWaitList, phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnMemBufferFill = dditable->ur.Enqueue.pfnMemBufferFill;
    if (nullptr == pfnMemBufferFill) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hBuffer = reinterpret_cast<ur_mem_object_t *>(hBuffer)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnMemBufferFill(hQueue, hBuffer, pPattern, patternSize, offset,
                              size, numEventsInWaitList,
                              phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnMemImageRead = dditable->ur.Enqueue.pfnMemImageRead;
    if (nullptr == pfnMemImageRead) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hImage = reinterpret_cast<ur_mem_object_t *>(hImage)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnMemImageRead(hQueue, hImage, blockingRead, origin, region,
                             rowPitch, slicePitch, pDst, numEventsInWaitList,
                             phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnMemImageWrite = dditable->ur.Enqueue.pfnMemImageWrite;
    if (nullptr == pfnMemImageWrite) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hImage = reinterpret_cast<ur_mem_object_t *>(hImage)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnMemImageWrite(hQueue, hImage, blockingWrite, origin, region,
                              rowPitch, slicePitch, pSrc, numEventsInWaitList,
                              phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnMemImageCopy = dditable->ur.Enqueue.pfnMemImageCopy;
    if (nullptr == pfnMemImageCopy) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hImageSrc = reinterpret_cast<ur_mem_object_t *>(hImageSrc)->handle;

    // convert loader handle to platform handle
    hImageDst = reinterpret_cast<ur_mem_object_t *>(hImageDst)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnMemImageCopy(hQueue, hImageSrc, hImageDst, srcOrigin, dstOrigin,
                             region, numEventsInWaitList,
                             phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnMemBufferMap = dditable->ur.Enqueue.pfnMemBufferMap;
    if (nullptr == pfnMemBufferMap) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hBuffer = reinterpret_cast<ur_mem_object_t *>(hBuffer)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnMemBufferMap(hQueue, hBuffer, blockingMap, mapFlags, offset,
                             size, numEventsInWaitList,
                             phEventWaitListLocal.data(), phEvent, ppRetMap);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnMemUnmap = dditable->ur.Enqueue.pfnMemUnmap;
    if (nullptr == pfnMemUnmap) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hMem = reinterpret_cast<ur_mem_object_t *>(hMem)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnMemUnmap(hQueue, hMem, pMappedPtr, numEventsInWaitList,
                         phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnUSMFill = dditable->ur.Enqueue.pfnUSMFill;
    if (nullptr == pfnUSMFill) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result =
        pfnUSMFill(hQueue, pMem, patternSize, pPattern, size,
                   numEventsInWaitList, phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnUSMMemcpy = dditable->ur.Enqueue.pfnUSMMemcpy;
    if (nullptr == pfnUSMMemcpy) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result =
        pfnUSMMemcpy(hQueue, blocking, pDst, pSrc, size, numEventsInWaitList,
                     phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnUSMPrefetch = dditable->ur.Enqueue.pfnUSMPrefetch;
    if (nullptr == pfnUSMPrefetch) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnUSMPrefetch(hQueue, pMem, size, flags, numEventsInWaitList,
                            phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnUSMAdvise = dditable->ur.Enqueue.pfnUSMAdvise;
    if (nullptr == pfnUSMAdvise) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // forward to device-platform
    result = pfnUSMAdvise(hQueue, pMem, size, advice, phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnUSMFill2D = dditable->ur.Enqueue.pfnUSMFill2D;
    if (nullptr == pfnUSMFill2D) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result =
        pfnUSMFill2D(hQueue, pMem, pitch, patternSize, pPattern, width, height,
                     numEventsInWaitList, phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnUSMMemcpy2D = dditable->ur.Enqueue.pfnUSMMemcpy2D;
    if (nullptr == pfnUSMMemcpy2D) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnUSMMemcpy2D(hQueue, blocking, pDst, dstPitch, pSrc, srcPitch,
                            width, height, numEventsInWaitList,
                            phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnDeviceGlobalVariableWrite =
        dditable->ur.Enqueue.pfnDeviceGlobalVariableWrite;
    if (nullptr == pfnDeviceGlobalVariableWrite) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnDeviceGlobalVariableWrite(
        hQueue, hProgram, name, blockingWrite, count, offset, pSrc,
        numEventsInWaitList, phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnDeviceGlobalVariableRead =
        dditable->ur.Enqueue.pfnDeviceGlobalVariableRead;
    if (nullptr == pfnDeviceGlobalVariableRead) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnDeviceGlobalVariableRead(
        hQueue, hProgram, name, blockingRead, count, offset, pDst,
        numEventsInWaitList, phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnReadHostPipe = dditable->ur.Enqueue.pfnReadHostPipe;
    if (nullptr == pfnReadHostPipe) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnReadHostPipe(hQueue, hProgram, pipe_symbol, blocking, pDst,
                             size, numEventsInWaitList,
                             phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnWriteHostPipe = dditable->ur.Enqueue.pfnWriteHostPipe;
    if (nullptr == pfnWriteHostPipe) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnWriteHostPipe(hQueue, hProgram, pipe_symbol, blocking, pSrc,
                              size, numEventsInWaitList,
                              phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnPitchedAllocExp = dditable->ur.USMExp.pfnPitchedAllocExp;
    if (nullptr == pfnPitchedAllocExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    pool = (pool) ? reinterpret_cast<ur_usm_pool_object_t *>(pool)->handle
                  : nullptr;

    // forward to device-platform
    result = pfnPitchedAllocExp(hContext, hDevice, pUSMDesc, pool, widthInBytes,
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnUnsampledImageHandleDestroyExp =
        dditable->ur.BindlessImagesExp.pfnUnsampledImageHandleDestroyExp;
    if (nullptr == pfnUnsampledImageHandleDestroyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    hImage = reinterpret_cast<ur_exp_image_object_t *>(hImage)->handle;

    // forward to device-platform
    result = pfnUnsampledImageHandleDestroyExp(hContext, hDevice, hImage);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnSampledImageHandleDestroyExp =
        dditable->ur.BindlessImagesExp.pfnSampledImageHandleDestroyExp;
    if (nullptr == pfnSampledImageHandleDestroyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    hImage = reinterpret_cast<ur_exp_image_object_t *>(hImage)->handle;

    // forward to device-platform
    result = pfnSampledImageHandleDestroyExp(hContext, hDevice, hImage);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnImageAllocateExp =
        dditable->ur.BindlessImagesExp.pfnImageAllocateExp;
    if (nullptr == pfnImageAllocateExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnImageAllocateExp(hContext, hDevice, pImageFormat, pImageDesc,
                                 phImageMem);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phImageMem = reinterpret_cast<ur_exp_image_mem_handle_t>(
            ur_exp_image_mem_factory.getInstance(*phImageMem, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnImageFreeExp = dditable->ur.BindlessImagesExp.pfnImageFreeExp;
    if (nullptr == pfnImageFreeExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    hImageMem =
        reinterpret_cast<ur_exp_image_mem_object_t *>(hImageMem)->handle;

    // forward to device-platform
    result = pfnImageFreeExp(hContext, hDevice, hImageMem);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnUnsampledImageCreateExp =
        dditable->ur.BindlessImagesExp.pfnUnsampledImageCreateExp;
    if (nullptr == pfnUnsampledImageCreateExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    hImageMem =
        reinterpret_cast<ur_exp_image_mem_object_t *>(hImageMem)->handle;

    // forward to device-platform
    result = pfnUnsampledImageCreateExp(
        hContext, hDevice, hImageMem, pImageFormat, pImageDesc, phMem, phImage);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phMem = reinterpret_cast<ur_mem_handle_t>(
            ur_mem_factory.getInstance(*phMem, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    try {
        // convert platform handle to loader handle
        *phImage = reinterpret_cast<ur_exp_image_handle_t>(
            ur_exp_image_factory.getInstance(*phImage, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnSampledImageCreateExp =
        dditable->ur.BindlessImagesExp.pfnSampledImageCreateExp;
    if (nullptr == pfnSampledImageCreateExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    hImageMem =
        reinterpret_cast<ur_exp_image_mem_object_t *>(hImageMem)->handle;

    // convert loader handle to platform handle
    hSampler = reinterpret_cast<ur_sampler_object_t *>(hSampler)->handle;

    // forward to device-platform
    result =
        pfnSampledImageCreateExp(hContext, hDevice, hImageMem, pImageFormat,
                                 pImageDesc, hSampler, phMem, phImage);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phMem = reinterpret_cast<ur_mem_handle_t>(
            ur_mem_factory.getInstance(*phMem, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    try {
        // convert platform handle to loader handle
        *phImage = reinterpret_cast<ur_exp_image_handle_t>(
            ur_exp_image_factory.getInstance(*phImage, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnImageCopyExp = dditable->ur.BindlessImagesExp.pfnImageCopyExp;
    if (nullptr == pfnImageCopyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnImageCopyExp(hQueue, pDst, pSrc, pImageFormat, pImageDesc,
                             imageCopyFlags, srcOffset, dstOffset, copyExtent,
                             hostExtent, numEventsInWaitList,
                             phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_image_mem_object_t *>(hImageMem)->dditable;
    auto pfnImageGetInfoExp = dditable->ur.BindlessImagesExp.pfnImageGetInfoExp;
    if (nullptr == pfnImageGetInfoExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hImageMem =
        reinterpret_cast<ur_exp_image_mem_object_t *>(hImageMem)->handle;

    // forward to device-platform
    result = pfnImageGetInfoExp(hImageMem, propName, pPropValue, pPropSizeRet);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnMipmapGetLevelExp =
        dditable->ur.BindlessImagesExp.pfnMipmapGetLevelExp;
    if (nullptr == pfnMipmapGetLevelExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    hImageMem =
        reinterpret_cast<ur_exp_image_mem_object_t *>(hImageMem)->handle;

    // forward to device-platform
    result = pfnMipmapGetLevelExp(hContext, hDevice, hImageMem, mipmapLevel,
                                  phImageMem);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phImageMem = reinterpret_cast<ur_exp_image_mem_handle_t>(
            ur_exp_image_mem_factory.getInstance(*phImageMem, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesMipmapFreeExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesMipmapFreeExp(
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    ur_device_handle_t hDevice,    ///< [in] handle of the device object
    ur_exp_image_mem_handle_t hMem ///< [in] handle of image memory to be freed
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnMipmapFreeExp = dditable->ur.BindlessImagesExp.pfnMipmapFreeExp;
    if (nullptr == pfnMipmapFreeExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    hMem = reinterpret_cast<ur_exp_image_mem_object_t *>(hMem)->handle;

    // forward to device-platform
    result = pfnMipmapFreeExp(hContext, hDevice, hMem);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnImportOpaqueFDExp =
        dditable->ur.BindlessImagesExp.pfnImportOpaqueFDExp;
    if (nullptr == pfnImportOpaqueFDExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnImportOpaqueFDExp(hContext, hDevice, size, pInteropMemDesc,
                                  phInteropMem);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phInteropMem = reinterpret_cast<ur_exp_interop_mem_handle_t>(
            ur_exp_interop_mem_factory.getInstance(*phInteropMem, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnMapExternalArrayExp =
        dditable->ur.BindlessImagesExp.pfnMapExternalArrayExp;
    if (nullptr == pfnMapExternalArrayExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    hInteropMem =
        reinterpret_cast<ur_exp_interop_mem_object_t *>(hInteropMem)->handle;

    // forward to device-platform
    result = pfnMapExternalArrayExp(hContext, hDevice, pImageFormat, pImageDesc,
                                    hInteropMem, phImageMem);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phImageMem = reinterpret_cast<ur_exp_image_mem_handle_t>(
            ur_exp_image_mem_factory.getInstance(*phImageMem, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnReleaseInteropExp =
        dditable->ur.BindlessImagesExp.pfnReleaseInteropExp;
    if (nullptr == pfnReleaseInteropExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    hInteropMem =
        reinterpret_cast<ur_exp_interop_mem_object_t *>(hInteropMem)->handle;

    // forward to device-platform
    result = pfnReleaseInteropExp(hContext, hDevice, hInteropMem);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnImportExternalSemaphoreOpaqueFDExp =
        dditable->ur.BindlessImagesExp.pfnImportExternalSemaphoreOpaqueFDExp;
    if (nullptr == pfnImportExternalSemaphoreOpaqueFDExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result = pfnImportExternalSemaphoreOpaqueFDExp(
        hContext, hDevice, pInteropSemaphoreDesc, phInteropSemaphore);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phInteropSemaphore =
            reinterpret_cast<ur_exp_interop_semaphore_handle_t>(
                ur_exp_interop_semaphore_factory.getInstance(
                    *phInteropSemaphore, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnDestroyExternalSemaphoreExp =
        dditable->ur.BindlessImagesExp.pfnDestroyExternalSemaphoreExp;
    if (nullptr == pfnDestroyExternalSemaphoreExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // convert loader handle to platform handle
    hInteropSemaphore =
        reinterpret_cast<ur_exp_interop_semaphore_object_t *>(hInteropSemaphore)
            ->handle;

    // forward to device-platform
    result =
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnWaitExternalSemaphoreExp =
        dditable->ur.BindlessImagesExp.pfnWaitExternalSemaphoreExp;
    if (nullptr == pfnWaitExternalSemaphoreExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hSemaphore =
        reinterpret_cast<ur_exp_interop_semaphore_object_t *>(hSemaphore)
            ->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result =
        pfnWaitExternalSemaphoreExp(hQueue, hSemaphore, numEventsInWaitList,
                                    phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnSignalExternalSemaphoreExp =
        dditable->ur.BindlessImagesExp.pfnSignalExternalSemaphoreExp;
    if (nullptr == pfnSignalExternalSemaphoreExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hSemaphore =
        reinterpret_cast<ur_exp_interop_semaphore_object_t *>(hSemaphore)
            ->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result =
        pfnSignalExternalSemaphoreExp(hQueue, hSemaphore, numEventsInWaitList,
                                      phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferCreateExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferCreateExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_exp_command_buffer_desc_t
        *pCommandBufferDesc, ///< [in][optional] CommandBuffer descriptor
    ur_exp_command_buffer_handle_t
        *phCommandBuffer ///< [out] pointer to Command-Buffer handle
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnCreateExp = dditable->ur.CommandBufferExp.pfnCreateExp;
    if (nullptr == pfnCreateExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handle to platform handle
    hDevice = reinterpret_cast<ur_device_object_t *>(hDevice)->handle;

    // forward to device-platform
    result =
        pfnCreateExp(hContext, hDevice, pCommandBufferDesc, phCommandBuffer);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phCommandBuffer = reinterpret_cast<ur_exp_command_buffer_handle_t>(
            ur_exp_command_buffer_factory.getInstance(*phCommandBuffer,
                                                      dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferRetainExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferRetainExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer ///< [in] handle of the command-buffer object
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnRetainExp = dditable->ur.CommandBufferExp.pfnRetainExp;
    if (nullptr == pfnRetainExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // forward to device-platform
    result = pfnRetainExp(hCommandBuffer);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferReleaseExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferReleaseExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer ///< [in] handle of the command-buffer object
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnReleaseExp = dditable->ur.CommandBufferExp.pfnReleaseExp;
    if (nullptr == pfnReleaseExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // forward to device-platform
    result = pfnReleaseExp(hCommandBuffer);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferFinalizeExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferFinalizeExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer ///< [in] handle of the command-buffer object
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnFinalizeExp = dditable->ur.CommandBufferExp.pfnFinalizeExp;
    if (nullptr == pfnFinalizeExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // forward to device-platform
    result = pfnFinalizeExp(hCommandBuffer);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendKernelLaunchExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,         ///< [in] handle of the command-buffer object
    ur_kernel_handle_t hKernel, ///< [in] kernel to append
    uint32_t workDim,           ///< [in] dimension of the kernel execution
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
    ur_exp_command_buffer_sync_point_t
        *pSyncPoint ///< [out][optional] sync point associated with this command
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnAppendKernelLaunchExp =
        dditable->ur.CommandBufferExp.pfnAppendKernelLaunchExp;
    if (nullptr == pfnAppendKernelLaunchExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // forward to device-platform
    result = pfnAppendKernelLaunchExp(hCommandBuffer, hKernel, workDim,
                                      pGlobalWorkOffset, pGlobalWorkSize,
                                      pLocalWorkSize, numSyncPointsInWaitList,
                                      pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendUSMMemcpyExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendUSMMemcpyExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer, ///< [in] handle of the command-buffer object.
    void *pDst,         ///< [in] Location the data will be copied to.
    const void *pSrc,   ///< [in] The data to be copied.
    size_t size,        ///< [in] The number of bytes to copy
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t
        *pSyncPoint ///< [out][optional] sync point associated with this command
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnAppendUSMMemcpyExp =
        dditable->ur.CommandBufferExp.pfnAppendUSMMemcpyExp;
    if (nullptr == pfnAppendUSMMemcpyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // forward to device-platform
    result = pfnAppendUSMMemcpyExp(hCommandBuffer, pDst, pSrc, size,
                                   numSyncPointsInWaitList, pSyncPointWaitList,
                                   pSyncPoint);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnAppendUSMFillExp =
        dditable->ur.CommandBufferExp.pfnAppendUSMFillExp;
    if (nullptr == pfnAppendUSMFillExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // forward to device-platform
    result = pfnAppendUSMFillExp(hCommandBuffer, pMemory, pPattern, patternSize,
                                 size, numSyncPointsInWaitList,
                                 pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferCopyExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] handle of the command-buffer object.
    ur_mem_handle_t hSrcMem, ///< [in] The data to be copied.
    ur_mem_handle_t hDstMem, ///< [in] The location the data will be copied to.
    size_t srcOffset,        ///< [in] Offset into the source memory.
    size_t dstOffset,        ///< [in] Offset into the destination memory
    size_t size,             ///< [in] The number of bytes to be copied.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t
        *pSyncPoint ///< [out][optional] sync point associated with this command
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnAppendMemBufferCopyExp =
        dditable->ur.CommandBufferExp.pfnAppendMemBufferCopyExp;
    if (nullptr == pfnAppendMemBufferCopyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // convert loader handle to platform handle
    hSrcMem = reinterpret_cast<ur_mem_object_t *>(hSrcMem)->handle;

    // convert loader handle to platform handle
    hDstMem = reinterpret_cast<ur_mem_object_t *>(hDstMem)->handle;

    // forward to device-platform
    result = pfnAppendMemBufferCopyExp(
        hCommandBuffer, hSrcMem, hDstMem, srcOffset, dstOffset, size,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferWriteExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] handle of the buffer object.
    size_t offset,           ///< [in] offset in bytes in the buffer object.
    size_t size,             ///< [in] size in bytes of data being written.
    const void *
        pSrc, ///< [in] pointer to host memory where data is to be written from.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t
        *pSyncPoint ///< [out][optional] sync point associated with this command
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnAppendMemBufferWriteExp =
        dditable->ur.CommandBufferExp.pfnAppendMemBufferWriteExp;
    if (nullptr == pfnAppendMemBufferWriteExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // convert loader handle to platform handle
    hBuffer = reinterpret_cast<ur_mem_object_t *>(hBuffer)->handle;

    // forward to device-platform
    result = pfnAppendMemBufferWriteExp(hCommandBuffer, hBuffer, offset, size,
                                        pSrc, numSyncPointsInWaitList,
                                        pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferReadExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] handle of the buffer object.
    size_t offset,           ///< [in] offset in bytes in the buffer object.
    size_t size,             ///< [in] size in bytes of data being written.
    void *pDst, ///< [in] pointer to host memory where data is to be written to.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t
        *pSyncPoint ///< [out][optional] sync point associated with this command
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnAppendMemBufferReadExp =
        dditable->ur.CommandBufferExp.pfnAppendMemBufferReadExp;
    if (nullptr == pfnAppendMemBufferReadExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // convert loader handle to platform handle
    hBuffer = reinterpret_cast<ur_mem_object_t *>(hBuffer)->handle;

    // forward to device-platform
    result = pfnAppendMemBufferReadExp(hCommandBuffer, hBuffer, offset, size,
                                       pDst, numSyncPointsInWaitList,
                                       pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferCopyRectExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyRectExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] handle of the command-buffer object.
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
    ur_exp_command_buffer_sync_point_t
        *pSyncPoint ///< [out][optional] sync point associated with this command
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnAppendMemBufferCopyRectExp =
        dditable->ur.CommandBufferExp.pfnAppendMemBufferCopyRectExp;
    if (nullptr == pfnAppendMemBufferCopyRectExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // convert loader handle to platform handle
    hSrcMem = reinterpret_cast<ur_mem_object_t *>(hSrcMem)->handle;

    // convert loader handle to platform handle
    hDstMem = reinterpret_cast<ur_mem_object_t *>(hDstMem)->handle;

    // forward to device-platform
    result = pfnAppendMemBufferCopyRectExp(
        hCommandBuffer, hSrcMem, hDstMem, srcOrigin, dstOrigin, region,
        srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferWriteRectExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteRectExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] handle of the buffer object.
    ur_rect_offset_t bufferOffset, ///< [in] 3D offset in the buffer.
    ur_rect_offset_t hostOffset,   ///< [in] 3D offset in the host region.
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth.
    size_t
        bufferRowPitch, ///< [in] length of each row in bytes in the buffer object.
    size_t
        bufferSlicePitch, ///< [in] length of each 2D slice in bytes in the buffer object being
                          ///< written.
    size_t
        hostRowPitch, ///< [in] length of each row in bytes in the host memory region pointed to
                      ///< by pSrc.
    size_t
        hostSlicePitch, ///< [in] length of each 2D slice in bytes in the host memory region
                        ///< pointed to by pSrc.
    void *
        pSrc, ///< [in] pointer to host memory where data is to be written from.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t
        *pSyncPoint ///< [out][optional] sync point associated with this command
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnAppendMemBufferWriteRectExp =
        dditable->ur.CommandBufferExp.pfnAppendMemBufferWriteRectExp;
    if (nullptr == pfnAppendMemBufferWriteRectExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // convert loader handle to platform handle
    hBuffer = reinterpret_cast<ur_mem_object_t *>(hBuffer)->handle;

    // forward to device-platform
    result = pfnAppendMemBufferWriteRectExp(
        hCommandBuffer, hBuffer, bufferOffset, hostOffset, region,
        bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferReadRectExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadRectExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] handle of the buffer object.
    ur_rect_offset_t bufferOffset, ///< [in] 3D offset in the buffer.
    ur_rect_offset_t hostOffset,   ///< [in] 3D offset in the host region.
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth.
    size_t
        bufferRowPitch, ///< [in] length of each row in bytes in the buffer object.
    size_t
        bufferSlicePitch, ///< [in] length of each 2D slice in bytes in the buffer object being read.
    size_t
        hostRowPitch, ///< [in] length of each row in bytes in the host memory region pointed to
                      ///< by pDst.
    size_t
        hostSlicePitch, ///< [in] length of each 2D slice in bytes in the host memory region
                        ///< pointed to by pDst.
    void *pDst, ///< [in] pointer to host memory where data is to be read into.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t
        *pSyncPoint ///< [out][optional] sync point associated with this command
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnAppendMemBufferReadRectExp =
        dditable->ur.CommandBufferExp.pfnAppendMemBufferReadRectExp;
    if (nullptr == pfnAppendMemBufferReadRectExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // convert loader handle to platform handle
    hBuffer = reinterpret_cast<ur_mem_object_t *>(hBuffer)->handle;

    // forward to device-platform
    result = pfnAppendMemBufferReadRectExp(
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnAppendMemBufferFillExp =
        dditable->ur.CommandBufferExp.pfnAppendMemBufferFillExp;
    if (nullptr == pfnAppendMemBufferFillExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // convert loader handle to platform handle
    hBuffer = reinterpret_cast<ur_mem_object_t *>(hBuffer)->handle;

    // forward to device-platform
    result = pfnAppendMemBufferFillExp(
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnAppendUSMPrefetchExp =
        dditable->ur.CommandBufferExp.pfnAppendUSMPrefetchExp;
    if (nullptr == pfnAppendUSMPrefetchExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // forward to device-platform
    result = pfnAppendUSMPrefetchExp(hCommandBuffer, pMemory, size, flags,
                                     numSyncPointsInWaitList,
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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnAppendUSMAdviseExp =
        dditable->ur.CommandBufferExp.pfnAppendUSMAdviseExp;
    if (nullptr == pfnAppendUSMAdviseExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // forward to device-platform
    result = pfnAppendUSMAdviseExp(hCommandBuffer, pMemory, size, advice,
                                   numSyncPointsInWaitList, pSyncPointWaitList,
                                   pSyncPoint);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferEnqueueExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer, ///< [in] handle of the command-buffer object.
    ur_queue_handle_t
        hQueue, ///< [in] the queue to submit this command-buffer for execution.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the command-buffer execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating no wait
    ///< events.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command-buffer execution instance.
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->dditable;
    auto pfnEnqueueExp = dditable->ur.CommandBufferExp.pfnEnqueueExp;
    if (nullptr == pfnEnqueueExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hCommandBuffer =
        reinterpret_cast<ur_exp_command_buffer_object_t *>(hCommandBuffer)
            ->handle;

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnEnqueueExp(hCommandBuffer, hQueue, numEventsInWaitList,
                           phEventWaitListLocal.data(), phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_queue_object_t *>(hQueue)->dditable;
    auto pfnCooperativeKernelLaunchExp =
        dditable->ur.EnqueueExp.pfnCooperativeKernelLaunchExp;
    if (nullptr == pfnCooperativeKernelLaunchExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hQueue = reinterpret_cast<ur_queue_object_t *>(hQueue)->handle;

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // convert loader handles to platform handles
    auto phEventWaitListLocal =
        std::vector<ur_event_handle_t>(numEventsInWaitList);
    for (size_t i = 0; i < numEventsInWaitList; ++i) {
        phEventWaitListLocal[i] =
            reinterpret_cast<ur_event_object_t *>(phEventWaitList[i])->handle;
    }

    // forward to device-platform
    result = pfnCooperativeKernelLaunchExp(
        hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
        pLocalWorkSize, numEventsInWaitList, phEventWaitListLocal.data(),
        phEvent);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        if (nullptr != phEvent) {
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance(*phEvent, dditable));
        }
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSuggestMaxCooperativeGroupCountExp
__urdlllocal ur_result_t UR_APICALL urKernelSuggestMaxCooperativeGroupCountExp(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t *pGroupCountRet    ///< [out] pointer to maximum number of groups
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_kernel_object_t *>(hKernel)->dditable;
    auto pfnSuggestMaxCooperativeGroupCountExp =
        dditable->ur.KernelExp.pfnSuggestMaxCooperativeGroupCountExp;
    if (nullptr == pfnSuggestMaxCooperativeGroupCountExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hKernel = reinterpret_cast<ur_kernel_object_t *>(hKernel)->handle;

    // forward to device-platform
    result = pfnSuggestMaxCooperativeGroupCountExp(hKernel, pGroupCountRet);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_program_object_t *>(hProgram)->dditable;
    auto pfnBuildExp = dditable->ur.ProgramExp.pfnBuildExp;
    if (nullptr == pfnBuildExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // convert loader handles to platform handles
    auto phDevicesLocal = std::vector<ur_device_handle_t>(numDevices);
    for (size_t i = 0; i < numDevices; ++i) {
        phDevicesLocal[i] =
            reinterpret_cast<ur_device_object_t *>(phDevices[i])->handle;
    }

    // forward to device-platform
    result = pfnBuildExp(hProgram, numDevices, phDevicesLocal.data(), pOptions);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_program_object_t *>(hProgram)->dditable;
    auto pfnCompileExp = dditable->ur.ProgramExp.pfnCompileExp;
    if (nullptr == pfnCompileExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hProgram = reinterpret_cast<ur_program_object_t *>(hProgram)->handle;

    // convert loader handles to platform handles
    auto phDevicesLocal = std::vector<ur_device_handle_t>(numDevices);
    for (size_t i = 0; i < numDevices; ++i) {
        phDevicesLocal[i] =
            reinterpret_cast<ur_device_object_t *>(phDevices[i])->handle;
    }

    // forward to device-platform
    result =
        pfnCompileExp(hProgram, numDevices, phDevicesLocal.data(), pOptions);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnLinkExp = dditable->ur.ProgramExp.pfnLinkExp;
    if (nullptr == pfnLinkExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // convert loader handles to platform handles
    auto phDevicesLocal = std::vector<ur_device_handle_t>(numDevices);
    for (size_t i = 0; i < numDevices; ++i) {
        phDevicesLocal[i] =
            reinterpret_cast<ur_device_object_t *>(phDevices[i])->handle;
    }

    // convert loader handles to platform handles
    auto phProgramsLocal = std::vector<ur_program_handle_t>(count);
    for (size_t i = 0; i < count; ++i) {
        phProgramsLocal[i] =
            reinterpret_cast<ur_program_object_t *>(phPrograms[i])->handle;
    }

    // forward to device-platform
    result = pfnLinkExp(hContext, numDevices, phDevicesLocal.data(), count,
                        phProgramsLocal.data(), pOptions, phProgram);

    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    try {
        // convert platform handle to loader handle
        *phProgram = reinterpret_cast<ur_program_handle_t>(
            ur_program_factory.getInstance(*phProgram, dditable));
    } catch (std::bad_alloc &) {
        result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMImportExp
__urdlllocal ur_result_t UR_APICALL urUSMImportExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem,                   ///< [in] pointer to host memory object
    size_t size ///< [in] size in bytes of the host memory object to be imported
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnImportExp = dditable->ur.USMExp.pfnImportExp;
    if (nullptr == pfnImportExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnImportExp(hContext, pMem, size);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMReleaseExp
__urdlllocal ur_result_t UR_APICALL urUSMReleaseExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to host memory object
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable = reinterpret_cast<ur_context_object_t *>(hContext)->dditable;
    auto pfnReleaseExp = dditable->ur.USMExp.pfnReleaseExp;
    if (nullptr == pfnReleaseExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    hContext = reinterpret_cast<ur_context_object_t *>(hContext)->handle;

    // forward to device-platform
    result = pfnReleaseExp(hContext, pMem);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PEnablePeerAccessExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PEnablePeerAccessExp(
    ur_device_handle_t
        commandDevice,            ///< [in] handle of the command device object
    ur_device_handle_t peerDevice ///< [in] handle of the peer device object
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_device_object_t *>(commandDevice)->dditable;
    auto pfnEnablePeerAccessExp = dditable->ur.UsmP2PExp.pfnEnablePeerAccessExp;
    if (nullptr == pfnEnablePeerAccessExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    commandDevice =
        reinterpret_cast<ur_device_object_t *>(commandDevice)->handle;

    // convert loader handle to platform handle
    peerDevice = reinterpret_cast<ur_device_object_t *>(peerDevice)->handle;

    // forward to device-platform
    result = pfnEnablePeerAccessExp(commandDevice, peerDevice);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PDisablePeerAccessExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PDisablePeerAccessExp(
    ur_device_handle_t
        commandDevice,            ///< [in] handle of the command device object
    ur_device_handle_t peerDevice ///< [in] handle of the peer device object
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_device_object_t *>(commandDevice)->dditable;
    auto pfnDisablePeerAccessExp =
        dditable->ur.UsmP2PExp.pfnDisablePeerAccessExp;
    if (nullptr == pfnDisablePeerAccessExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    commandDevice =
        reinterpret_cast<ur_device_object_t *>(commandDevice)->handle;

    // convert loader handle to platform handle
    peerDevice = reinterpret_cast<ur_device_object_t *>(peerDevice)->handle;

    // forward to device-platform
    result = pfnDisablePeerAccessExp(commandDevice, peerDevice);

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
    ur_result_t result = UR_RESULT_SUCCESS;

    // extract platform's function pointer table
    auto dditable =
        reinterpret_cast<ur_device_object_t *>(commandDevice)->dditable;
    auto pfnPeerAccessGetInfoExp =
        dditable->ur.UsmP2PExp.pfnPeerAccessGetInfoExp;
    if (nullptr == pfnPeerAccessGetInfoExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    // convert loader handle to platform handle
    commandDevice =
        reinterpret_cast<ur_device_object_t *>(commandDevice)->handle;

    // convert loader handle to platform handle
    peerDevice = reinterpret_cast<ur_device_object_t *>(peerDevice)->handle;

    // forward to device-platform
    result = pfnPeerAccessGetInfoExp(commandDevice, peerDevice, propName,
                                     propSize, pPropValue, pPropSizeRet);

    return result;
}

} // namespace ur_loader

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Global table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetGlobalProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_global_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetGlobalProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                                 "urGetGlobalProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus = getTable(version, &platform.dditable.ur.Global);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnAdapterGet = ur_loader::urAdapterGet;
            pDdiTable->pfnAdapterRelease = ur_loader::urAdapterRelease;
            pDdiTable->pfnAdapterRetain = ur_loader::urAdapterRetain;
            pDdiTable->pfnAdapterGetLastError =
                ur_loader::urAdapterGetLastError;
            pDdiTable->pfnAdapterGetInfo = ur_loader::urAdapterGetInfo;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.Global;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's BindlessImagesExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetBindlessImagesExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_bindless_images_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable =
            reinterpret_cast<ur_pfnGetBindlessImagesExpProcAddrTable_t>(
                ur_loader::LibLoader::getFunctionPtr(
                    platform.handle.get(),
                    "urGetBindlessImagesExpProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus =
            getTable(version, &platform.dditable.ur.BindlessImagesExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnUnsampledImageHandleDestroyExp =
                ur_loader::urBindlessImagesUnsampledImageHandleDestroyExp;
            pDdiTable->pfnSampledImageHandleDestroyExp =
                ur_loader::urBindlessImagesSampledImageHandleDestroyExp;
            pDdiTable->pfnImageAllocateExp =
                ur_loader::urBindlessImagesImageAllocateExp;
            pDdiTable->pfnImageFreeExp =
                ur_loader::urBindlessImagesImageFreeExp;
            pDdiTable->pfnUnsampledImageCreateExp =
                ur_loader::urBindlessImagesUnsampledImageCreateExp;
            pDdiTable->pfnSampledImageCreateExp =
                ur_loader::urBindlessImagesSampledImageCreateExp;
            pDdiTable->pfnImageCopyExp =
                ur_loader::urBindlessImagesImageCopyExp;
            pDdiTable->pfnImageGetInfoExp =
                ur_loader::urBindlessImagesImageGetInfoExp;
            pDdiTable->pfnMipmapGetLevelExp =
                ur_loader::urBindlessImagesMipmapGetLevelExp;
            pDdiTable->pfnMipmapFreeExp =
                ur_loader::urBindlessImagesMipmapFreeExp;
            pDdiTable->pfnImportOpaqueFDExp =
                ur_loader::urBindlessImagesImportOpaqueFDExp;
            pDdiTable->pfnMapExternalArrayExp =
                ur_loader::urBindlessImagesMapExternalArrayExp;
            pDdiTable->pfnReleaseInteropExp =
                ur_loader::urBindlessImagesReleaseInteropExp;
            pDdiTable->pfnImportExternalSemaphoreOpaqueFDExp =
                ur_loader::urBindlessImagesImportExternalSemaphoreOpaqueFDExp;
            pDdiTable->pfnDestroyExternalSemaphoreExp =
                ur_loader::urBindlessImagesDestroyExternalSemaphoreExp;
            pDdiTable->pfnWaitExternalSemaphoreExp =
                ur_loader::urBindlessImagesWaitExternalSemaphoreExp;
            pDdiTable->pfnSignalExternalSemaphoreExp =
                ur_loader::urBindlessImagesSignalExternalSemaphoreExp;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable = ur_loader::context->platforms.front()
                             .dditable.ur.BindlessImagesExp;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's CommandBufferExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetCommandBufferExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_command_buffer_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable =
            reinterpret_cast<ur_pfnGetCommandBufferExpProcAddrTable_t>(
                ur_loader::LibLoader::getFunctionPtr(
                    platform.handle.get(),
                    "urGetCommandBufferExpProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus =
            getTable(version, &platform.dditable.ur.CommandBufferExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnCreateExp = ur_loader::urCommandBufferCreateExp;
            pDdiTable->pfnRetainExp = ur_loader::urCommandBufferRetainExp;
            pDdiTable->pfnReleaseExp = ur_loader::urCommandBufferReleaseExp;
            pDdiTable->pfnFinalizeExp = ur_loader::urCommandBufferFinalizeExp;
            pDdiTable->pfnAppendKernelLaunchExp =
                ur_loader::urCommandBufferAppendKernelLaunchExp;
            pDdiTable->pfnAppendUSMMemcpyExp =
                ur_loader::urCommandBufferAppendUSMMemcpyExp;
            pDdiTable->pfnAppendUSMFillExp =
                ur_loader::urCommandBufferAppendUSMFillExp;
            pDdiTable->pfnAppendMemBufferCopyExp =
                ur_loader::urCommandBufferAppendMemBufferCopyExp;
            pDdiTable->pfnAppendMemBufferWriteExp =
                ur_loader::urCommandBufferAppendMemBufferWriteExp;
            pDdiTable->pfnAppendMemBufferReadExp =
                ur_loader::urCommandBufferAppendMemBufferReadExp;
            pDdiTable->pfnAppendMemBufferCopyRectExp =
                ur_loader::urCommandBufferAppendMemBufferCopyRectExp;
            pDdiTable->pfnAppendMemBufferWriteRectExp =
                ur_loader::urCommandBufferAppendMemBufferWriteRectExp;
            pDdiTable->pfnAppendMemBufferReadRectExp =
                ur_loader::urCommandBufferAppendMemBufferReadRectExp;
            pDdiTable->pfnAppendMemBufferFillExp =
                ur_loader::urCommandBufferAppendMemBufferFillExp;
            pDdiTable->pfnAppendUSMPrefetchExp =
                ur_loader::urCommandBufferAppendUSMPrefetchExp;
            pDdiTable->pfnAppendUSMAdviseExp =
                ur_loader::urCommandBufferAppendUSMAdviseExp;
            pDdiTable->pfnEnqueueExp = ur_loader::urCommandBufferEnqueueExp;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable = ur_loader::context->platforms.front()
                             .dditable.ur.CommandBufferExp;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Context table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetContextProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_context_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetContextProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                                 "urGetContextProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus = getTable(version, &platform.dditable.ur.Context);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnCreate = ur_loader::urContextCreate;
            pDdiTable->pfnRetain = ur_loader::urContextRetain;
            pDdiTable->pfnRelease = ur_loader::urContextRelease;
            pDdiTable->pfnGetInfo = ur_loader::urContextGetInfo;
            pDdiTable->pfnGetNativeHandle = ur_loader::urContextGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle =
                ur_loader::urContextCreateWithNativeHandle;
            pDdiTable->pfnSetExtendedDeleter =
                ur_loader::urContextSetExtendedDeleter;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.Context;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Enqueue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_enqueue_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetEnqueueProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                                 "urGetEnqueueProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus = getTable(version, &platform.dditable.ur.Enqueue);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnKernelLaunch = ur_loader::urEnqueueKernelLaunch;
            pDdiTable->pfnEventsWait = ur_loader::urEnqueueEventsWait;
            pDdiTable->pfnEventsWaitWithBarrier =
                ur_loader::urEnqueueEventsWaitWithBarrier;
            pDdiTable->pfnMemBufferRead = ur_loader::urEnqueueMemBufferRead;
            pDdiTable->pfnMemBufferWrite = ur_loader::urEnqueueMemBufferWrite;
            pDdiTable->pfnMemBufferReadRect =
                ur_loader::urEnqueueMemBufferReadRect;
            pDdiTable->pfnMemBufferWriteRect =
                ur_loader::urEnqueueMemBufferWriteRect;
            pDdiTable->pfnMemBufferCopy = ur_loader::urEnqueueMemBufferCopy;
            pDdiTable->pfnMemBufferCopyRect =
                ur_loader::urEnqueueMemBufferCopyRect;
            pDdiTable->pfnMemBufferFill = ur_loader::urEnqueueMemBufferFill;
            pDdiTable->pfnMemImageRead = ur_loader::urEnqueueMemImageRead;
            pDdiTable->pfnMemImageWrite = ur_loader::urEnqueueMemImageWrite;
            pDdiTable->pfnMemImageCopy = ur_loader::urEnqueueMemImageCopy;
            pDdiTable->pfnMemBufferMap = ur_loader::urEnqueueMemBufferMap;
            pDdiTable->pfnMemUnmap = ur_loader::urEnqueueMemUnmap;
            pDdiTable->pfnUSMFill = ur_loader::urEnqueueUSMFill;
            pDdiTable->pfnUSMMemcpy = ur_loader::urEnqueueUSMMemcpy;
            pDdiTable->pfnUSMPrefetch = ur_loader::urEnqueueUSMPrefetch;
            pDdiTable->pfnUSMAdvise = ur_loader::urEnqueueUSMAdvise;
            pDdiTable->pfnUSMFill2D = ur_loader::urEnqueueUSMFill2D;
            pDdiTable->pfnUSMMemcpy2D = ur_loader::urEnqueueUSMMemcpy2D;
            pDdiTable->pfnDeviceGlobalVariableWrite =
                ur_loader::urEnqueueDeviceGlobalVariableWrite;
            pDdiTable->pfnDeviceGlobalVariableRead =
                ur_loader::urEnqueueDeviceGlobalVariableRead;
            pDdiTable->pfnReadHostPipe = ur_loader::urEnqueueReadHostPipe;
            pDdiTable->pfnWriteHostPipe = ur_loader::urEnqueueWriteHostPipe;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.Enqueue;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's EnqueueExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_enqueue_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetEnqueueExpProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(
                platform.handle.get(), "urGetEnqueueExpProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus =
            getTable(version, &platform.dditable.ur.EnqueueExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnCooperativeKernelLaunchExp =
                ur_loader::urEnqueueCooperativeKernelLaunchExp;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.EnqueueExp;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Event table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetEventProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_event_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetEventProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                                 "urGetEventProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus = getTable(version, &platform.dditable.ur.Event);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnGetInfo = ur_loader::urEventGetInfo;
            pDdiTable->pfnGetProfilingInfo = ur_loader::urEventGetProfilingInfo;
            pDdiTable->pfnWait = ur_loader::urEventWait;
            pDdiTable->pfnRetain = ur_loader::urEventRetain;
            pDdiTable->pfnRelease = ur_loader::urEventRelease;
            pDdiTable->pfnGetNativeHandle = ur_loader::urEventGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle =
                ur_loader::urEventCreateWithNativeHandle;
            pDdiTable->pfnSetCallback = ur_loader::urEventSetCallback;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.Event;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Kernel table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetKernelProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_kernel_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetKernelProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                                 "urGetKernelProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus = getTable(version, &platform.dditable.ur.Kernel);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnCreate = ur_loader::urKernelCreate;
            pDdiTable->pfnGetInfo = ur_loader::urKernelGetInfo;
            pDdiTable->pfnGetGroupInfo = ur_loader::urKernelGetGroupInfo;
            pDdiTable->pfnGetSubGroupInfo = ur_loader::urKernelGetSubGroupInfo;
            pDdiTable->pfnRetain = ur_loader::urKernelRetain;
            pDdiTable->pfnRelease = ur_loader::urKernelRelease;
            pDdiTable->pfnGetNativeHandle = ur_loader::urKernelGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle =
                ur_loader::urKernelCreateWithNativeHandle;
            pDdiTable->pfnSetArgValue = ur_loader::urKernelSetArgValue;
            pDdiTable->pfnSetArgLocal = ur_loader::urKernelSetArgLocal;
            pDdiTable->pfnSetArgPointer = ur_loader::urKernelSetArgPointer;
            pDdiTable->pfnSetExecInfo = ur_loader::urKernelSetExecInfo;
            pDdiTable->pfnSetArgSampler = ur_loader::urKernelSetArgSampler;
            pDdiTable->pfnSetArgMemObj = ur_loader::urKernelSetArgMemObj;
            pDdiTable->pfnSetSpecializationConstants =
                ur_loader::urKernelSetSpecializationConstants;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.Kernel;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's KernelExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetKernelExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_kernel_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetKernelExpProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(
                platform.handle.get(), "urGetKernelExpProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus =
            getTable(version, &platform.dditable.ur.KernelExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnSuggestMaxCooperativeGroupCountExp =
                ur_loader::urKernelSuggestMaxCooperativeGroupCountExp;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.KernelExp;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Mem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetMemProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_mem_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetMemProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                                 "urGetMemProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus = getTable(version, &platform.dditable.ur.Mem);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnImageCreate = ur_loader::urMemImageCreate;
            pDdiTable->pfnBufferCreate = ur_loader::urMemBufferCreate;
            pDdiTable->pfnRetain = ur_loader::urMemRetain;
            pDdiTable->pfnRelease = ur_loader::urMemRelease;
            pDdiTable->pfnBufferPartition = ur_loader::urMemBufferPartition;
            pDdiTable->pfnGetNativeHandle = ur_loader::urMemGetNativeHandle;
            pDdiTable->pfnBufferCreateWithNativeHandle =
                ur_loader::urMemBufferCreateWithNativeHandle;
            pDdiTable->pfnImageCreateWithNativeHandle =
                ur_loader::urMemImageCreateWithNativeHandle;
            pDdiTable->pfnGetInfo = ur_loader::urMemGetInfo;
            pDdiTable->pfnImageGetInfo = ur_loader::urMemImageGetInfo;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable = ur_loader::context->platforms.front().dditable.ur.Mem;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's PhysicalMem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_physical_mem_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetPhysicalMemProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(
                platform.handle.get(), "urGetPhysicalMemProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus =
            getTable(version, &platform.dditable.ur.PhysicalMem);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnCreate = ur_loader::urPhysicalMemCreate;
            pDdiTable->pfnRetain = ur_loader::urPhysicalMemRetain;
            pDdiTable->pfnRelease = ur_loader::urPhysicalMemRelease;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.PhysicalMem;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Platform table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_platform_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetPlatformProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                                 "urGetPlatformProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus = getTable(version, &platform.dditable.ur.Platform);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnGet = ur_loader::urPlatformGet;
            pDdiTable->pfnGetInfo = ur_loader::urPlatformGetInfo;
            pDdiTable->pfnGetNativeHandle =
                ur_loader::urPlatformGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle =
                ur_loader::urPlatformCreateWithNativeHandle;
            pDdiTable->pfnGetApiVersion = ur_loader::urPlatformGetApiVersion;
            pDdiTable->pfnGetBackendOption =
                ur_loader::urPlatformGetBackendOption;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.Platform;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Program table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetProgramProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_program_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetProgramProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                                 "urGetProgramProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus = getTable(version, &platform.dditable.ur.Program);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnCreateWithIL = ur_loader::urProgramCreateWithIL;
            pDdiTable->pfnCreateWithBinary =
                ur_loader::urProgramCreateWithBinary;
            pDdiTable->pfnBuild = ur_loader::urProgramBuild;
            pDdiTable->pfnCompile = ur_loader::urProgramCompile;
            pDdiTable->pfnLink = ur_loader::urProgramLink;
            pDdiTable->pfnRetain = ur_loader::urProgramRetain;
            pDdiTable->pfnRelease = ur_loader::urProgramRelease;
            pDdiTable->pfnGetFunctionPointer =
                ur_loader::urProgramGetFunctionPointer;
            pDdiTable->pfnGetInfo = ur_loader::urProgramGetInfo;
            pDdiTable->pfnGetBuildInfo = ur_loader::urProgramGetBuildInfo;
            pDdiTable->pfnSetSpecializationConstants =
                ur_loader::urProgramSetSpecializationConstants;
            pDdiTable->pfnGetNativeHandle = ur_loader::urProgramGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle =
                ur_loader::urProgramCreateWithNativeHandle;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.Program;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's ProgramExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetProgramExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_program_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetProgramExpProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(
                platform.handle.get(), "urGetProgramExpProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus =
            getTable(version, &platform.dditable.ur.ProgramExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnBuildExp = ur_loader::urProgramBuildExp;
            pDdiTable->pfnCompileExp = ur_loader::urProgramCompileExp;
            pDdiTable->pfnLinkExp = ur_loader::urProgramLinkExp;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.ProgramExp;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Queue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetQueueProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_queue_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetQueueProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                                 "urGetQueueProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus = getTable(version, &platform.dditable.ur.Queue);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnGetInfo = ur_loader::urQueueGetInfo;
            pDdiTable->pfnCreate = ur_loader::urQueueCreate;
            pDdiTable->pfnRetain = ur_loader::urQueueRetain;
            pDdiTable->pfnRelease = ur_loader::urQueueRelease;
            pDdiTable->pfnGetNativeHandle = ur_loader::urQueueGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle =
                ur_loader::urQueueCreateWithNativeHandle;
            pDdiTable->pfnFinish = ur_loader::urQueueFinish;
            pDdiTable->pfnFlush = ur_loader::urQueueFlush;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.Queue;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Sampler table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_sampler_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetSamplerProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                                 "urGetSamplerProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus = getTable(version, &platform.dditable.ur.Sampler);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnCreate = ur_loader::urSamplerCreate;
            pDdiTable->pfnRetain = ur_loader::urSamplerRetain;
            pDdiTable->pfnRelease = ur_loader::urSamplerRelease;
            pDdiTable->pfnGetInfo = ur_loader::urSamplerGetInfo;
            pDdiTable->pfnGetNativeHandle = ur_loader::urSamplerGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle =
                ur_loader::urSamplerCreateWithNativeHandle;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.Sampler;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USM table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetUSMProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_usm_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetUSMProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                                 "urGetUSMProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus = getTable(version, &platform.dditable.ur.USM);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnHostAlloc = ur_loader::urUSMHostAlloc;
            pDdiTable->pfnDeviceAlloc = ur_loader::urUSMDeviceAlloc;
            pDdiTable->pfnSharedAlloc = ur_loader::urUSMSharedAlloc;
            pDdiTable->pfnFree = ur_loader::urUSMFree;
            pDdiTable->pfnGetMemAllocInfo = ur_loader::urUSMGetMemAllocInfo;
            pDdiTable->pfnPoolCreate = ur_loader::urUSMPoolCreate;
            pDdiTable->pfnPoolRetain = ur_loader::urUSMPoolRetain;
            pDdiTable->pfnPoolRelease = ur_loader::urUSMPoolRelease;
            pDdiTable->pfnPoolGetInfo = ur_loader::urUSMPoolGetInfo;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable = ur_loader::context->platforms.front().dditable.ur.USM;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USMExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetUSMExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_usm_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetUSMExpProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                                 "urGetUSMExpProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus = getTable(version, &platform.dditable.ur.USMExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnPitchedAllocExp = ur_loader::urUSMPitchedAllocExp;
            pDdiTable->pfnImportExp = ur_loader::urUSMImportExp;
            pDdiTable->pfnReleaseExp = ur_loader::urUSMReleaseExp;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.USMExp;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's UsmP2PExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetUsmP2PExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_usm_p2p_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetUsmP2PExpProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(
                platform.handle.get(), "urGetUsmP2PExpProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus =
            getTable(version, &platform.dditable.ur.UsmP2PExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnEnablePeerAccessExp =
                ur_loader::urUsmP2PEnablePeerAccessExp;
            pDdiTable->pfnDisablePeerAccessExp =
                ur_loader::urUsmP2PDisablePeerAccessExp;
            pDdiTable->pfnPeerAccessGetInfoExp =
                ur_loader::urUsmP2PPeerAccessGetInfoExp;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.UsmP2PExp;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's VirtualMem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetVirtualMemProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_virtual_mem_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetVirtualMemProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(
                platform.handle.get(), "urGetVirtualMemProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus =
            getTable(version, &platform.dditable.ur.VirtualMem);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnGranularityGetInfo =
                ur_loader::urVirtualMemGranularityGetInfo;
            pDdiTable->pfnReserve = ur_loader::urVirtualMemReserve;
            pDdiTable->pfnFree = ur_loader::urVirtualMemFree;
            pDdiTable->pfnMap = ur_loader::urVirtualMemMap;
            pDdiTable->pfnUnmap = ur_loader::urVirtualMemUnmap;
            pDdiTable->pfnSetAccess = ur_loader::urVirtualMemSetAccess;
            pDdiTable->pfnGetInfo = ur_loader::urVirtualMemGetInfo;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.VirtualMem;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Device table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_device_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (ur_loader::context->version < version) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for (auto &platform : ur_loader::context->platforms) {
        if (platform.initStatus != UR_RESULT_SUCCESS) {
            continue;
        }
        auto getTable = reinterpret_cast<ur_pfnGetDeviceProcAddrTable_t>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                                 "urGetDeviceProcAddrTable"));
        if (!getTable) {
            continue;
        }
        platform.initStatus = getTable(version, &platform.dditable.ur.Device);
    }

    if (UR_RESULT_SUCCESS == result) {
        if (ur_loader::context->platforms.size() != 1 ||
            ur_loader::context->forceIntercept) {
            // return pointers to loader's DDIs
            pDdiTable->pfnGet = ur_loader::urDeviceGet;
            pDdiTable->pfnGetInfo = ur_loader::urDeviceGetInfo;
            pDdiTable->pfnRetain = ur_loader::urDeviceRetain;
            pDdiTable->pfnRelease = ur_loader::urDeviceRelease;
            pDdiTable->pfnPartition = ur_loader::urDevicePartition;
            pDdiTable->pfnSelectBinary = ur_loader::urDeviceSelectBinary;
            pDdiTable->pfnGetNativeHandle = ur_loader::urDeviceGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle =
                ur_loader::urDeviceCreateWithNativeHandle;
            pDdiTable->pfnGetGlobalTimestamps =
                ur_loader::urDeviceGetGlobalTimestamps;
        } else {
            // return pointers directly to platform's DDIs
            *pDdiTable =
                ur_loader::context->platforms.front().dditable.ur.Device;
        }
    }

    return result;
}

#if defined(__cplusplus)
}
#endif
