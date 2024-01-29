/*
 *
 * Copyright (C) 2019-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_null.cpp
 *
 */
#include "ur_null.hpp"

namespace driver {
//////////////////////////////////////////////////////////////////////////
context_t d_context;

//////////////////////////////////////////////////////////////////////////
context_t::context_t() {
    platform = get();
    //////////////////////////////////////////////////////////////////////////
    urDdiTable.Global.pfnAdapterGet = [](uint32_t NumAdapters,
                                         ur_adapter_handle_t *phAdapters,
                                         uint32_t *pNumAdapters) {
        if (phAdapters != nullptr && NumAdapters != 1) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
        if (pNumAdapters != nullptr) {
            *pNumAdapters = 1;
        }
        if (nullptr != phAdapters) {
            *reinterpret_cast<void **>(phAdapters) = d_context.platform;
        }

        return UR_RESULT_SUCCESS;
    };
    //////////////////////////////////////////////////////////////////////////
    urDdiTable.Global.pfnAdapterRelease = [](ur_adapter_handle_t) {
        return UR_RESULT_SUCCESS;
    };
    //////////////////////////////////////////////////////////////////////////
    urDdiTable.Platform.pfnGet =
        [](ur_adapter_handle_t *, uint32_t, uint32_t NumEntries,
           ur_platform_handle_t *phPlatforms, uint32_t *pNumPlatforms) {
            if (phPlatforms != nullptr && NumEntries != 1) {
                return UR_RESULT_ERROR_INVALID_SIZE;
            }
            if (pNumPlatforms != nullptr) {
                *pNumPlatforms = 1;
            }
            if (nullptr != phPlatforms) {
                *reinterpret_cast<void **>(phPlatforms) = d_context.platform;
            }
            return UR_RESULT_SUCCESS;
        };

    //////////////////////////////////////////////////////////////////////////
    urDdiTable.Platform.pfnGetApiVersion = [](ur_platform_handle_t,
                                              ur_api_version_t *version) {
        *version = d_context.version;
        return UR_RESULT_SUCCESS;
    };

    //////////////////////////////////////////////////////////////////////////
    urDdiTable.Platform.pfnGetInfo =
        [](ur_platform_handle_t hPlatform, ur_platform_info_t PlatformInfoType,
           size_t Size, void *pPlatformInfo, size_t *pSizeRet) {
            if (!hPlatform) {
                return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
            }

            switch (PlatformInfoType) {
            case UR_PLATFORM_INFO_NAME: {
                const char null_platform_name[] = "UR_PLATFORM_NULL";
                if (pSizeRet) {
                    *pSizeRet = sizeof(null_platform_name);
                }
                if (pPlatformInfo && Size != sizeof(null_platform_name)) {
                    return UR_RESULT_ERROR_INVALID_SIZE;
                }
                if (pPlatformInfo) {
#if defined(_WIN32)
                    strncpy_s(reinterpret_cast<char *>(pPlatformInfo), Size,
                              null_platform_name, sizeof(null_platform_name));
#else
                    strncpy(reinterpret_cast<char *>(pPlatformInfo),
                            null_platform_name, Size);
#endif
                }
            } break;

            default:
                return UR_RESULT_ERROR_INVALID_ENUMERATION;
            }

            return UR_RESULT_SUCCESS;
        };

    //////////////////////////////////////////////////////////////////////////
    urDdiTable.Device.pfnGet =
        [](ur_platform_handle_t hPlatform, ur_device_type_t DevicesType,
           uint32_t NumEntries, ur_device_handle_t *phDevices,
           uint32_t *pNumDevices) {
            (void)DevicesType;
            if (hPlatform == nullptr) {
                return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
            }
            if (UR_DEVICE_TYPE_VPU < DevicesType) {
                return UR_RESULT_ERROR_INVALID_ENUMERATION;
            }
            if (phDevices != nullptr && NumEntries != 1) {
                return UR_RESULT_ERROR_INVALID_SIZE;
            }
            if (pNumDevices != nullptr) {
                *pNumDevices = 1;
            }
            if (nullptr != phDevices) {
                *reinterpret_cast<void **>(phDevices) = d_context.get();
            }
            return UR_RESULT_SUCCESS;
        };

    //////////////////////////////////////////////////////////////////////////
    urDdiTable.Device.pfnGetInfo = [](ur_device_handle_t,
                                      ur_device_info_t infoType,
                                      size_t propSize, void *pDeviceInfo,
                                      size_t *pPropSizeRet) {
        switch (infoType) {
        case UR_DEVICE_INFO_TYPE:
            if (pDeviceInfo && propSize != sizeof(ur_device_type_t)) {
                return UR_RESULT_ERROR_INVALID_SIZE;
            }

            if (pDeviceInfo != nullptr) {
                *reinterpret_cast<ur_device_type_t *>(pDeviceInfo) =
                    UR_DEVICE_TYPE_GPU;
            }
            if (pPropSizeRet != nullptr) {
                *pPropSizeRet = sizeof(ur_device_type_t);
            }
            break;

        case UR_DEVICE_INFO_NAME: {
            char deviceName[] = "Null Device";
            if (pDeviceInfo && propSize < sizeof(deviceName)) {
                return UR_RESULT_ERROR_INVALID_SIZE;
            }
            if (pDeviceInfo != nullptr) {
#if defined(_WIN32)
                strncpy_s(reinterpret_cast<char *>(pDeviceInfo), propSize,
                          deviceName, sizeof(deviceName));
#else
                strncpy(reinterpret_cast<char *>(pDeviceInfo), deviceName,
                        propSize);
#endif
            }
            if (pPropSizeRet != nullptr) {
                *pPropSizeRet = sizeof(deviceName);
            }
        } break;
        case UR_DEVICE_INFO_PLATFORM: {
            if (pDeviceInfo && propSize < sizeof(pDeviceInfo)) {
                return UR_RESULT_ERROR_INVALID_SIZE;
            }
            if (pDeviceInfo != nullptr) {
                *reinterpret_cast<void **>(pDeviceInfo) = d_context.platform;
            }
            if (pPropSizeRet != nullptr) {
                *pPropSizeRet = sizeof(intptr_t);
            }
        } break;
        default:
            return UR_RESULT_ERROR_INVALID_ARGUMENT;
        }
        return UR_RESULT_SUCCESS;
    };

    //////////////////////////////////////////////////////////////////////////
    urDdiTable.USM.pfnHostAlloc = [](ur_context_handle_t, const ur_usm_desc_t *,
                                     ur_usm_pool_handle_t, size_t size,
                                     void **ppMem) {
        if (size == 0) {
            *ppMem = nullptr;
            return UR_RESULT_ERROR_UNSUPPORTED_SIZE;
        }
        *ppMem = malloc(size);
        if (*ppMem == nullptr) {
            return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        return UR_RESULT_SUCCESS;
    };

    //////////////////////////////////////////////////////////////////////////
    urDdiTable.USM.pfnDeviceAlloc =
        [](ur_context_handle_t, ur_device_handle_t, const ur_usm_desc_t *,
           ur_usm_pool_handle_t, size_t size, void **ppMem) {
            if (size == 0) {
                *ppMem = nullptr;
                return UR_RESULT_ERROR_UNSUPPORTED_SIZE;
            }
            *ppMem = malloc(size);
            if (*ppMem == nullptr) {
                return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
            }
            return UR_RESULT_SUCCESS;
        };

    //////////////////////////////////////////////////////////////////////////
    urDdiTable.USM.pfnFree = [](ur_context_handle_t, void *pMem) {
        free(pMem);
        return UR_RESULT_SUCCESS;
    };

    //////////////////////////////////////////////////////////////////////////
    urDdiTable.USM.pfnGetMemAllocInfo =
        [](ur_context_handle_t, const void *pMem, ur_usm_alloc_info_t propName,
           size_t, void *pPropValue, size_t *pPropSizeRet) {
            switch (propName) {
            case UR_USM_ALLOC_INFO_TYPE:
                *reinterpret_cast<ur_usm_type_t *>(pPropValue) =
                    pMem ? UR_USM_TYPE_DEVICE : UR_USM_TYPE_UNKNOWN;
                if (pPropSizeRet != nullptr) {
                    *pPropSizeRet = sizeof(ur_usm_type_t);
                }
                break;
            case UR_USM_ALLOC_INFO_SIZE:
                *reinterpret_cast<size_t *>(pPropValue) = pMem ? SIZE_MAX : 0;
                if (pPropSizeRet != nullptr) {
                    *pPropSizeRet = sizeof(size_t);
                }
                break;
            default:
                pPropValue = nullptr;
                break;
            }
            return UR_RESULT_SUCCESS;
        };
}
} // namespace driver
