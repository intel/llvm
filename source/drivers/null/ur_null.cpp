/*
 *
 * Copyright (C) 2019-2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_null.cpp
 *
 */
#include "ur_null.h"

namespace driver
{
    //////////////////////////////////////////////////////////////////////////
    context_t d_context;

    //////////////////////////////////////////////////////////////////////////
    context_t::context_t()
    {
        //////////////////////////////////////////////////////////////////////////
        urDdiTable.Platform.pfnGet = [](
            uint32_t NumEntries,
            ur_platform_handle_t* phPlatforms,
            uint32_t* pNumPlatforms)
        {
            if (phPlatforms != nullptr && NumEntries != 1) return UR_RESULT_ERROR_INVALID_SIZE;
            if (pNumPlatforms != nullptr) *pNumPlatforms = 1;
            if( nullptr != phPlatforms ) *reinterpret_cast<void**>( phPlatforms ) = d_context.get();
            return UR_RESULT_SUCCESS;
        };

        //////////////////////////////////////////////////////////////////////////
        urDdiTable.Device.pfnGet = [](
            ur_platform_handle_t hPlatform,
            ur_device_type_t DevicesType,
            uint32_t NumEntries,
            ur_device_handle_t* phDevices,
            uint32_t* pNumDevices )
        {
            (void)DevicesType;
            if (phDevices != nullptr && NumEntries != 1) return UR_RESULT_ERROR_INVALID_SIZE;
            if (pNumDevices != nullptr) *pNumDevices = 1;
            if( nullptr != phDevices ) *reinterpret_cast<void**>( phDevices ) = d_context.get();
            return UR_RESULT_SUCCESS;
        };

        //////////////////////////////////////////////////////////////////////////
        urDdiTable.Device.pfnGetInfo = [](
            ur_device_handle_t hDevice,
            ur_device_info_t infoType,
            size_t propSize,
            void* pDeviceInfo,
            size_t* pPropSizeRet)
        {
            switch (infoType) {
                case UR_DEVICE_INFO_TYPE:
                    if (propSize != sizeof(ur_device_type_t)) return UR_RESULT_ERROR_INVALID_SIZE;

                    if (pDeviceInfo != nullptr) {
                        *reinterpret_cast<ur_device_type_t *>(pDeviceInfo) = UR_DEVICE_TYPE_GPU;
                    }
                    if (pPropSizeRet != nullptr) {
                        *pPropSizeRet = sizeof(ur_device_type_t);
                    }
                    break;

                case UR_DEVICE_INFO_NAME:
                    if (pDeviceInfo != nullptr) {
#if defined(_WIN32)
                        strncpy_s( reinterpret_cast<char *>(pDeviceInfo), "Null Device", propSize );
#else
                        strncpy( reinterpret_cast<char *>(pDeviceInfo), "Null Device", propSize );
#endif
                    }
                    if (pPropSizeRet != nullptr) {
                        *pPropSizeRet = sizeof(pDeviceInfo);
                    }
                    break;

                default:
                    return UR_RESULT_ERROR_INVALID_ARGUMENT;
            }
            return UR_RESULT_SUCCESS;

        };
    }
} // namespace driver
