/*
 * Copyright (C) 2020-2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef _ZEX_DRIVER_H
#define _ZEX_DRIVER_H

#if defined(__cplusplus)
#pragma once
#endif

#include <level_zero/ze_api.h>

#if defined(__cplusplus)
extern "C" {
#endif

ZE_APIEXPORT ze_result_t ZE_APICALL zexDriverImportExternalPointer(
    ze_driver_handle_t hDriver, ///< [in] handle of the driver
    void *ptr,                  ///< [in] pointer to be imported to the driver
    size_t size                 ///< [in] size to be imported
);

ZE_APIEXPORT ze_result_t ZE_APICALL zexDriverReleaseImportedPointer(
    ze_driver_handle_t hDriver, ///< [in] handle of the driver
    void *ptr                   ///< [in] pointer to be released from the driver
);

ZE_APIEXPORT ze_result_t ZE_APICALL zexDriverGetHostPointerBaseAddress(
    ze_driver_handle_t hDriver, ///< [in] handle of the driver
    void *ptr,         ///< [in] pointer to be checked if imported to the driver
    void **baseAddress ///< [out] if not null, returns address of the base
                       ///< pointer of the imported pointer
);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZEX_DRIVER_H
