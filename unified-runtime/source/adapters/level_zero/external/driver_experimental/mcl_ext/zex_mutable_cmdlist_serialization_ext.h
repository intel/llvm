/*
 * Copyright (C) 2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#pragma once

#include <level_zero/ze_api.h>

#if defined(__cplusplus)
extern "C" {
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve native MCL binary from cmdlist.
ze_result_t ZE_APICALL zexCommandListGetNativeBinary(
    ze_command_list_handle_t
        hCommandList, ///< [in] handle of mutable command list
    void *pBinary,    ///< [in,out][optional] byte pointer to native MCL binary
    size_t *pBinarySize, ///< [in,out] size of native MCL binary in bytes
    const void *pModule, ///< [in] byte pointer to module containing kernels'
                         ///< used in cmdlist
    size_t moduleSize    ///< [in] size in bytes of module containing kernels's
                         ///< used in cmdlist
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Load command list from native MCL binary.
ze_result_t ZE_APICALL zexCommandListLoadNativeBinary(
    ze_command_list_handle_t
        hCommandList,    ///< [in] handle of mutable command list
    const void *pBinary, ///< [in] byte pointer to native MCL binary
    size_t binarySize    ///< [in] size of native MCL binary in bytes
);

typedef ze_result_t(ZE_APICALL *zex_pfnCommandListGetNativeBinaryCb_t)(
    ze_command_list_handle_t hCommandList, void *pBinary, size_t *pBinarySize,
    const void *pModule, size_t moduleSize);

typedef ze_result_t(ZE_APICALL *zex_pfnCommandListLoadNativeBinaryCb_t)(
    ze_command_list_handle_t hCommandList, const void *pBinary,
    size_t binarySize);

#if defined(__cplusplus)
} // extern "C"
#endif
