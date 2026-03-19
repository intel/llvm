/*
 * Copyright (C) 2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#pragma once

#include "level_zero/driver_experimental/mcl_ext/zex_mutable_cmdlist_variable_ext.h"
#include <level_zero/ze_api.h>

#if defined(__cplusplus)
extern "C" {
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Sets kernel group size to provided variable.
ze_result_t ZE_APICALL zexKernelSetVariableGroupSize(
    ze_kernel_handle_t hKernel,              ///< [in] handle of kernel
    zex_variable_handle_t hGroupSizeVariable ///< [in] handle of variable
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Appends kernel launch to command list with variable group size.
ze_result_t ZE_APICALL zexCommandListAppendVariableLaunchKernel(
    ze_command_list_handle_t
        hCommandList,           ///< [in] hnadle of mutable command list
    ze_kernel_handle_t hKernel, ///< [in] handle of kernel
    zex_variable_handle_t hGroupCountVariable, ///< [in] handle of variable
    ze_event_handle_t hSignalEvent,            ///< [in] handle to signal event
    uint32_t numWaitEvents,                    ///< [in] num events to wait for
    ze_event_handle_t *phWaitEvents); ///< [in] array of events to wait for

typedef ze_result_t(ZE_APICALL *zex_pfnKernelSetVariableGroupSizeCb_t)(
    ze_kernel_handle_t hKernel, zex_variable_handle_t hGroupSizeVariable);

typedef ze_result_t(
    ZE_APICALL *zex_pfnCommandListAppendVariableLaunchKernelCb_t)(
    ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel,
    zex_variable_handle_t hGroupCountVariable, ze_event_handle_t hSignalEvent,
    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);

#if defined(__cplusplus)
} // extern "C"
#endif
