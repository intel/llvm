/*
 * Copyright (C) 2022-2026 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef _ZEX_CMDLIST_H
#define _ZEX_CMDLIST_H
#if defined(__cplusplus)
#pragma once
#endif

#include <level_zero/ze_api.h>

#include "zex_common.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(_WIN32)
#if !defined(ZE_CALLBACK)
#define ZE_CALLBACK __stdcall
#endif
#else
#if !defined(ZE_CALLBACK)
#define ZE_CALLBACK
#endif
#endif

ze_result_t ZE_APICALL zexCommandListAppendWaitOnMemory(
    zex_command_list_handle_t hCommandList, zex_wait_on_mem_desc_t *desc,
    void *ptr, uint32_t data, zex_event_handle_t hSignalEvent);

ze_result_t ZE_APICALL zexCommandListAppendWaitOnMemory64(
    zex_command_list_handle_t hCommandList, zex_wait_on_mem_desc_t *desc,
    void *ptr, uint64_t data, zex_event_handle_t hSignalEvent);

ze_result_t ZE_APICALL zexCommandListAppendWriteToMemory(
    zex_command_list_handle_t hCommandList, zex_write_to_mem_desc_t *desc,
    void *ptr, uint64_t data);

typedef void(ZE_CALLBACK *ze_host_function_callback_t)(void *pUserData);

ze_result_t ZE_APICALL zeCommandListAppendHostFunction(
    ze_command_list_handle_t hCommandList, ///< [in] handle of the command list
    ze_host_function_callback_t pHostFunction, ///< [in] host function to call
    void *
        pUserData, ///< [in] user specific data that would be passed to function
    void *pNext,   ///< [in][optional] extensions
    ze_event_handle_t hSignalEvent, ///< [in][optional] handle of the event to
                                    ///< signal on completion
    uint32_t numWaitEvents, ///< [in][optional] number of events to wait on
                            ///< before launching
    ze_event_handle_t
        *phWaitEvents); ///< [in][optional][range(0, numWaitEvents)] handle of
                        ///< the events to wait on before launching

ze_result_t ZE_APICALL zexCommandListAppendMemoryCopyWithParameters(
    ze_command_list_handle_t hCommandList, ///< [in] handle of the command list
    void *dstptr,                          ///< [in] destination pointer
    const void *srcptr,                    ///< [in] source pointer
    size_t size,                           ///< [in] size in bytes to copy
    const void *pNext,      ///< [in][optional] additional copy parameters
    uint32_t numWaitEvents, ///< [in][optional] number of events to wait on
                            ///< before launching
    ze_event_handle_t
        *phWaitEvents, ///< [in][optional][range(0, numWaitEvents)] handle of
                       ///< the events to wait on before launching
    ze_event_handle_t hSignalEvent); ///< [in][optional] handle of the event to
                                     ///< signal on completion

ze_result_t ZE_APICALL zexCommandListAppendMemoryFillWithParameters(
    ze_command_list_handle_t hCommandList, ///< [in] handle of the command list
    void *ptr,           ///< [in] pointer to memory to initialize
    const void *pattern, ///< [in] pointer to value to initialize memory to
    size_t patternSize,  ///< [in] size in bytes of the value to initialize
                         ///< memory to
    size_t size,         ///< [in] size in bytes to initialize
    const void *pNext,   ///< [in][optional] additional copy parameters
    ze_event_handle_t
        hEvent, ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents, ///< [in][optional] number of events to wait on
                            ///< before launching
    ze_event_handle_t
        *phWaitEvents); ///< [in][optional][range(0, numWaitEvents)] handle of
                        ///< the events to wait on before launching

ze_result_t ZE_APICALL zexCommandListAppendCustomOperation(
    ze_command_list_handle_t hCommandList, ///< [in] handle of the command list
    const void *pNext, ///< [in] operation parameters, there may be only a
                       ///< single operation defined
    ze_event_handle_t
        hEvent, ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents, ///< [in][optional] number of events to wait on
                            ///< before launching
    ze_event_handle_t
        *phWaitEvents); ///< [in][optional][range(0, numWaitEvents)] handle of
                        ///< the events to wait on before launching

typedef void (*zex_command_list_cleanup_callback_fn_t)(void *pUserData);

ze_result_t ZE_APICALL zexCommandListSetCleanupCallback(
    ze_command_list_handle_t hCommandList, ///< [in] handle of the command list
    zex_command_list_cleanup_callback_fn_t
        pfnCallback, ///< [in] host function to call
    void *
        pUserData, ///< [in] user specific data that would be passed to function
    const void *pNext); ///< [in][optional] must be null or a pointer to an
                        ///< extension-specific structure

ze_result_t ZE_APICALL zexCommandListVerifyMemory(
    ze_command_list_handle_t hCommandList, const void *allocationPtr,
    const void *expectedData, size_t sizeOfComparison,
    zex_verify_memory_compare_type_t comparisonMode);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZEX_CMDLIST_H
