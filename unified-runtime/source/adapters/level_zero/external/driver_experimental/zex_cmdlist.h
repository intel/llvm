/*
 * Copyright (C) 2022-2025 Intel Corporation
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

ZE_APIEXPORT ze_result_t ZE_APICALL zexCommandListAppendWaitOnMemory(
    zex_command_list_handle_t hCommandList, zex_wait_on_mem_desc_t *desc,
    void *ptr, uint32_t data, zex_event_handle_t hSignalEvent);

ZE_APIEXPORT ze_result_t ZE_APICALL zexCommandListAppendWaitOnMemory64(
    zex_command_list_handle_t hCommandList, zex_wait_on_mem_desc_t *desc,
    void *ptr, uint64_t data, zex_event_handle_t hSignalEvent);

ZE_APIEXPORT ze_result_t ZE_APICALL zexCommandListAppendWriteToMemory(
    zex_command_list_handle_t hCommandList, zex_write_to_mem_desc_t *desc,
    void *ptr, uint64_t data);

ZE_APIEXPORT ze_result_t ZE_APICALL zexCommandListAppendHostFunction(
    ze_command_list_handle_t hCommandList, void *pHostFunction, void *pUserData,
    void *pNext, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents,
    ze_event_handle_t *phWaitEvents);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZEX_CMDLIST_H
