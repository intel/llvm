/*
 * Copyright (C) 2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#pragma once

#include <level_zero/ze_api.h>

typedef ze_result_t(ZE_APICALL *ze_pfnCommandListGetNextCommandIdExpCb_t)(
    ze_command_list_handle_t hCommandList,
    const ze_mutable_command_id_exp_desc_t *desc, uint64_t *pCommandId);

typedef ze_result_t(ZE_APICALL *ze_pfnCommandListUpdateMutableCommandsExpCb_t)(
    ze_command_list_handle_t hCommandList,
    const ze_mutable_commands_exp_desc_t *desc);

typedef ze_result_t(
    ZE_APICALL *ze_pfnCommandListUpdateMutableCommandSignalEventExpCb_t)(
    ze_command_list_handle_t hCommandList, uint64_t commandId,
    ze_event_handle_t hSignalEvent);

typedef ze_result_t(
    ZE_APICALL *ze_pfnCommandListUpdateMutableCommandWaitEventsExpCb_t)(
    ze_command_list_handle_t hCommandList, uint64_t commandId,
    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);

typedef ze_result_t(
    ZE_APICALL *ze_pfnCommandListGetNextCommandIdWithKernelsExpCb_t)(
    ze_command_list_handle_t hCommandList,
    const ze_mutable_command_id_exp_desc_t *desc, uint32_t numKernels,
    ze_kernel_handle_t *phKernels, uint64_t *pCommandId);

typedef ze_result_t(
    ZE_APICALL *ze_pfnCommandListUpdateMutableCommandKernelsExpCb_t)(
    ze_command_list_handle_t hCommandList, uint32_t numKernels,
    uint64_t *pCommandId, ze_kernel_handle_t *phKernels);
