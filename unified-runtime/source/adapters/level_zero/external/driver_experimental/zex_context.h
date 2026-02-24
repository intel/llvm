/*
 * Copyright (C) 2024-2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef _ZEX_CONTEXT_H
#define _ZEX_CONTEXT_H
#if defined(__cplusplus)
#pragma once
#endif

#include "level_zero/ze_intel_gpu.h"
#include <level_zero/ze_api.h>

#include "zex_common.h"

#if defined(__cplusplus)
extern "C" {
#endif

ZE_APIEXPORT ze_result_t ZE_APICALL zeIntelMediaCommunicationCreate(
    ze_context_handle_t hContext, ze_device_handle_t hDevice,
    ze_intel_media_communication_desc_t *desc,
    ze_intel_media_doorbell_handle_desc_t *phDoorbell);
ZE_APIEXPORT ze_result_t ZE_APICALL zeIntelMediaCommunicationDestroy(
    ze_context_handle_t hContext, ze_device_handle_t hDevice,
    ze_intel_media_doorbell_handle_desc_t *phDoorbell);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZEX_CONTEXT_H
