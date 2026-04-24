//===--------- adapter.hpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ur_api.h>

namespace ur::level_zero::common {

ur_result_t urAdapterGet(uint32_t NumEntries, ur_adapter_handle_t *phAdapters,
                         uint32_t *pNumAdapters);
ur_result_t urAdapterRelease(ur_adapter_handle_t hAdapter);
ur_result_t urAdapterRetain(ur_adapter_handle_t hAdapter);
ur_result_t urAdapterGetLastError(ur_adapter_handle_t hAdapter,
                                  const char **ppMessage, int32_t *pError);
ur_result_t urAdapterGetInfo(ur_adapter_handle_t hAdapter,
                             ur_adapter_info_t propName, size_t propSize,
                             void *pPropValue, size_t *pPropSizeRet);
ur_result_t urAdapterSetLoggerCallback(ur_adapter_handle_t hAdapter,
                                       ur_logger_callback_t pfnLoggerCallback,
                                       void *pUserData,
                                       ur_logger_level_t level);
ur_result_t urAdapterSetLoggerCallbackLevel(ur_adapter_handle_t hAdapter,
                                            ur_logger_level_t level);

} // namespace ur::level_zero::common
