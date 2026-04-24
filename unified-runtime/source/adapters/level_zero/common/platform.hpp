//===--------- platform.hpp - Level Zero Adapter -------------------------===//
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

ur_result_t urPlatformGet(ur_adapter_handle_t hAdapter, uint32_t NumEntries,
                          ur_platform_handle_t *phPlatforms,
                          uint32_t *pNumPlatforms);
ur_result_t urPlatformGetInfo(ur_platform_handle_t hPlatform,
                              ur_platform_info_t propName, size_t propSize,
                              void *pPropValue, size_t *pPropSizeRet);
ur_result_t urPlatformGetApiVersion(ur_platform_handle_t hPlatform,
                                    ur_api_version_t *pVersion);
ur_result_t urPlatformGetNativeHandle(ur_platform_handle_t hPlatform,
                                      ur_native_handle_t *phNativePlatform);
ur_result_t urPlatformCreateWithNativeHandle(
    ur_native_handle_t hNativePlatform, ur_adapter_handle_t hAdapter,
    const ur_platform_native_properties_t *pProperties,
    ur_platform_handle_t *phPlatform);
ur_result_t urPlatformGetBackendOption(ur_platform_handle_t hPlatform,
                                       const char *pFrontendOption,
                                       const char **ppPlatformOption);

} // namespace ur::level_zero::common
