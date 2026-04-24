//===---------------- virtual_mem.hpp - Level Zero Adapter ----------------===//
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

ur_result_t urVirtualMemGranularityGetInfo(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    size_t allocationSize, ur_virtual_mem_granularity_info_t propName,
    size_t propSize, void *pPropValue, size_t *pPropSizeRet);
ur_result_t urVirtualMemReserve(ur_context_handle_t hContext,
                                const void *pStart, size_t size,
                                void **ppStart);
ur_result_t urVirtualMemFree(ur_context_handle_t hContext, const void *pStart,
                             size_t size);
ur_result_t urVirtualMemMap(ur_context_handle_t hContext, const void *pStart,
                            size_t size, ur_physical_mem_handle_t hPhysicalMem,
                            size_t offset, ur_virtual_mem_access_flags_t flags);
ur_result_t urVirtualMemUnmap(ur_context_handle_t hContext, const void *pStart,
                              size_t size);
ur_result_t urVirtualMemSetAccess(ur_context_handle_t hContext,
                                  const void *pStart, size_t size,
                                  ur_virtual_mem_access_flags_t flags);
ur_result_t urVirtualMemGetInfo(ur_context_handle_t hContext,
                                const void *pStart, size_t size,
                                ur_virtual_mem_info_t propName, size_t propSize,
                                void *pPropValue, size_t *pPropSizeRet);

} // namespace ur::level_zero::common
