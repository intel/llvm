//===--------- memory_export.hpp - Level Zero Adapter ---------------------===//
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

ur_result_t urMemoryExportAllocExportableMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t alignment,
    size_t size, ur_exp_external_mem_type_t handleTypeToExport, void **ppMem);
ur_result_t urMemoryExportFreeExportableMemoryExp(ur_context_handle_t hContext,
                                                  ur_device_handle_t hDevice,
                                                  void *pMem);
ur_result_t urMemoryExportExportMemoryHandleExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_mem_type_t handleTypeToExport, void *pMem,
    void *pMemHandleRet);

} // namespace ur::level_zero::common
