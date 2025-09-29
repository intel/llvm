//===--------- memory_export.cpp - Level Zero Adapter ---------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#ifdef UR_ADAPTER_LEVEL_ZERO_V2
#include "v2/context.hpp"
#else
#include "context.hpp"
#endif
#include "ur_api.h"

namespace ur::level_zero {

ur_result_t urMemoryExportAllocExportableMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t alignment,
    size_t size, ur_exp_external_mem_type_t handleTypeToExport, void **ppMem) {

  UR_ASSERT(handleTypeToExport == UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD ||
                handleTypeToExport == UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT,
            UR_RESULT_ERROR_INVALID_ENUMERATION);

  ze_external_memory_export_desc_t MemExportDesc{};
  MemExportDesc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC;

  switch (handleTypeToExport) {
  case UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD:
    MemExportDesc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD;
    break;
  case UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT:
    MemExportDesc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;
    break;
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  ze_device_mem_alloc_desc_t MemAllocDesc{};
  MemAllocDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  MemAllocDesc.pNext = &MemExportDesc;

  ZE2UR_CALL(zeMemAllocDevice, (hContext->getZeHandle(), &MemAllocDesc, size,
                                alignment, hDevice->ZeDevice, ppMem));

  return UR_RESULT_SUCCESS;
}

ur_result_t urMemoryExportFreeExportableMemoryExp(
    ur_context_handle_t hContext, [[maybe_unused]] ur_device_handle_t hDevice,
    void *pMem) {
  ZE2UR_CALL(zeMemFree, (hContext->getZeHandle(), pMem));
  return UR_RESULT_SUCCESS;
}

ur_result_t urMemoryExportExportMemoryHandleExp(
    ur_context_handle_t hContext, [[maybe_unused]] ur_device_handle_t hDevice,
    ur_exp_external_mem_type_t handleTypeToExport, void *pMem,
    void *pMemHandleRet) {

  ze_memory_allocation_properties_t MemAllocProps{};
  MemAllocProps.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;

  ze_external_memory_export_fd_t MemExportFD{};
  ze_external_memory_export_win32_handle_t MemExportWin32{};

  switch (handleTypeToExport) {
  case UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD: {
    MemExportFD.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD;
    MemExportFD.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD;
    MemAllocProps.pNext = &MemExportFD;
    ZE2UR_CALL(zeMemGetAllocProperties,
               (hContext->getZeHandle(), pMem, &MemAllocProps, nullptr));
    int *pMemHandleRetIntPtr = static_cast<int *>(pMemHandleRet);
    *pMemHandleRetIntPtr = MemExportFD.fd;
    break;
  }
  case UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT: {
    MemExportWin32.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_WIN32;
    MemExportWin32.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;
    MemAllocProps.pNext = &MemExportWin32;
    ZE2UR_CALL(zeMemGetAllocProperties,
               (hContext->getZeHandle(), pMem, &MemAllocProps, nullptr));
    void **ppMemHandleRet = static_cast<void **>(&pMemHandleRet);
    *ppMemHandleRet = MemExportWin32.handle;
    break;
  }
  default: {
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
  }

  return UR_RESULT_SUCCESS;
}

} // namespace ur::level_zero
