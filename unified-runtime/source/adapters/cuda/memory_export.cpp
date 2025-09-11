//===--------- memory_export.cpp - CUDA Adapter ---------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common/logger/ur_logger.hpp"
#include "ur_api.h"

UR_APIEXPORT ur_result_t UR_APICALL urMemoryExportAllocExportableMemoryExp(
    ur_context_handle_t /*hContext*/, ur_device_handle_t /*hDevice*/,
    size_t /*aligment*/, size_t /*size*/,
    ur_exp_external_mem_type_t /*handleTypeToExport*/, void ** /*ppMem*/) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemoryExportFreeExportableMemoryExp(
    ur_context_handle_t /*hContext*/, ur_device_handle_t /*hDevice*/,
    void * /*pMem*/) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemoryExportExportMemoryHandleExp(
    ur_context_handle_t /*hContext*/, ur_device_handle_t /*hDevice*/,
    ur_exp_external_mem_type_t /*handleTypeToExport*/, void * /*pMem*/,
    void * /*pMemHandleRet*/) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
