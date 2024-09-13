//===--------- memory_helpers.cpp - Level Zero Adapter -------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory_helpers.hpp"
#include "../common.hpp"

ze_memory_type_t getMemoryType(ze_context_handle_t hContext, void *ptr) {
  // TODO: use UMF once
  // https://github.com/oneapi-src/unified-memory-framework/issues/687 is
  // implemented
  ZeStruct<ze_memory_allocation_properties_t> zeMemoryAllocationProperties;
  ZE2UR_CALL_THROWS(zeMemGetAllocProperties,
                    (hContext, ptr, &zeMemoryAllocationProperties, nullptr));
  return zeMemoryAllocationProperties.type;
}

bool maybeImportUSM(ze_driver_handle_t hTranslatedDriver,
                    ze_context_handle_t hContext, void *ptr, size_t size) {
  if (ZeUSMImport.Enabled && ptr != nullptr &&
      getMemoryType(hContext, ptr) == ZE_MEMORY_TYPE_UNKNOWN) {
    // Promote the host ptr to USM host memory
    ZeUSMImport.doZeUSMImport(hTranslatedDriver, ptr, size);
    return true;
  }
  return false;
}
