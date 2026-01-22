//===---------------- physical_mem.hpp - Level Zero Adapter ---------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../common.hpp"
#include "common/ur_ref_count.hpp"
#include "context_interface.hpp"

namespace ur::level_zero::common {

struct ur_physical_mem_handle_t_ : ur_shared_handle_base_t {
  ur_physical_mem_handle_t_(ze_physical_mem_handle_t ZePhysicalMem,
                            ur_context_handle_t Context)
      : ZePhysicalMem{ZePhysicalMem}, Context{Context} {
    // Auto-populate ddi_table from the owning context so the loader's
    // intercept layer can dispatch this handle back to the correct adapter.
    ddi_table = ddiTableOf(Context);
  }

  // Level Zero physical memory handle.
  ze_physical_mem_handle_t ZePhysicalMem;

  // Keeps the PI context of this memory handle.
  ur_context_handle_t Context;

  ur::RefCount RefCount;
};

// Cast from opaque UR physical mem handle to the common concrete type.
inline ur_physical_mem_handle_t_ *cast(::ur_physical_mem_handle_t h) {
  return reinterpret_cast<ur_physical_mem_handle_t_ *>(h);
}

ur_result_t urPhysicalMemCreate(ur_context_handle_t hContext,
                                ur_device_handle_t hDevice, size_t size,
                                const ur_physical_mem_properties_t *pProperties,
                                ur_physical_mem_handle_t *phPhysicalMem);
ur_result_t urPhysicalMemRetain(ur_physical_mem_handle_t hPhysicalMem);
ur_result_t urPhysicalMemRelease(ur_physical_mem_handle_t hPhysicalMem);
ur_result_t urPhysicalMemGetInfo(ur_physical_mem_handle_t hPhysicalMem,
                                 ur_physical_mem_info_t propName,
                                 size_t propSize, void *pPropValue,
                                 size_t *pPropSizeRet);

} // namespace ur::level_zero::common
