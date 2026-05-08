/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_allocator.hpp
 *
 */

#pragma once

#include "sanitizer_common.hpp"

namespace ur_sanitizer_layer {

enum class AllocType {
  UNKNOWN,
  DEVICE_USM,
  SHARED_USM,
  HOST_USM,
  MEM_BUFFER,
  DEVICE_GLOBAL,
  EXPORTABLE_MEM
};

struct AllocMemoryParams {
  // For normal USM allocations
  const ur_usm_desc_t *USMDesc = nullptr;
  ur_usm_pool_handle_t Pool = {};
  // For exportable memory allocations
  ur_exp_external_mem_type_t HandleTypeToExport = {};
  size_t Alignment = 0;

  static AllocMemoryParams forUSM(const ur_usm_desc_t *USMDesc,
                                  ur_usm_pool_handle_t Pool) {
    AllocMemoryParams Params;
    Params.USMDesc = USMDesc;
    Params.Pool = Pool;
    return Params;
  }

  static AllocMemoryParams
  forExportableMem(size_t Alignment,
                   ur_exp_external_mem_type_t HandleTypeToExport) {
    AllocMemoryParams Params;
    Params.Alignment = Alignment;
    Params.HandleTypeToExport = HandleTypeToExport;
    return Params;
  }
};

inline const char *ToString(AllocType Type) {
  switch (Type) {
  case AllocType::DEVICE_USM:
    return "Device USM";
  case AllocType::HOST_USM:
    return "Host USM";
  case AllocType::SHARED_USM:
    return "Shared USM";
  case AllocType::MEM_BUFFER:
    return "Memory Buffer";
  case AllocType::DEVICE_GLOBAL:
    return "Device Global";
  case AllocType::EXPORTABLE_MEM:
    return "Exportable Memory";
  default:
    return "Unknown Type";
  }
}

// Allocating USM with validation, so that we can ensure the allocated addresses
// satisfy the assumption we made for shadow memory
ur_result_t SafeAllocate(ur_context_handle_t Context, ur_device_handle_t Device,
                         uptr Size, const ur_usm_desc_t *Properties,
                         ur_usm_pool_handle_t Pool, AllocType Type,
                         void **Allocated);

} // namespace ur_sanitizer_layer
