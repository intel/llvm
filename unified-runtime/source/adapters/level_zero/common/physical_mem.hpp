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

#include "api.hpp"
#include "common/ur_ref_count.hpp"
#include "interfaces.hpp"

namespace ur::level_zero {

struct ur_physical_mem_handle_t_ : ur::level_zero::ur_object_t {
  ur_physical_mem_handle_t_(ze_physical_mem_handle_t ZePhysicalMem,
                            ur_context_handle_t Context,
                            ur_device_handle_t Device, size_t Size,
                            bool EnableIpc)
      : ZePhysicalMem{ZePhysicalMem}, Context{Context}, Device{Device},
        Size{Size}, EnableIpc{EnableIpc} {
    // Populate ddi_table from the owning context.
    ddi_table = ddiTableOf(Context);
  }

  // Level Zero physical memory handle.
  ze_physical_mem_handle_t ZePhysicalMem;

  // Keeps the PI context of this memory handle.
  ur_context_handle_t Context;

  // Device this physical memory was allocated on.
  ur_device_handle_t Device;

  // Size in bytes of this physical memory allocation.
  size_t Size;

  // Whether this allocation was created with IPC export enabled.
  bool EnableIpc;

  ur::RefCount RefCount;
};

} // namespace ur::level_zero
