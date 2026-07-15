//===---------------- physical_mem.hpp - Level Zero Adapter ---------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "common/ur_ref_count.hpp"

struct ur_physical_mem_handle_t_ : ur_object {
  ur_physical_mem_handle_t_(ze_physical_mem_handle_t ZePhysicalMem,
                            ur_context_handle_t Context,
                            ur_device_handle_t Device, size_t Size,
                            bool EnableIpc)
      : ZePhysicalMem{ZePhysicalMem}, Context{Context}, Device{Device},
        Size{Size}, EnableIpc{EnableIpc} {}

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
