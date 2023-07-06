//===---------------- physical_mem.hpp - Level Zero Adapter ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"

struct ur_physical_mem_handle_t_ : _ur_object {
  ur_physical_mem_handle_t_(ze_physical_mem_handle_t ZePhysicalMem,
                            ur_context_handle_t Context)
      : ZePhysicalMem{ZePhysicalMem}, Context{Context} {}

  // Level Zero physical memory handle.
  ze_physical_mem_handle_t ZePhysicalMem;

  // Keeps the PI context of this memory handle.
  ur_context_handle_t Context;
};
