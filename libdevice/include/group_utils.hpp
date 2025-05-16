//==------------------ group_utils.hpp - utils for group -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "spirv_vars.h"

#if defined(__SPIR__) || defined(__SPIRV__)

static inline size_t WorkGroupLinearId() {
  return __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
             __spirv_BuiltInNumWorkgroups.z +
         __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
         __spirv_BuiltInWorkgroupId.z;
}

// For GPU device, each sub group is a hardware thread
static inline size_t SubGroupLinearId() {
  return __spirv_BuiltInGlobalLinearId / __spirv_BuiltInSubgroupSize;
}

static inline void SubGroupBarrier() {
  __spirv_ControlBarrier(__spv::Scope::Subgroup, __spv::Scope::Subgroup,
                         __spv::MemorySemanticsMask::SequentiallyConsistent |
                             __spv::MemorySemanticsMask::CrossWorkgroupMemory |
                             __spv::MemorySemanticsMask::WorkgroupMemory);
}

#endif // __SPIR__ || __SPIRV__
