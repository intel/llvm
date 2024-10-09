//==------- group_helper.hpp - utils related to work-group operations-------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//
#pragma once
#include "spirv_decls.hpp"
#include "spirv_vars.h"
#include <cstdint>
#if defined(__SPIR__) || defined(__SPIRV__)

static inline size_t __get_wg_local_range() {
  return __spirv_BuiltInWorkgroupSize.x * __spirv_BuiltInWorkgroupSize.y *
         __spirv_BuiltInWorkgroupSize.z;
}

static inline size_t __get_wg_local_linear_id() {
  return (__spirv_BuiltInLocalInvocationId.x * __spirv_BuiltInWorkgroupSize.y *
          __spirv_BuiltInWorkgroupSize.z) +
         (__spirv_BuiltInLocalInvocationId.y * __spirv_BuiltInWorkgroupSize.z) +
         __spirv_BuiltInLocalInvocationId.z;
}

static inline void group_barrier() {
  __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                         __spv::MemorySemanticsMask::SequentiallyConsistent |
                             __spv::MemorySemanticsMask::SubgroupMemory |
                             __spv::MemorySemanticsMask::WorkgroupMemory |
                             __spv::MemorySemanticsMask::CrossWorkgroupMemory);
}

static inline uint64_t group_broadcast(uint64_t x) {
  return __spirv_GroupBroadcast(__spv::Scope::Flag::Workgroup, x, 0);
}
#endif // __SPIR__ || __SPIRV__
