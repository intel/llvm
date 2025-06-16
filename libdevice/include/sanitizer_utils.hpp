//==------------------ group_utils.hpp - utils for group -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "sanitizer_defs.hpp"
#include "spirv_vars.h"

#if defined(__SPIR__) || defined(__SPIRV__)

size_t WorkGroupLinearId() {
  return __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
             __spirv_BuiltInNumWorkgroups.z +
         __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
         __spirv_BuiltInWorkgroupId.z;
}

// For GPU device, each sub group is a hardware thread
size_t SubGroupLinearId() {
  return __spirv_BuiltInGlobalLinearId / __spirv_BuiltInSubgroupSize;
}

void SubGroupBarrier() {
  __spirv_ControlBarrier(__spv::Scope::Subgroup, __spv::Scope::Subgroup,
                         __spv::MemorySemanticsMask::SequentiallyConsistent |
                             __spv::MemorySemanticsMask::CrossWorkgroupMemory |
                             __spv::MemorySemanticsMask::WorkgroupMemory);
}

__SYCL_GLOBAL__ void *ToGlobal(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToGlobal(ptr, 5);
}
__SYCL_LOCAL__ void *ToLocal(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToLocal(ptr, 4);
}
__SYCL_PRIVATE__ void *ToPrivate(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToPrivate(ptr, 7);
}

template <typename T> SYCL_EXTERNAL T Memset(T ptr, int value, size_t size) {
  for (size_t i = 0; i < size; i++) {
    ptr[i] = value;
  }
  return ptr;
}

template <typename DstT, typename SrcT>
SYCL_EXTERNAL DstT Memcpy(DstT dst, SrcT src, size_t size) {
  for (size_t i = 0; i < size; i++) {
    dst[i] = src[i];
  }
  return dst;
}

template <typename DstT, typename SrcT>
SYCL_EXTERNAL DstT Memmove(DstT dst, SrcT src, size_t size) {
  if ((uptr)dst < (uptr)src) {
    for (size_t i = 0; i < size; i++) {
      dst[i] = src[i];
    }
  } else {
    for (size_t i = size; i > 0; i--) {
      dst[i - 1] = src[i - 1];
    }
  }
  return dst;
}

#endif // __SPIR__ || __SPIRV__
