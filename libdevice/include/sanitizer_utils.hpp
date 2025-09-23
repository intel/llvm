//==------------- sanitizer_utils.hpp - utils for sanitizers ---------------==//
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

inline size_t WorkGroupLinearId() {
  return __spirv_BuiltInWorkgroupId(0) * __spirv_BuiltInNumWorkgroups(1) *
             __spirv_BuiltInNumWorkgroups(2) +
         __spirv_BuiltInWorkgroupId(1) * __spirv_BuiltInNumWorkgroups(2) +
         __spirv_BuiltInWorkgroupId(2);
}

static inline size_t LocalLinearId() {
  return __spirv_BuiltInLocalInvocationId(0) * __spirv_BuiltInWorkgroupSize(1) *
             __spirv_BuiltInWorkgroupSize(2) +
         __spirv_BuiltInLocalInvocationId(1) * __spirv_BuiltInWorkgroupSize(2) +
         __spirv_BuiltInLocalInvocationId(2);
}

// For GPU device, each sub group is a hardware thread
inline size_t SubGroupLinearId() {
  return WorkGroupLinearId() * __spirv_BuiltInNumSubgroups() +
         __spirv_BuiltInSubgroupId();
}

inline void SubGroupBarrier() {
  __spirv_ControlBarrier(__spv::Scope::Subgroup, __spv::Scope::Subgroup,
                         __spv::MemorySemanticsMask::SequentiallyConsistent |
                             __spv::MemorySemanticsMask::CrossWorkgroupMemory |
                             __spv::MemorySemanticsMask::WorkgroupMemory);
}

inline __SYCL_GLOBAL__ void *ToGlobal(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToGlobal(ptr, 5);
}
inline __SYCL_LOCAL__ void *ToLocal(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToLocal(ptr, 4);
}
inline __SYCL_PRIVATE__ void *ToPrivate(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToPrivate(ptr, 7);
}

template <typename T> T Memset(T ptr, int value, size_t size) {
  for (size_t i = 0; i < size; i++) {
    ptr[i] = value;
  }
  return ptr;
}

template <typename DstT, typename SrcT>
DstT Memcpy(DstT dst, SrcT src, size_t size) {
  for (size_t i = 0; i < size; i++) {
    dst[i] = src[i];
  }
  return dst;
}

template <typename DstT, typename SrcT>
DstT Memmove(DstT dst, SrcT src, size_t size) {
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
