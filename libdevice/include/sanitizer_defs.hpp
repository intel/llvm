//==-- sanitizer_defs.hpp - common macros shared by sanitizers ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "atomic.hpp"
#include "spir_global_var.hpp"
#include "spirv_vars.h"
#include <cstdint>

using uptr = uintptr_t;
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using s8 = int8_t;
using s16 = int16_t;
using s32 = int32_t;
using s64 = int64_t;

enum ADDRESS_SPACE : uint32_t {
  ADDRESS_SPACE_PRIVATE = 0,
  ADDRESS_SPACE_GLOBAL = 1,
  ADDRESS_SPACE_CONSTANT = 2,
  ADDRESS_SPACE_LOCAL = 3,
  ADDRESS_SPACE_GENERIC = 4,
};

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define NORETURN __declspec(noreturn)

#if defined(__SPIR__) || defined(__SPIRV__)

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif // SYCL_EXTERNAL

extern SYCL_EXTERNAL int
__spirv_ocl_printf(const __SYCL_CONSTANT__ char *Format, ...);

extern SYCL_EXTERNAL __SYCL_GLOBAL__ void *
__spirv_GenericCastToPtrExplicit_ToGlobal(void *, int) noexcept;
extern SYCL_EXTERNAL __SYCL_LOCAL__ void *
__spirv_GenericCastToPtrExplicit_ToLocal(void *, int) noexcept;
extern SYCL_EXTERNAL __SYCL_PRIVATE__ void *
__spirv_GenericCastToPtrExplicit_ToPrivate(void *, int) noexcept;

extern SYCL_EXTERNAL __attribute__((convergent)) void
__spirv_ControlBarrier(int32_t Execution, int32_t Memory,
                       int32_t Semantics) noexcept;

extern "C" SYCL_EXTERNAL void __devicelib_exit();

__SYCL_GLOBAL__ void *ToGlobal(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToGlobal(ptr, 5);
}
__SYCL_LOCAL__ void *ToLocal(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToLocal(ptr, 4);
}
__SYCL_PRIVATE__ void *ToPrivate(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToPrivate(ptr, 7);
}

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

#endif // __SPIR__ || __SPIRV__
