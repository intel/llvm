//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_LIBSPIRV_ATOMIC_ATOMIC_HELPER_H__
#define __CLC_LIBSPIRV_ATOMIC_ATOMIC_HELPER_H__

#include <clc/clcfunc.h>
#include <libspirv/spirv_types.h>

static _CLC_INLINE int __spirv_get_clang_memory_scope(int Scope) {
  switch (Scope) {
  case CrossDevice:
    return __MEMORY_SCOPE_SYSTEM;
  case Device:
    return __MEMORY_SCOPE_DEVICE;
  case Workgroup:
    return __MEMORY_SCOPE_WRKGRP;
  case Subgroup:
    return __MEMORY_SCOPE_WVFRNT;
  case Invocation:
    return __MEMORY_SCOPE_SINGLE;
  default:
    __builtin_unreachable();
  }
}

static _CLC_INLINE int __spirv_get_clang_memory_order(int Semantics) {
  switch (Semantics & 0x1F) {
  case None:
    return __ATOMIC_RELAXED;
  case Acquire:
    return __ATOMIC_ACQUIRE;
  case Release:
    return __ATOMIC_RELEASE;
  case (Acquire | Release):
  case AcquireRelease:
    return __ATOMIC_ACQ_REL;
  case SequentiallyConsistent:
    // FIXME use __ATOMIC_SEQ_CST
    return __ATOMIC_ACQ_REL;
  default:
    __builtin_unreachable();
  }
}

#endif // __CLC_LIBSPIRV_ATOMIC_ATOMIC_HELPER_H__
