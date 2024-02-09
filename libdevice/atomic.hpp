//==-------------- atomic.hpp - support of atomic operations ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "spirv_decls.hpp"

#if defined(__SPIR__)
/// Atomically set the value in *Ptr with Desired if and only if it is Expected
/// Return the value which already was in *Ptr
static inline int atomicCompareAndSet(SPIR_GLOBAL int *Ptr, int Desired,
                                      int Expected) {
  return __spirv_AtomicCompareExchange(
      Ptr, __spv::Scope::Device,
      __spv::MemorySemanticsMask::SequentiallyConsistent,
      __spv::MemorySemanticsMask::SequentiallyConsistent, Desired, Expected);
}

static inline int atomicCompareAndSet(int *Ptr, int Desired, int Expected) {
  return __spirv_AtomicCompareExchange(
      Ptr, __spv::Scope::Device,
      __spv::MemorySemanticsMask::SequentiallyConsistent,
      __spv::MemorySemanticsMask::SequentiallyConsistent, Desired, Expected);
}

static inline int atomicLoad(SPIR_GLOBAL int *Ptr) {
  return __spirv_AtomicLoad(Ptr, __spv::Scope::Device,
                            __spv::MemorySemanticsMask::SequentiallyConsistent);
}

static inline void atomicStore(SPIR_GLOBAL int *Ptr, int V) {
  __spirv_AtomicStore(Ptr, __spv::Scope::Device,
                      __spv::MemorySemanticsMask::SequentiallyConsistent, V);
}

static inline void atomicStore(int *Ptr, int V) {
  __spirv_AtomicStore(Ptr, __spv::Scope::Device,
                      __spv::MemorySemanticsMask::SequentiallyConsistent, V);
}

#endif // __SPIR__
