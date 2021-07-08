//==-------------- atomic.hpp - support of atomic operations ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstdint>

#include "device.h"

#ifdef __SPIR__

#define SPIR_GLOBAL __attribute__((opencl_global))

namespace __spv {
struct Scope {

  enum Flag : uint32_t {
    CrossDevice = 0,
    Device = 1,
    Workgroup = 2,
    Subgroup = 3,
    Invocation = 4,
  };

  constexpr Scope(Flag flag) : flag_value(flag) {}

  constexpr operator uint32_t() const { return flag_value; }

  Flag flag_value;
};

struct MemorySemanticsMask {

  enum Flag : uint32_t {
    None = 0x0,
    Acquire = 0x2,
    Release = 0x4,
    AcquireRelease = 0x8,
    SequentiallyConsistent = 0x10,
    UniformMemory = 0x40,
    SubgroupMemory = 0x80,
    WorkgroupMemory = 0x100,
    CrossWorkgroupMemory = 0x200,
    AtomicCounterMemory = 0x400,
    ImageMemory = 0x800,
  };

  constexpr MemorySemanticsMask(Flag flag) : flag_value(flag) {}

  constexpr operator uint32_t() const { return flag_value; }

  Flag flag_value;
};
} // namespace __spv

extern DEVICE_EXTERNAL int
__spirv_AtomicCompareExchange(int SPIR_GLOBAL *, __spv::Scope::Flag,
                              __spv::MemorySemanticsMask::Flag,
                              __spv::MemorySemanticsMask::Flag, int, int);

extern DEVICE_EXTERNAL int __spirv_AtomicLoad(const int SPIR_GLOBAL *,
                                              __spv::Scope::Flag,
                                              __spv::MemorySemanticsMask::Flag);

extern DEVICE_EXTERNAL void
__spirv_AtomicStore(int SPIR_GLOBAL *, __spv::Scope::Flag,
                    __spv::MemorySemanticsMask::Flag, int);

/// Atomically set the value in *Ptr with Desired if and only if it is Expected
/// Return the value which already was in *Ptr
static inline int atomicCompareAndSet(SPIR_GLOBAL int *Ptr, int Desired,
                                      int Expected) {
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

#endif // __SPIR__
