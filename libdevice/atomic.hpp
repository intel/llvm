//==-------------- atomic.cpp - support of atomic operations ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "device.h"

#define __SYCL_GLOBAL__ __attribute__((opencl_global))

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
    __spirv_AtomicCompareExchange(int __SYCL_GLOBAL__ *, __spv::Scope::Flag,
                                  __spv::MemorySemanticsMask::Flag,
                                  __spv::MemorySemanticsMask::Flag,
                                  int, int);

inline void CAS(__SYCL_GLOBAL__ int *Ptr, int Expected, int Desired) {
  __spirv_AtomicCompareExchange(
      Ptr, __spv::Scope::Device,
      __spv::MemorySemanticsMask::SequentiallyConsistent,
      __spv::MemorySemanticsMask::SequentiallyConsistent, Desired, Expected);
}

