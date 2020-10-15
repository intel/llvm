//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLC_SPIRV_TYPES
#define CLC_SPIRV_TYPES

#include <lp64_types.h>

enum Scope {
  CrossDevice = 0,
  Device = 1,
  Workgroup = 2,
  Subgroup = 3,
  Invocation = 4,
};

enum MemorySemanticsMask {
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

enum FPRoundingMode {
  SPV_RTE = 0,
  SPV_RTZ = 1,
  SPV_RTP = 2,
  SPV_RTN = 3,
};

enum GroupOperation {
  Reduce = 0,
  InclusiveScan = 1,
  ExclusiveScan = 2,
};

#endif // CLC_SPIRV_TYPES
