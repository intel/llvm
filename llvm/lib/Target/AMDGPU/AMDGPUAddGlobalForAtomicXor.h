//===- AMDGPUAddGlobalForAtomicXor.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Some AMDGPU atomic instructions require a prefetch in order for them to work
// properly when using hipMallocManaged. This pass scans a module for the
// problematic atomic instructions and creates a global PrefetchNeeded if the
// builtin is present. This allows the prefetch to happen at runtime only if the
// problematic builtin is chosen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_ADDGLOBALFORATOMICXOR_H
#define LLVM_LIB_TARGET_AMDGPU_ADDGLOBALFORATOMICXOR_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class AMDGPUAddGlobalForAtomicXorPass
    : public PassInfoMixin<AMDGPUAddGlobalForAtomicXorPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_ADDGLOBALFORATOMICXOR_H
