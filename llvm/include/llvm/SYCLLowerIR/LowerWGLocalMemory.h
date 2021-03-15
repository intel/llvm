//===-- LowerWGLocalMemory.h - SYCL kernel local memory allocation pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replaces calls to __sycl_allocateLocalMemory(Size, Alignment) function with
// allocation of memory in local address space at the kernel scope.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCLLOWERIR_LOWERWGLOCALMEMORY_H
#define LLVM_SYCLLOWERIR_LOWERWGLOCALMEMORY_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLLowerWGLocalMemoryPass
    : public PassInfoMixin<SYCLLowerWGLocalMemoryPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

ModulePass *createSYCLLowerWGLocalMemoryPass();
void initializeSYCLLowerWGLocalMemoryLegacyPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_SYCLLOWERIR_LOWERWGLOCALMEMORY_H
