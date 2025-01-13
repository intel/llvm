//===- LocalAccessorToSharedMemory.cpp - Local Accessor Support for CUDA --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_LOCALACCESSORTOSHAREDMEMORY_H
#define LLVM_SYCL_LOCALACCESSORTOSHAREDMEMORY_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/SYCLLowerIR/TargetHelpers.h"

namespace llvm {

class ModulePass;
class PassRegistry;

/// This pass operates on SYCL kernels. It modifies kernel entry points which
/// take pointers to shared memory and alters them to take offsets into shared
/// memory (represented by a symbol in the shared address space). The SYCL
/// runtime is expected to provide offsets rather than pointers to these
/// functions.
class LocalAccessorToSharedMemoryPass
    : public PassInfoMixin<LocalAccessorToSharedMemoryPass> {
public:
  explicit LocalAccessorToSharedMemoryPass() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
  static StringRef getPassName() {
    return "SYCL Local Accessor to Shared Memory";
  }

private:
  /// This function replaces pointers to shared memory with offsets to a global
  /// symbol in shared memory.
  /// It alters the signature of the kernel (pointer vs offset value) as well
  /// as the access (dereferencing the argument pointer vs GEP to the global
  /// symbol).
  ///
  /// \param F The kernel to be processed.
  ///
  /// \returns A new function with global symbol accesses.
  Function *processKernel(Module &M, Function *F);

private:
  /// The value for NVVM's ADDRESS_SPACE_SHARED and AMD's LOCAL_ADDRESS happen
  /// to be 3.
  const unsigned SharedASValue = 3;
};

ModulePass *createLocalAccessorToSharedMemoryPassLegacy();
void initializeLocalAccessorToSharedMemoryLegacyPass(PassRegistry &);

} // end namespace llvm

#endif
