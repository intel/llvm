//===---------- GlobalOffset.h - Global Offset Support for CUDA ---------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass operates on SYCL kernels being compiled to CUDA. It looks for uses
// of the `llvm.nvvm.implicit.offset` intrinsic and replaces it with a offset
// parameter which will be threaded through from the kernel entry point.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_GLOBALOFFSET_H
#define LLVM_SYCL_GLOBALOFFSET_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/SYCLLowerIR/TargetHelpers.h"

namespace llvm {

class ModulePass;
class PassRegistry;

class GlobalOffsetPass : public PassInfoMixin<GlobalOffsetPass> {
private:
  using KernelPayload = TargetHelpers::KernelPayload;
  using ArchType = TargetHelpers::ArchType;

public:
  explicit GlobalOffsetPass() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
  static StringRef getPassName() { return "Add implicit SYCL global offset"; }

private:
  void processKernelEntryPoint(Module &M, Function *Func);
  void addImplicitParameterToCallers(Module &M, Value *Callee,
                                     Function *CalleeWithImplicitParam);
  std::pair<Function *, Value *>
  addOffsetArgumentToFunction(Module &M, Function *Func,
                              Type *ImplicitArgumentType = nullptr,
                              bool KeepOriginal = false);
  DenseMap<Function *, MDNode *>
  validateKernels(Module &M, SmallVectorImpl<KernelPayload> &KernelPayloads);

private:
  // Keep track of which functions have been processed to avoid processing twice
  llvm::DenseMap<Function *, Value *> ProcessedFunctions;
  // Keep a map of all entry point functions with metadata
  llvm::DenseMap<Function *, MDNode *> EntryPointMetadata;
  llvm::Type *KernelImplicitArgumentType;
  llvm::Type *ImplicitOffsetPtrType;

  ArchType AT;
  unsigned TargetAS = 0;
};

ModulePass *createGlobalOffsetPassLegacy();
void initializeGlobalOffsetLegacyPass(PassRegistry &);

} // end namespace llvm

#endif
