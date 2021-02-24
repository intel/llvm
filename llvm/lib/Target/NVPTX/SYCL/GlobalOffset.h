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

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class Function;
class Value;
class Type;
class MDNode;

ModulePass *createGlobalOffsetLegacyPass();

class GlobalOffsetPass : public PassInfoMixin<GlobalOffsetPass> {
public:
  PreservedAnalyses run(Module &, ModuleAnalysisManager &);
  static llvm::DenseMap<Function *, MDNode *> getEntryPointMetadata(Module &);

private:
  void processKernelEntryPoint(Module &, Function *);
  void addImplicitParameterToCallers(Module &, Value *, Function *);
  std::pair<Function *, Value *>
  addOffsetArgumentToFunction(Module &, Function *,
                              Type *ImplicitArgumentType = nullptr,
                              bool KeepOriginal = false);

  // Keep track of which functions have been processed to avoid processing twice
  llvm::DenseMap<Function *, Value *> ProcessedFunctions;
  // Keep a map of all entry point functions with metadata
  llvm::DenseMap<Function *, MDNode *> EntryPointMetadata;
  llvm::Type *KernelImplicitArgumentType;
  llvm::Type *ImplicitOffsetPtrType;
};

} // end namespace llvm

#endif
