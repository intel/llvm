//===- CleanupSYCLCompilerInternalMetadata.cpp - remove SYCL compiler MD --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Cleanup SYCL compiler internal metadata (usually inserted by sycl-post-link)
// as it will never be used in the compilation ever again
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/CleanupSYCLCompilerInternalMetadata.h"

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {

void cleanupSYCLCompilerModuleMetadata(const Module &M, llvm::StringRef MD) {
  NamedMDNode *Node = M.getNamedMetadata(MD);
  if (!Node)
    return;
  Node->clearOperands();
  Node->dropAllReferences();
  Node->eraseFromParent();
}

} // anonymous namespace

PreservedAnalyses
CleanupSYCLCompilerInternalMetadataPass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  // Remove SYCL module-level metadata that will never be used again to avoid
  // its duplication of their operands during llvm-link hence preventing
  // increase of the module size
  llvm::SmallVector<llvm::StringRef, 2> ModuleMDToRemove = {
      "sycl_aspects", "sycl_types_that_use_aspects"};
  for (const auto &MD : ModuleMDToRemove)
    cleanupSYCLCompilerModuleMetadata(M, MD);

  return PreservedAnalyses::all();
}
