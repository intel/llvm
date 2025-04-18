//===--------- CleanupSYCLMetadata.cpp - CleanupSYCLMetadata Pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Cleanup SYCL compiler internal metadata inserted by the frontend as it will
// never be used in the compilation ever again
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/CleanupSYCLMetadata.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/SYCLLowerIR/DeviceGlobals.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"

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

// GV is supposed to be either llvm.compiler.used or llvm.used.
SmallVector<Constant *>
eraseGlobalVariableAndReturnOperands(GlobalVariable *GV) {
  assert(GV->user_empty() && "Users aren't expected");
  Constant *Initializer = GV->getInitializer();
  GV->setInitializer(nullptr);
  GV->eraseFromParent();

  // Destroy the initializer and save operands.
  SmallVector<Constant *> Operands;
  Operands.resize(0);
  for (auto &Op : Initializer->operands())
    Operands.push_back(cast<Constant>(Op));

  assert(isSafeToDestroyConstant(Initializer) &&
         "Cannot remove initializer of the given GV");

  Initializer->destroyConstant();
  return Operands;
}

} // anonymous namespace

PreservedAnalyses CleanupSYCLMetadataPass::run(Module &M,
                                               ModuleAnalysisManager &MAM) {
  // Remove SYCL module-level metadata that will never be used again to avoid
  // duplication of their operands during llvm-link hence preventing
  // increase of the module size
  SmallVector<StringRef, 2> ModuleMDToRemove = {"sycl_aspects",
                                                "sycl_types_that_use_aspects"};
  for (const auto &MD : ModuleMDToRemove)
    cleanupSYCLCompilerModuleMetadata(M, MD);

  // Cleanup no longer needed function metadata.
  for (auto &F : M) {
    if (F.getMetadata("srcloc"))
      F.setMetadata("srcloc", nullptr);
  }

  return PreservedAnalyses::all();
}

PreservedAnalyses
CleanupSYCLMetadataFromLLVMUsed::run(Module &M, ModuleAnalysisManager &) {
  GlobalVariable *GV = M.getGlobalVariable("llvm.used");
  if (!GV)
    return PreservedAnalyses::all();

  SmallVector<Constant *, 8> IOperands =
      eraseGlobalVariableAndReturnOperands(GV);
  // Erase all operands.
  for (auto *Op : IOperands) {
    auto StrippedOp = Op->stripPointerCasts();
    auto *F = dyn_cast<Function>(StrippedOp);
    if (isSafeToDestroyConstant(Op))
      (Op)->destroyConstant();
    else if (F && F->getCallingConv() == CallingConv::SPIR_KERNEL &&
             !F->use_empty()) {
      // The element in "llvm.used" array has other users. That is Ok for
      // specialization constants, but is wrong for kernels.
      report_fatal_error("Unexpected usage of SYCL kernel");
    }

    // Remove unused kernel declarations to avoid LLVM IR check fails.
    if (F && F->isDeclaration() && F->use_empty())
      F->eraseFromParent();
  }

  return PreservedAnalyses::none();
}

PreservedAnalyses
RemoveDeviceGlobalFromLLVMCompilerUsed::run(Module &M,
                                            ModuleAnalysisManager &) {
  GlobalVariable *GV = M.getGlobalVariable("llvm.compiler.used");
  if (!GV)
    return PreservedAnalyses::all();

  const auto *VAT = cast<ArrayType>(GV->getValueType());
  // Destroy the initializer. Keep the operands so we keep the ones we need.
  SmallVector<Constant *> IOperands = eraseGlobalVariableAndReturnOperands(GV);

  // Iterate through all operands. If they are device_global then we drop them
  // and erase them if they have no uses afterwards. All other values are kept.
  SmallVector<Constant *> NewOperands;
  for (auto *Op : IOperands) {
    auto *DG = dyn_cast<GlobalVariable>(Op->stripPointerCasts());

    // If it is not a device_global we keep it.
    if (!DG || !isDeviceGlobalVariable(*DG)) {
      NewOperands.push_back(Op);
      continue;
    }

    // Destroy the device_global operand.
    if (isSafeToDestroyConstant(Op))
      Op->destroyConstant();

    // Remove device_global if it no longer has any uses.
    if (!DG->isConstantUsed())
      DG->eraseFromParent();
  }

  // If we have any operands left from the original llvm.compiler.used we create
  // a new one with the new size.
  if (!NewOperands.empty()) {
    ArrayType *ATy = ArrayType::get(VAT->getElementType(), NewOperands.size());
    GlobalVariable *NGV =
        new GlobalVariable(M, ATy, false, GlobalValue::AppendingLinkage,
                           ConstantArray::get(ATy, NewOperands), "");
    NGV->setName("llvm.compiler.used");
    NGV->setSection("llvm.metadata");
  }

  return PreservedAnalyses::none();
}
