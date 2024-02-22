//==---------------------------- Cleanup.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Cleanup.h"

#include <algorithm>
#include <utility>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>

#include "Kernel.h"
#include "kernel-info/SYCLKernelInfo.h"

using namespace llvm;

static FunctionType *createMaskedFunctionType(const BitVector &Mask,
                                              const FunctionType *FTy) {
  SmallVector<Type *> Params;
  std::transform(Mask.set_bits_begin(), Mask.set_bits_end(),
                 std::back_inserter(Params),
                 [&](unsigned I) { return FTy->getParamType(I); });
  return FunctionType::get(FTy->getReturnType(), Params, false);
}

static void copyAttributesFrom(const BitVector &Mask, Function *NF,
                               const Function *F) {
  // Copies attributes, calling convention and other relevant data.
  NF->copyAttributesFrom(F);
  // Drop masked-out attributes.
  SmallVector<AttributeSet> Attributes;
  const AttributeList PAL = NF->getAttributes();
  std::transform(Mask.set_bits_begin(), Mask.set_bits_end(),
                 std::back_inserter(Attributes),
                 [&](unsigned I) { return PAL.getParamAttrs(I); });
  NF->setAttributes(AttributeList::get(NF->getContext(), PAL.getFnAttrs(),
                                       PAL.getRetAttrs(), Attributes));
}

static Function *createMaskedFunction(const BitVector &Mask, Function *F,
                                      TargetFusionInfo &TFI) {
  // Declare
  FunctionType *NFTy = createMaskedFunctionType(Mask, F->getFunctionType());
  Function *NF = Function::Create(NFTy, F->getLinkage(), F->getAddressSpace(),
                                  F->getName(), F->getParent());
  NF->IsNewDbgInfoFormat = UseNewDbgInfoFormat;
  copyAttributesFrom(Mask, NF, F);
  NF->setComdat(F->getComdat());
  NF->takeName(F);

  // Copy body
  NF->splice(NF->begin(), F);
  {
    // Transfer uses to new arguments.
    auto *ArgsIt = NF->arg_begin();
    auto SetIt = Mask.set_bits_begin();
    const auto End = Mask.set_bits_end();
    for (; SetIt != End; ++ArgsIt, ++SetIt) {
      auto *OldArg = F->getArg(*SetIt);
      OldArg->replaceAllUsesWith(ArgsIt);
      ArgsIt->takeName(OldArg);
    }
  }

  {
    // Copy metadata.
    SmallVector<std::pair<unsigned, MDNode *>> MDs;
    F->getAllMetadata(MDs);
    for (auto &MD : MDs) {
      NF->addMetadata(MD.first, *MD.second);
    }
  }

  // Erase old function
  TFI.notifyFunctionsDelete(F);
  F->eraseFromParent();
  TFI.addKernelFunction(NF);
  return NF;
}

static void updateArgUsageMask(jit_compiler::SYCLKernelInfo *Info,
                               ArrayRef<jit_compiler::ArgUsageUT> NewArgInfo) {
  auto &KernelMask = Info->Args.UsageMask;
  auto New = NewArgInfo.begin();
  for (auto &C : KernelMask) {
    if (C & jit_compiler::ArgUsage::Used) {
      if (*New & jit_compiler::ArgUsage::Used) {
        // Preserve previously applied usage information (e.g., internalization
        // flags) if the argument is still used.
        C |= *New;
      } else {
        C = *New;
      }
      ++New;
    }
  }
}

static void applyArgMask(ArrayRef<jit_compiler::ArgUsageUT> NewArgInfo,
                         const BitVector &Mask, Function *F,
                         ModuleAnalysisManager &AM, TargetFusionInfo &TFI) {
  // Create the function without the masked-out args.
  Function *NF = createMaskedFunction(Mask, F, TFI);
  // Update the unused args mask.
  jit_compiler::SYCLModuleInfo *ModuleInfo =
      AM.getResult<SYCLModuleInfoAnalysis>(*NF->getParent()).ModuleInfo;
  jit_compiler::SYCLKernelInfo *Info =
      ModuleInfo->getKernelFor(NF->getName().str());
  if (!Info) {
    errs() << "No info available for kernel " << NF->getName().str() << "\n";
    return;
  }
  updateArgUsageMask(Info, NewArgInfo);
}

static void maskMD(const BitVector &Mask, Function *F) {
  LLVMContext &LLVMCtx = F->getContext();
  // Get old metadata.
  SmallVector<std::pair<unsigned, MDNode *>> MD;
  F->getAllMetadata(MD);
  for (const auto &Entry : MD) {
    if (Entry.second->getNumOperands() != Mask.size()) {
      // Some metadata, e.g., the metadata for reqd_work_group_size and
      // work_group_size_hint is independent from the number of arguments
      // and must not be filtered by the argument usage mask.
      continue;
    }
    SmallVector<Metadata *> NewMD;
    // Add only MD for enabled arguments.
    std::transform(
        Mask.set_bits_begin(), Mask.set_bits_end(), std::back_inserter(NewMD),
        [&](unsigned I) { return Entry.second->getOperand(I).get(); });
    F->setMetadata(Entry.first, MDNode::get(LLVMCtx, NewMD));
  }
}

void llvm::fullCleanup(ArrayRef<jit_compiler::ArgUsageUT> ArgUsageInfo,
                       Function *F, ModuleAnalysisManager &AM,
                       TargetFusionInfo &TFI, ArrayRef<StringRef> MDToErase) {
  // Erase metadata.
  for (auto Key : MDToErase) {
    F->setMetadata(Key, nullptr);
  }
  BitVector CleanupMask{static_cast<unsigned int>(ArgUsageInfo.size()), true};
  for (auto I : enumerate(ArgUsageInfo)) {
    if (!(I.value() & jit_compiler::ArgUsage::Used)) {
      CleanupMask.reset(I.index());
    }
  }
  // Update metadata.
  maskMD(CleanupMask, F);
  // Remove arguments.
  applyArgMask(ArgUsageInfo, CleanupMask, F, AM, TFI);
}
