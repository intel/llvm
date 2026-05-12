//===------------------ SanitizerPostSplitProcessing.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//
#include "llvm/SYCLPostLink/SanitizerPostSplitProcessing.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {

bool createSanitizerModuleID(Module &M) {
  constexpr StringRef Prefixes[] = {"__AsanKernelMetadata",
                                    "__MsanKernelMetadata",
                                    "__TsanKernelMetadata"};
  SmallVector<StringRef, 3> ModuleIDs;
  for (GlobalVariable &GV : M.globals()) {
    auto GVName = GV.getName();
    for (StringRef Prefix : Prefixes) {
      if (GVName.starts_with(Prefix)) {
        ModuleIDs.push_back(GVName.drop_front(Prefix.size()));
        break;
      }
    }
  }

  if (ModuleIDs.empty())
    return false;

  for (StringRef ModuleID : ModuleIDs) {
    FunctionType *FTy =
        FunctionType::get(Type::getVoidTy(M.getContext()), {}, false);
    Function *F = Function::Create(FTy, GlobalValue::WeakODRLinkage,
                                   "__sanitizerModule" + ModuleID, &M);
    F->setCallingConv(CallingConv::SPIR_KERNEL);
    F->setDSOLocal(true);
    F->addFnAttr("sycl-entry-point");
    BasicBlock *BB = BasicBlock::Create(M.getContext(), "entry", F);
    ReturnInst::Create(M.getContext(), BB);
  }

  return true;
}

} // namespace

bool llvm::sycl_post_link::handleSanitizers(
    llvm::SmallVectorImpl<std::unique_ptr<module_split::ModuleDesc>> &MDs) {
  bool Modified = false;
  for (std::unique_ptr<module_split::ModuleDesc> &MD : MDs)
    Modified |= createSanitizerModuleID(MD->getModule());
  return Modified;
}
