//===-- ESIMDRemoveHostCode.cpp - remove host code for ESIMD -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// ESIMD code is not often run on the host, but we still compile for the host.
// If requested by the user, remove the implementations of all ESIMD functions
// to possibly speed up host compilation time.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ESIMDRemoveHostCodePass"

#include "llvm/Demangle/Demangle.h"
#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"
#include "llvm/SYCLLowerIR/ESIMD/LowerESIMD.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::esimd;
namespace id = itanium_demangle;
PreservedAnalyses ESIMDRemoveHostCodePass::run(Module &M,
                                               ModuleAnalysisManager &) {
  // TODO: Remove this pass once ESIMD headers are updated to
  // guard vectors to be device only.
  bool Modified = false;
  assert(!Triple(M.getTargetTriple()).isSPIR() &&
         "Pass should not be run for SPIR targets");
  for (auto &F : M.functions()) {
    if (F.isDeclaration())
      continue;
    StringRef MangledName = F.getName();
    id::ManglingParser<SimpleAllocator> Parser(MangledName.begin(),
                                               MangledName.end());
    id::Node *AST = Parser.parse();
    if (!AST || AST->getKind() != id::Node::KFunctionEncoding)
      continue;

    auto *FE = static_cast<id::FunctionEncoding *>(AST);
    const id::Node *NameNode = FE->getName();
    if (!NameNode)
      continue;

    id::OutputBuffer NameBuf;
    NameNode->print(NameBuf);
    StringRef Name(NameBuf.getBuffer(), NameBuf.getCurrentPosition());
    if (!Name.starts_with("sycl::_V1::ext::intel::esimd::") &&
        !Name.starts_with("sycl::_V1::ext::intel::experimental::esimd::"))
      continue;
    SmallVector<BasicBlock *> BBV;
    for (BasicBlock &BB : F) {
      BB.dropAllReferences();
      BBV.push_back(&BB);
    }
    for (auto *BB : BBV)
      BB->eraseFromParent();

    Value *Ret = nullptr;
    Type *RetTy = F.getFunctionType()->getReturnType();
    if (!RetTy->isVoidTy())
      Ret = Constant::getNullValue(RetTy);

    LLVMContext &Ctx = F.getParent()->getContext();
    BasicBlock *BB = BasicBlock::Create(Ctx, "", &F);
    ReturnInst::Create(Ctx, Ret, BB);
    Modified = true;
  }
  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
