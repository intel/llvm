//===------ MutatePrintfAddrspace.cpp - SYCL printf AS mutation Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which detects non-constant address space
// literals usage for the first argument of SYCL experimental printf
// function, and moves the string literal to constant address
// space. This a temporary solution for printf's support of generic
// address space literals; the pass should be dropped once SYCL device
// backends learn to handle the generic address-spaced argument properly.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/MutatePrintfAddrspace.h"

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

namespace {
constexpr unsigned ConstantAddrspaceID = 2;

// Wrapper for the pass to make it working with the old pass manager
class SYCLMutatePrintfAddrspaceLegacyPass : public ModulePass {
public:
  static char ID;
  SYCLMutatePrintfAddrspaceLegacyPass() : ModulePass(ID) {
    initializeSYCLMutatePrintfAddrspaceLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  // run the SYCLMutatePrintfAddrspace pass on the specified module
  bool runOnModule(Module &M) override {
    ModuleAnalysisManager MAM;
    auto PA = Impl.run(M, MAM);
    return !PA.areAllPreserved();
  }

private:
  SYCLMutatePrintfAddrspacePass Impl;
};

Constant *getCASLiteral(IRBuilder<> &Builder, CallInst *CI) {
  auto *Literal = cast<Constant>(CI->getArgOperand(0));
  Type *LiteralType = Literal->getType();
  // Generate/find a correct literal
  auto *CASLiteralType = PointerType::get(LiteralType->getPointerElementType(),
                                          ConstantAddrspaceID);
  StringRef CASLiteralName(Literal->getName().str() + "._AS2");
  return CI->getModule()->getOrInsertGlobal(
      CASLiteralName, CASLiteralType, [&] {
        StringRef LiteralValue;
        getConstantStringInfo(Literal, LiteralValue);
        GlobalVariable *GV = Builder.CreateGlobalString(
            LiteralValue, CASLiteralName, ConstantAddrspaceID, CI->getModule());
        GV->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
        GV->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
        return GV;
      });
}

FunctionCallee getCASPrintfFunction(Function &GenericASPrintfFunc,
                                    Type *CASLiteralTy) {
  auto *CASPrintfFuncTy = FunctionType::get(GenericASPrintfFunc.getReturnType(),
                                            CASLiteralTy, /*isVarArg=*/true);
  FunctionCallee CASPrintfFunc =
      GenericASPrintfFunc.getParent()->getOrInsertFunction(
          "_Z18__spirv_ocl_printfPU3AS2Kcz", CASPrintfFuncTy,
          GenericASPrintfFunc.getAttributes());
  auto *Callee = cast<Function>(CASPrintfFunc.getCallee());
  Callee->setCallingConv(CallingConv::SPIR_FUNC);
  Callee->setDSOLocal(true);
  return CASPrintfFunc;
}

CallInst *buildCASPrintfCall(IRBuilder<> &Builder, FunctionCallee CASPrintfFunc,
                             Constant *CASLiteral, CallInst *CI) {
  SmallVector<Value *, 4> CallOperands(CI->data_operands_begin(),
                                       CI->data_operands_end());
  CallOperands[0] = CASLiteral;
  Builder.SetInsertPoint(CI);
  CallInst *CASCall = Builder.CreateCall(CASPrintfFunc, CallOperands,
                                         CI->getName().str() + "._AS2");
  CASCall->setTailCall(true);
  return CASCall;
}
} // namespace

PreservedAnalyses
SYCLMutatePrintfAddrspacePass::run(Module &M, ModuleAnalysisManager &MAM) {
  size_t ReplacedCallsCount = 0;
  SmallVector<Function *, 1> FunctionsToDrop;

  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;
    if (!F.getName().contains("__spirv_ocl_printf"))
      continue;
    auto *LiteralType = F.getArg(0)->getType();
    if (LiteralType->getPointerAddressSpace() == ConstantAddrspaceID)
      // No need to replace the literal type and its printf users
      continue;

    IRBuilder<> Builder(M.getContext());
    SmallVector<CallInst *, 8> CallInstsToDrop;
    for (User *U : F.users()) {
      if (!isa<CallInst>(U))
        continue;
      auto *CI = cast<CallInst>(U);
      Constant *CASLiteral = getCASLiteral(Builder, CI);
      FunctionCallee CASPrintfFunc =
          getCASPrintfFunction(F, CASLiteral->getType());
      CI->replaceAllUsesWith(
          buildCASPrintfCall(Builder, CASPrintfFunc, CASLiteral, CI));
      CallInstsToDrop.emplace_back(CI);
      ++ReplacedCallsCount;
    }
    for (CallInst *CI : CallInstsToDrop)
      CI->eraseFromParent();
    if (F.hasNUses(0))
      FunctionsToDrop.emplace_back(&F);
  }

  for (Function *F : FunctionsToDrop)
    F->eraseFromParent();
  return ReplacedCallsCount ? PreservedAnalyses::all()
                            : PreservedAnalyses::none();
}

char SYCLMutatePrintfAddrspaceLegacyPass::ID = 0;
INITIALIZE_PASS(SYCLMutatePrintfAddrspaceLegacyPass,
                "SYCLMutatePrintfAddrspace",
                "Move SYCL printf literal arguments to constant address space",
                false, false)

// Public interface to the SYCLMutatePrintfAddrspacePass.
ModulePass *llvm::createSYCLMutatePrintfAddrspaceLegacyPass() {
  return new SYCLMutatePrintfAddrspaceLegacyPass();
}
