//===-- LowerWGLocalMemory.cpp - SYCL kernel local memory allocation pass -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See intro comments in the header.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/LowerWGLocalMemory.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "LowerWGLocalMemory"

static constexpr char SYCL_ALLOCLOCALMEM_CALL[] = "__sycl_allocateLocalMemory";
static constexpr char LOCALMEMORY_GV_PREF[] = "WGLocalMem";

namespace {
class SYCLLowerWGLocalMemoryLegacy : public ModulePass {
public:
  static char ID;

  SYCLLowerWGLocalMemoryLegacy() : ModulePass(ID) {
    initializeSYCLLowerWGLocalMemoryLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    ModuleAnalysisManager DummyMAM;
    auto PA = Impl.run(M, DummyMAM);
    return !PA.areAllPreserved();
  }

private:
  SYCLLowerWGLocalMemoryPass Impl;
};
} // namespace

char SYCLLowerWGLocalMemoryLegacy::ID = 0;
INITIALIZE_PASS(SYCLLowerWGLocalMemoryLegacy, "sycllowerwglocalmemory",
                "Replace __sycl_allocateLocalMemory with allocation of memory "
                "in local address space",
                false, false)

ModulePass *llvm::createSYCLLowerWGLocalMemoryPass() {
  return new SYCLLowerWGLocalMemoryLegacy();
}

static void lowerAllocaLocalMemCall(CallInst *CI, Module &M) {
  Value *ArgSize = CI->getArgOperand(0);
  uint64_t Size = cast<llvm::ConstantInt>(ArgSize)->getZExtValue();
  Value *ArgAlign = CI->getArgOperand(1);
  uint64_t Alignment = cast<llvm::ConstantInt>(ArgAlign)->getZExtValue();

  IRBuilder<> Builder(CI);
  Type *LocalMemArrayTy = ArrayType::get(Builder.getInt8Ty(), Size);
  unsigned LocalAS =
      CI->getFunctionType()->getReturnType()->getPointerAddressSpace();
  auto *LocalMemArrayGV =
      new GlobalVariable(M,                                // module
                         LocalMemArrayTy,                  // type
                         false,                            // isConstant
                         GlobalValue::InternalLinkage,     // Linkage
                         UndefValue::get(LocalMemArrayTy), // Initializer
                         LOCALMEMORY_GV_PREF,              // Name prefix
                         nullptr,                          // InsertBefore
                         GlobalVariable::NotThreadLocal,   // ThreadLocalMode
                         LocalAS                           // AddressSpace
      );
  LocalMemArrayGV->setAlignment(Align(Alignment));

  Value *GVPtr =
      Builder.CreatePointerCast(LocalMemArrayGV, Builder.getInt8PtrTy(LocalAS));
  CI->replaceAllUsesWith(GVPtr);
  CI->eraseFromParent();
}

static bool allocaWGLocalMemory(Module &M) {
  SmallVector<CallInst *, 8> ToReplace;
  Function *allocaLocalMemF = nullptr;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    for (auto &I : instructions(F)) {
      auto *CI = dyn_cast<CallInst>(&I);
      Function *Callee = nullptr;
      if (!CI || !(Callee = CI->getCalledFunction()))
        continue;
      if (Callee->getName() != SYCL_ALLOCLOCALMEM_CALL)
        continue;

      assert(Callee->isDeclaration() &&
             "__sycl_allocateLocalMemory shouldn't have definition");

      // TODO: Static local memory allocation should be requested only in
      // spir kernel scope.
      CallingConv::ID CC = F.getCallingConv();
      assert((CC == llvm::CallingConv::SPIR_FUNC ||
              CC == llvm::CallingConv::SPIR_KERNEL) &&
             "WG static local memory can be allocated only in kernel scope");

      ToReplace.push_back(CI);
      allocaLocalMemF = Callee;
    }
  }

  if (ToReplace.empty())
    return false;

  for (auto *CI : ToReplace) {
    lowerAllocaLocalMemCall(CI, M);
  }

  // Remove declaration.
  assert(allocaLocalMemF->use_empty() &&
         "__sycl_allocateLocalMemory is still in use");
  allocaLocalMemF->eraseFromParent();

  return true;
}

PreservedAnalyses SYCLLowerWGLocalMemoryPass::run(Module &M,
                                                  ModuleAnalysisManager &) {
  if (allocaWGLocalMemory(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
