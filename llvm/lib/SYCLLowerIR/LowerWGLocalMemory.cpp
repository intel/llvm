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

ModulePass *llvm::createSYCLLowerWGLocalMemoryLegacyPass() {
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

  assert(CI->use_empty() && "removing live instruction");
  CI->eraseFromParent();
}

static bool allocaWGLocalMemory(Module &M) {
  for (Function &F : M) {
    if (!F.isDeclaration() || F.getName() != SYCL_ALLOCLOCALMEM_CALL)
      continue;

    SmallVector<CallInst *, 4> ALMCalls;
    for (auto *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        ALMCalls.push_back(CI);
    }

    for (auto &CI : ALMCalls) {
      // Static local memory allocation should be requested only in
      // spir kernel scope (not a spir function) in accordance to OpenCL
      // restriction. However, __sycl_allocateLocalMemory is invoced in kernel
      // lambda call operator's scope, which is technically not SPIR-V kernel
      // scope.
      // TODO: Check if restriction may be relaxed for SYCL or imrpove pass
      // to move allocation of memory up to a spir kernel scope for each nested
      // device function call.
      CallingConv::ID CC = CI->getCaller()->getCallingConv();
      assert((CC == llvm::CallingConv::SPIR_FUNC ||
              CC == llvm::CallingConv::SPIR_KERNEL) &&
             "WG static local memory can be allocated only in kernel scope");

      lowerAllocaLocalMemCall(CI, M);
    }

    // Remove __sycl_allocateLocalMemory declaration.
    assert(F.use_empty() && "__sycl_allocateLocalMemory is still in use");
    F.eraseFromParent();

    return true;
  }

  return false;
}

PreservedAnalyses SYCLLowerWGLocalMemoryPass::run(Module &M,
                                                  ModuleAnalysisManager &) {
  if (allocaWGLocalMemory(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
