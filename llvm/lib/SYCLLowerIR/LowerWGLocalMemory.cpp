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

// Static local memory allocation should be allowed only in a scope of a kernel
// (not a device function) and shouldn't be called inside loop or if statement
// to make it consistent with OpenCL restriction.
// TODO: Relax that restriction for SYCL or modify this pass to move allocation
// of memory up to a kernel scope at the beginning for each nested device
// function call, loop or if statement.
static void lowerAllocaLocalMemCall(CallInst *CI, Module &M) {
  assert(CI);

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
  Function *ALMFunc = M.getFunction(SYCL_ALLOCLOCALMEM_CALL);
  if (!ALMFunc)
    return false;

  assert(ALMFunc->isDeclaration() && "should have declaration only");

  for (User *U : ALMFunc->users()) {
    auto *CI = cast<CallInst>(U);
    lowerAllocaLocalMemCall(CI, M);
  }

  // Remove __sycl_allocateLocalMemory declaration.
  assert(ALMFunc->use_empty() && "__sycl_allocateLocalMemory is still in use");
  ALMFunc->eraseFromParent();

  return true;
}

PreservedAnalyses SYCLLowerWGLocalMemoryPass::run(Module &M,
                                                  ModuleAnalysisManager &) {
  if (allocaWGLocalMemory(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
