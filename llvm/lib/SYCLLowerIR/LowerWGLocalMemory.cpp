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
#include "llvm/TargetParser/Triple.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "LowerWGLocalMemory"

static constexpr char SYCL_ALLOCLOCALMEM_CALL[] = "__sycl_allocateLocalMemory";
static constexpr char SYCL_DYNAMIC_LOCALMEM_CALL[] = "__sycl_dynamicLocalMemoryPlaceholder";
static constexpr char LOCALMEMORY_GV_PREF[] = "WGLocalMem";
static constexpr char DYNAMIC_LOCALMEM_GV[] = "__sycl_dynamicLocalMemoryPlaceholder_GV";

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

// TODO: It should be checked that __sycl_allocateLocalMemory (or its source
// form - group_local_memory) does not occur:
//  - in a function (other than user lambda/functor)
//  - in a loop
//  - in a non-convergent control flow
// to make it consistent with OpenCL restriction.
// But LLVM pass is not the best place to diagnose these cases.
// Error checking should be done in the front-end compiler.
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
}

static void lowerDynamicLocalMemCallDirect(CallInst *CI, Triple TT, GlobalVariable *LocalMemPlaceholder) {
  assert(CI);

  Value *GVPtr = [&]() -> Value* {
    IRBuilder<> Builder(CI);
    if (TT.isSPIROrSPIRV()) {

      return Builder.CreateLoad(
          CI->getType(),
          LocalMemPlaceholder);
    } else {
      Value *ArgAlign = CI->getArgOperand(0);
      Align RequestedAlignment{
          cast<llvm::ConstantInt>(ArgAlign)->getZExtValue()};
      MaybeAlign CurrentAlignment = LocalMemPlaceholder->getAlign();
      if (!CurrentAlignment.has_value() ||
          (CurrentAlignment.value() < RequestedAlignment))
        LocalMemPlaceholder->setAlignment(RequestedAlignment);

      return Builder.CreatePointerCast(LocalMemPlaceholder,
                                       CI->getType());
    }
  }();
  CI->replaceAllUsesWith(GVPtr);
}

static void lowerLocalMemCall(Function *LocalMemAllocFunc,
                              std::function<void(CallInst *CI)> TransformCall) {
  SmallVector<CallInst *, 4> DelCalls;
  for (User *U : LocalMemAllocFunc->users()) {
    auto *CI = cast<CallInst>(U);
    TransformCall(CI);
    DelCalls.push_back(CI);
  }

  for (auto *CI : DelCalls) {
    assert(CI->use_empty() && "removing live instruction");
    CI->eraseFromParent();
  }

  // Remove __sycl_allocateLocalMemory declaration.
  assert(LocalMemAllocFunc->use_empty() &&
         "local mem allocation function is still in use");
  LocalMemAllocFunc->eraseFromParent();
}

static bool allocaWGLocalMemory(Module &M) {
  Function *ALMFunc = M.getFunction(SYCL_ALLOCLOCALMEM_CALL);
  if (!ALMFunc)
    return false;

  assert(ALMFunc->isDeclaration() && "should have declaration only");

  lowerLocalMemCall(ALMFunc,
                    [&](CallInst *CI) { lowerAllocaLocalMemCall(CI, M); });

  return true;
}

// For dynamic memory we have 2 case:
//   - Direct for CUDA/HIP: we create a placeholder and set the memory on launch
//   - Indirect for OpenCL/Level0: we create a shared value holding the pointer to the buffer passed as argument
static bool dynamicWGLocalMemory(Module &M) {
  Function *DLMFunc = M.getFunction(SYCL_DYNAMIC_LOCALMEM_CALL);
  if (!DLMFunc)
    return false;


  GlobalVariable *LocalMemArrayGV = M.getGlobalVariable(DYNAMIC_LOCALMEM_GV);
  Triple TT(M.getTargetTriple());

  if (!LocalMemArrayGV) {

    assert(DLMFunc->isDeclaration() && "should have declaration only");
    unsigned LocalAS = DLMFunc->getReturnType()->getPointerAddressSpace();
    Type *LocalMemArrayTy = TT.isSPIROrSPIRV() ? static_cast<Type*>(PointerType::get(M.getContext(), LocalAS)) :
    static_cast<Type*>(ArrayType::get(Type::getInt8Ty(M.getContext()), 0));
    LocalMemArrayGV =
        new GlobalVariable(M,                             // module
                          LocalMemArrayTy,                // type
                          false,                          // isConstant
                          GlobalValue::ExternalLinkage,   // Linkage
                          nullptr,                        // Initializer
                          DYNAMIC_LOCALMEM_GV,            // Name prefix
                          nullptr,                        // InsertBefore
                          GlobalVariable::NotThreadLocal, // ThreadLocalMode
                          LocalAS                         // AddressSpace
        );
  }
  lowerLocalMemCall(DLMFunc, [&](CallInst *CI) {
    lowerDynamicLocalMemCallDirect(CI, TT, LocalMemArrayGV);
  });

  return true;
}

PreservedAnalyses SYCLLowerWGLocalMemoryPass::run(Module &M,
                                                  ModuleAnalysisManager &) {
  bool MadeChanges = allocaWGLocalMemory(M);
  MadeChanges = dynamicWGLocalMemory(M) || MadeChanges;
  if (MadeChanges)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
