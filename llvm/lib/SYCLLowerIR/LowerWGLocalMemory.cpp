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
#include "llvm/IR/DIBuilder.h"
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

  // Make new global variable storage for local memory.
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

  // Add debug info for a new global variable using data from debug info for
  // __sycl_allocateLocalMemory call instruction if available.
  if (CI->getDebugLoc()) {
    DILocation *GroupLocMemInlineLoc = CI->getDebugLoc().getInlinedAt();
    llvm::DILocalScope *DScope = GroupLocMemInlineLoc->getScope();
    DISubprogram *DSubProg = DScope->getSubprogram();

    llvm::DIBuilder DBuilder(M);
    DebugInfoFinder DIF;
    DIF.processSubprogram(DSubProg);

    // Debug info for uint8_t should be in the scope of code block containing
    // __sycl_allocateLocalMemory call, because result of this call is assigned
    // to a variable of type `uint8_t *`.
    auto It = llvm::find_if(
        DIF.types(), [&](DIType *T) { return T->getName().equals("uint8_t"); });
    assert((It != DIF.types().end()) &&
           "debug info for uint8_t type not found");
    llvm::SmallVector<llvm::Metadata *, 1> Subscripts;
    auto *ElementsCountNode = llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::getSigned(Builder.getInt64Ty(), Size));
    Subscripts.push_back(DBuilder.getOrCreateSubrange(
        ElementsCountNode /*count*/, nullptr /*lowerBound*/,
        nullptr /*upperBound*/, nullptr /*stride*/));
    llvm::DINodeArray SubscriptArray = DBuilder.getOrCreateArray(Subscripts);
    llvm::DIType *LocalMemArrayDITy =
        DBuilder.createArrayType(Size * 8, Alignment * 8, *It, SubscriptArray);

    auto *LocalMemGVE = DBuilder.createGlobalVariableExpression(
        DScope,                             // Context
        LocalMemArrayGV->getName(),         // Name
        StringRef(),                        // LinkageName
        DScope->getFile(),                  // File
        GroupLocMemInlineLoc->getLine(),    // LineNo
        LocalMemArrayDITy,                  // Type
        LocalMemArrayGV->hasLocalLinkage(), // IsLocalToUnit
        true,                               // IsDefined
        nullptr,                            // Expr
        nullptr,                            // Decl
        nullptr,                            // TemplateParams
        Alignment * 8                       // AlignInBits
    );

    // Update Compile Unit globals list with a new debug info node
    DICompileUnit *DCU = DSubProg->getUnit();
    DIGlobalVariableExpressionArray CurrGVEs = DCU->getGlobalVariables();
    SmallVector<Metadata *, 64> NewGVEs(CurrGVEs.begin(), CurrGVEs.end());
    NewGVEs.push_back(LocalMemGVE);
    DCU->replaceGlobalVariables(MDTuple::get(M.getContext(), NewGVEs));

    LocalMemArrayGV->addDebugInfo(LocalMemGVE);
  }

  Value *GVPtr =
      Builder.CreatePointerCast(LocalMemArrayGV, Builder.getInt8PtrTy(LocalAS));
  CI->replaceAllUsesWith(GVPtr);
}

static bool allocaWGLocalMemory(Module &M) {
  Function *ALMFunc = M.getFunction(SYCL_ALLOCLOCALMEM_CALL);
  if (!ALMFunc)
    return false;

  assert(ALMFunc->isDeclaration() && "should have declaration only");

  SmallVector<CallInst *, 4> DelCalls;
  for (User *U : ALMFunc->users()) {
    auto *CI = cast<CallInst>(U);
    lowerAllocaLocalMemCall(CI, M);
    DelCalls.push_back(CI);
  }

  for (auto *CI : DelCalls) {
    assert(CI->use_empty() && "removing live instruction");
    CI->eraseFromParent();
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
