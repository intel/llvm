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

#include <memory>

using namespace llvm;

#define DEBUG_TYPE "LowerWGLocalMemory"

static constexpr char SYCL_ALLOCLOCALMEM_CALL[] = "__sycl_allocateLocalMemory";
static constexpr char LOCALMEMORY_GV_PREF[] = "WGLocalMem";

namespace {

class CallInstParams {
public:
  CallInstParams(const CallInst *CI) : LocalMemCallDLoc(CI->getDebugLoc()) {
    Value *ArgSize = CI->getArgOperand(0);
    LocalMemSize = cast<llvm::ConstantInt>(ArgSize)->getZExtValue();
    Value *ArgAlign = CI->getArgOperand(1);
    LocalMemAlignment = cast<llvm::ConstantInt>(ArgAlign)->getZExtValue();
    LocalMemAS =
        CI->getFunctionType()->getReturnType()->getPointerAddressSpace();
  }

  uint64_t LocalMemSize;
  uint64_t LocalMemAlignment;
  unsigned LocalMemAS;

  const DebugLoc &LocalMemCallDLoc;
};

class SYCLLowerWGLocalMemoryHelper {
  Module &M;
  Function *ALMFunc;

  IRBuilder<> Builder;
  llvm::DIBuilder DBuilder;

  SmallVector<CallInst *, 4> LoweredCalls = {};
  DIType *Uint8DIT = nullptr;

public:
  SYCLLowerWGLocalMemoryHelper(Module &SrcM)
      : M(SrcM), ALMFunc(SrcM.getFunction(SYCL_ALLOCLOCALMEM_CALL)),
        Builder(SrcM.getContext()), DBuilder(SrcM) {}

  ~SYCLLowerWGLocalMemoryHelper();

  bool lowerAllocaLocalMemCalls();

private:
  void lowerAllocaLocalMemCall(CallInst *CI);
  GlobalVariable *genStaticLocalMemStorage(const CallInstParams &Params);
  void setGlobalVarDebugInfo(GlobalVariable *GV, const CallInstParams &Params);
  DIType *getUInt8DITy(DISubprogram *DSubProg);
};

bool SYCLLowerWGLocalMemoryHelper::lowerAllocaLocalMemCalls() {
  if (!ALMFunc || ALMFunc->use_empty())
    return false;

  assert(ALMFunc->isDeclaration() && "should have declaration only");

  for (User *U : ALMFunc->users())
    lowerAllocaLocalMemCall(cast<CallInst>(U));

  assert(ALMFunc->use_empty() && "__sycl_allocateLocalMemory is still in use");

  return true;
}

SYCLLowerWGLocalMemoryHelper::~SYCLLowerWGLocalMemoryHelper() {
  // Delete old calls of __sycl_allocateLocalMemory from IR.
  for (auto *CI : LoweredCalls)
    CI->eraseFromParent();

  // Remove __sycl_allocateLocalMemory declaration.
  if (ALMFunc)
    ALMFunc->eraseFromParent();

  Uint8DIT = nullptr;
}

// TODO: It should be checked that __sycl_allocateLocalMemory (or its source
// form - group_local_memory) does not occur:
//  - in a function (other than user lambda/functor)
//  - in a loop
//  - in a non-convergent control flow
// to make it consistent with OpenCL restriction.
// But LLVM pass is not the best place to diagnose these cases.
// Error checking should be done in the front-end compiler.
void SYCLLowerWGLocalMemoryHelper::lowerAllocaLocalMemCall(CallInst *CI) {
  assert(CI);
  CallInstParams LocalMemParams(CI);
  Builder.SetInsertPoint(CI);

  GlobalVariable *NewGV = genStaticLocalMemStorage(LocalMemParams);
  setGlobalVarDebugInfo(NewGV, LocalMemParams);

  Value *GVPtr = Builder.CreatePointerCast(
      NewGV, Builder.getInt8PtrTy(LocalMemParams.LocalMemAS));
  CI->replaceAllUsesWith(GVPtr);

  assert(CI->use_empty() && "removing live instruction");
  LoweredCalls.push_back(CI);
}

// Make new global variable as storage for local memory.
GlobalVariable *SYCLLowerWGLocalMemoryHelper::genStaticLocalMemStorage(
    const CallInstParams &Params) {
  Type *LocalMemArrayTy =
      ArrayType::get(Builder.getInt8Ty(), Params.LocalMemSize);
  auto *LocalMemArrayGV =
      new GlobalVariable(M,                                // module
                         LocalMemArrayTy,                  // type
                         false,                            // isConstant
                         GlobalValue::InternalLinkage,     // Linkage
                         UndefValue::get(LocalMemArrayTy), // Initializer
                         LOCALMEMORY_GV_PREF,              // Name prefix
                         nullptr,                          // InsertBefore
                         GlobalVariable::NotThreadLocal,   // ThreadLocalMode
                         Params.LocalMemAS                 // AddressSpace
      );
  LocalMemArrayGV->setAlignment(Align(Params.LocalMemAlignment));
  return LocalMemArrayGV;
}

// Add debug info for a new global variable using data from debug info for
// __sycl_allocateLocalMemory call instruction if available.
void SYCLLowerWGLocalMemoryHelper::setGlobalVarDebugInfo(
    GlobalVariable *GV, const CallInstParams &Params) {
  if (!Params.LocalMemCallDLoc)
    return;

  DILocation *DLoc = Params.LocalMemCallDLoc.getInlinedAt();
  llvm::DILocalScope *DScope = DLoc->getScope();
  DIType *DTy = getUInt8DITy(DScope->getSubprogram());

  // Create debug info for an array of uint8_t elements which is a type of
  // global variable.
  llvm::SmallVector<llvm::Metadata *, 1> Subscripts;
  auto *ElementsCountNode =
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(
          Type::getInt64Ty(M.getContext()), Params.LocalMemSize));
  Subscripts.push_back(DBuilder.getOrCreateSubrange(
      ElementsCountNode /*count*/, nullptr /*lowerBound*/,
      nullptr /*upperBound*/, nullptr /*stride*/));
  llvm::DINodeArray SubscriptArray = DBuilder.getOrCreateArray(Subscripts);
  llvm::DIType *LocalMemArrayDITy = DBuilder.createArrayType(
      Params.LocalMemSize * 8, Params.LocalMemAlignment * 8, DTy,
      SubscriptArray);

  auto *LocalMemGVE = DBuilder.createGlobalVariableExpression(
      DScope,                      // Context
      GV->getName(),               // Name
      StringRef(),                 // LinkageName
      DScope->getFile(),           // File
      DLoc->getLine(),             // LineNo
      LocalMemArrayDITy,           // Type
      GV->hasLocalLinkage(),       // IsLocalToUnit
      true,                        // IsDefined
      nullptr,                     // Expr
      nullptr,                     // Decl
      nullptr,                     // TemplateParams
      Params.LocalMemAlignment * 8 // AlignInBits
  );

  // Update Compile Unit globals list with a new debug info node
  DICompileUnit *DCU = DScope->getSubprogram()->getUnit();
  DIGlobalVariableExpressionArray CurrGVEs = DCU->getGlobalVariables();
  SmallVector<Metadata *, 64> NewGVEs(CurrGVEs.begin(), CurrGVEs.end());
  NewGVEs.push_back(LocalMemGVE);
  DCU->replaceGlobalVariables(MDTuple::get(M.getContext(), NewGVEs));

  GV->addDebugInfo(LocalMemGVE);
}

DIType *SYCLLowerWGLocalMemoryHelper::getUInt8DITy(DISubprogram *DSubProg) {
  if (!Uint8DIT) {
    // Debug info for uint8_t should be in the scope of code block containing
    // __sycl_allocateLocalMemory call, because result of this call is assigned
    // to a variable of type `uint8_t *`.
    DebugInfoFinder DIF;
    DIF.processSubprogram(DSubProg);
    auto It = llvm::find_if(
        DIF.types(), [&](DIType *T) { return T->getName().equals("uint8_t"); });
    assert((It != DIF.types().end()) &&
           "debug info for uint8_t type not found");
    Uint8DIT = *It;
  }
  return Uint8DIT;
}

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

PreservedAnalyses SYCLLowerWGLocalMemoryPass::run(Module &M,
                                                  ModuleAnalysisManager &) {
  SYCLLowerWGLocalMemoryHelper Helper(M);
  if (Helper.lowerAllocaLocalMemCalls())
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
