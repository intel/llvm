//=== SYCLJointMatrixTransform.cpp - SYCL Joint Matrix transformation Pass ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which mutates Joint Matrix builtin calls to make them
// conformat with SPIR-V friendly LLVM IR specification.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLJointMatrixTransform.h"

#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace {

static constexpr char ACCESS_CHAIN[] = "_Z19__spirv_AccessChain";
static constexpr char MATRIX_TYPE[] = "spirv.CooperativeMatrixKHR";

// This routine extracts spirv.CooperativeMatrixKHR target extension type
// from sycl::joint_matrix class object if it's used in __spirv_AccessChain
// function call. It's necessary because otherwise OpAccessChain indices would
// be wrong.
bool transformAccessChain(Function *F) {
  bool ModuleChanged = false;
  for (auto I : F->users()) {
    auto *CI = dyn_cast<CallInst>(I);
    if (!CI)
      continue;
    Instruction *Ptr =
        dyn_cast<Instruction>(CI->getArgOperand(0)->stripPointerCasts());
    if (!Ptr || !isa<AllocaInst>(Ptr))
      continue;
    StructType *WrapperMatrixTy =
        dyn_cast<StructType>(cast<AllocaInst>(Ptr)->getAllocatedType());
    if (!WrapperMatrixTy)
      continue;
    TargetExtType *MatrixTy =
        dyn_cast<TargetExtType>(WrapperMatrixTy->getElementType(0));
    if (!MatrixTy)
      continue;
    StringRef Name = MatrixTy->getName();
    if (Name != MATRIX_TYPE)
      continue;

    AllocaInst *Alloca = nullptr;
    {
      IRBuilder Builder(CI);
      IRBuilderBase::InsertPointGuard IG(Builder);
      Builder.SetInsertPointPastAllocas(CI->getFunction());
      Alloca = Builder.CreateAlloca(MatrixTy);
    }
    Ptr->replaceAllUsesWith(Alloca);
    Ptr->dropAllReferences();
    Ptr->eraseFromParent();
    ModuleChanged = true;
  }
  return ModuleChanged;
}
} // namespace

PreservedAnalyses
SYCLJointMatrixTransformPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;
    if (F.getName().starts_with(ACCESS_CHAIN))
      ModuleChanged |= transformAccessChain(&F);
  }

  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
