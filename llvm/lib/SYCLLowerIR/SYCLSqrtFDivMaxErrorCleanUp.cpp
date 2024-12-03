//===- SYCLSqrtFDivMaxErrorCleanUp.cpp - SYCLSqrtFDivMaxErrorCleanUp Pass -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Remove llvm.fpbuiltin.[sqrt/fdiv] intrinsics to ensure compatibility with the
// old drivers (that don't support SPV_INTEL_fp_max_error extension).
// The intrinsic functions are removed in case if they are used with standard
// for OpenCL max-error (e.g [3.0/2.5] ULP) and there are no:
// - other llvm.fpbuiltin.* intrinsic functions;
// - fdiv instructions
// - @sqrt builtins (both C and C++-styles)/llvm intrinsic in the module.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLSqrtFDivMaxErrorCleanUp.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace {
static constexpr char SQRT_ERROR[] = "3.0";
static constexpr char FDIV_ERROR[] = "2.5";
} // namespace

PreservedAnalyses
SYCLSqrtFDivMaxErrorCleanUpPass::run(Module &M,
                                     ModuleAnalysisManager &MAM) {
  SmallVector<IntrinsicInst *, 16> WorkListSqrt;
  SmallVector<IntrinsicInst *, 16> WorkListFDiv;

  // Add all llvm.fpbuiltin.sqrt with 3.0 error and llvm.fpbuiltin.fdiv with
  // 2.5 error to the work list to remove them later. If attributes with other
  // values or other llvm.fpbuiltin.* intrinsic functions found - abort the
  // pass.
  for (auto &F : M) {
    if (!F.isDeclaration())
      continue;
    const auto ID = F.getIntrinsicID();
    if (ID != llvm::Intrinsic::fpbuiltin_sqrt &&
        ID != llvm::Intrinsic::fpbuiltin_fdiv)
      continue;

    for (auto *Use : F.users()) {
      auto *II = cast<IntrinsicInst>(Use);
      if (II && II->getCalledFunction()->getName().
          starts_with("llvm.fpbuiltin")) {
        // llvm.fpbuiltin.* intrinsics should always have fpbuiltin-max-error
        // attribute, but it's not a concern of the pass, so just do an early
        // exit here if the attribute is not attached.
        if (!II->getAttributes().hasFnAttr("fpbuiltin-max-error"))
          return PreservedAnalyses::none();
        StringRef MaxError = II->getAttributes().getFnAttr(
            "fpbuiltin-max-error").getValueAsString();

        if (ID == llvm::Intrinsic::fpbuiltin_sqrt) {
          if (MaxError != SQRT_ERROR)
            return PreservedAnalyses::none();
          WorkListSqrt.push_back(II);
        }
        else if (ID == llvm::Intrinsic::fpbuiltin_fdiv) {
          if (MaxError != FDIV_ERROR)
            return PreservedAnalyses::none();
          WorkListFDiv.push_back(II);
        } else {
          // Another llvm.fpbuiltin.* intrinsic was found - the module is
          // already not backward compatible.
          return PreservedAnalyses::none();
        }
      }
    }
  }

  // No intrinsics at all - do an early exist.
  if (WorkListSqrt.empty() && WorkListFDiv.empty())
    return PreservedAnalyses::none();

  // If @sqrt, @_Z4sqrt*, @llvm.sqrt. or fdiv present in the module - do
  // nothing.
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    for (auto &BB : F) {
      for (auto &II : BB) {
        if (auto *CI = dyn_cast<CallInst>(&II)) {
          auto *SqrtF = CI->getCalledFunction();
          if (SqrtF->getName() == "sqrt" ||
              SqrtF->getName().starts_with("_Z4sqrt") ||
              SqrtF->getIntrinsicID() == llvm::Intrinsic::sqrt)
            return PreservedAnalyses::none();
        }
        if (auto *FPI = dyn_cast<FPMathOperator>(&II)) {
          auto Opcode = FPI->getOpcode();
          if (Opcode == Instruction::FDiv)
            return PreservedAnalyses::none();
        }
      }
    }
  }

  // Replace @llvm.fpbuiltin.sqrt call with @llvm.sqrt. llvm-spirv will handle
  // it later.
  SmallSet<Function *, 2> DeclToRemove;
  for (auto *Sqrt : WorkListSqrt) {
    DeclToRemove.insert(Sqrt->getCalledFunction());
    IRBuilder Builder(Sqrt);
    Builder.SetInsertPoint(Sqrt);
    Type *Ty = Sqrt->getType();
    AttributeList Attrs = Sqrt->getAttributes();
    Function *NewSqrtF =
          Intrinsic::getDeclaration(&M, llvm::Intrinsic::sqrt, Ty);
    auto *NewSqrt = Builder.CreateCall(NewSqrtF, { Sqrt->getOperand(0) },
                                       Sqrt->getName());

    // Copy FP flags, metadata and attributes. Replace old call with a new call.
    Attrs = Attrs.removeFnAttribute(Sqrt->getContext(), "fpbuiltin-max-error");
    NewSqrt->setAttributes(Attrs);
    NewSqrt->copyMetadata(*Sqrt);
    FPMathOperator *FPOp = cast<FPMathOperator>(Sqrt);
    FastMathFlags FMF = FPOp->getFastMathFlags();
    NewSqrt->setFastMathFlags(FMF);
    Sqrt->replaceAllUsesWith(NewSqrt);
    Sqrt->dropAllReferences();
    Sqrt->eraseFromParent();
  }

  // Replace @llvm.fpbuiltin.fdiv call with fdiv.
  for (auto *FDiv : WorkListFDiv) {
    DeclToRemove.insert(FDiv->getCalledFunction());
    IRBuilder Builder(FDiv);
    Builder.SetInsertPoint(FDiv);
    Instruction *NewFDiv =
        cast<Instruction>(Builder.CreateFDiv(
              FDiv->getOperand(0), FDiv->getOperand(1), FDiv->getName()));

    // Copy FP flags and metadata. Replace old call with a new instruction.
    cast<Instruction>(NewFDiv)->copyMetadata(*FDiv);
    FPMathOperator *FPOp = cast<FPMathOperator>(FDiv);
    FastMathFlags FMF = FPOp->getFastMathFlags();
    NewFDiv->setFastMathFlags(FMF);
    FDiv->replaceAllUsesWith(NewFDiv);
    FDiv->dropAllReferences();
    FDiv->eraseFromParent();
  }

  // Clear old declarations.
  for (auto *Decl : DeclToRemove) {
    assert(Decl->isDeclaration() &&
           "attempting to remove a function definition");
    Decl->dropAllReferences();
    Decl->eraseFromParent();
  }

  return PreservedAnalyses::all();
}
