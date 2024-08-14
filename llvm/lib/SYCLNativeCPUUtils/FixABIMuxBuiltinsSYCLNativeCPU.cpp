//===-- FixABIMuxBuiltinsSYCLNativeCPU.cpp - Fixup mux ABI issues       ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fixes up the ABI for any mux builtins which meet the architecture
// ABI but not the mux_* usage. For now this is restricted to mux_shuffle*
// builtins which take a float2 input.
//
//===----------------------------------------------------------------------===//

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/SYCLLowerIR/FixABIMuxBuiltinsSYCLNativeCPU.h>

#define DEBUG_TYPE "fix-abi-mux-builtins"

using namespace llvm;

PreservedAnalyses FixABIMuxBuiltinsPass::run(Module &M,
                                             ModuleAnalysisManager &AM) {
  bool Changed = false;

  // Decide if a function needs updated and if so what parameters need changing,
  // as well as the return value
  auto functionNeedsFixing =
      [&M](
          Function &F,
          llvm::SmallVectorImpl<std::pair<unsigned int, llvm::Type *>> &updates,
          llvm::Type *&RetVal) {
        updates.clear();
        // At the moment only cover sub groups
        if (F.getName().starts_with("__mux_sub_group")) {
          RetVal = F.getReturnType();
          // TODO: Support a richer set of fixing parameters by comparing the
          // name against expected parameter and building up replacements.
          bool Ret = false;
          unsigned int ArgIndex = 0;
          // Check if the mux function expects 2x vector i32. We know if this
          // has double parameters they need fixed up. For other cases if there
          // are byval parameters they need fixed up too.
          bool IsV2I32 = F.getName().ends_with("_v2i32");
          for (auto &Arg : F.args()) {
            if (IsV2I32 && Arg.getType()->isDoubleTy()) {
              auto *ReplType = llvm::FixedVectorType::get(
                  llvm::Type::getInt32Ty(F.getContext()), 2);
              updates.push_back(
                  std::pair<unsigned int, llvm::Type *>(ArgIndex, ReplType));
              RetVal = ReplType;
              Ret = true;
            } else if (Arg.hasByValAttr()) {
              updates.push_back(std::pair<unsigned int, llvm::Type *>(
                  ArgIndex, Arg.getParamByValType()));
              Ret = true;
            }
            ArgIndex++;
          }
          return Ret;
        }
        return false;
      };

  llvm::SmallVector<Function *, 4> FuncsToProcess;
  for (auto &F : M.functions()) {
    FuncsToProcess.push_back(&F);
  }

  for (auto *F : FuncsToProcess) {
    llvm::SmallVector<std::pair<unsigned int, llvm::Type *>, 4> ArgUpdates;
    llvm::Type *RetType = nullptr;
    if (!functionNeedsFixing(*F, ArgUpdates, RetType)) {
      continue;
    }
    if (!F->isDeclaration()) {
      continue;
    }
    Changed = true;
    IRBuilder<> ir(BasicBlock::Create(F->getContext(), "", F));

    std::string OrigName = F->getName().str();
    F->setName(F->getName() + "_abi_wrapper");
    llvm::SmallVector<Type *, 8> Args;
    unsigned int ArgIndex = 0;
    unsigned int UpdateIndex = 0;
    for (auto &Arg : F->args()) {
      if (UpdateIndex < ArgUpdates.size() &&
          std::get<0>(ArgUpdates[UpdateIndex]) == ArgIndex) {
        Args.push_back(std::get<1>(ArgUpdates[UpdateIndex]));
        UpdateIndex++;
      } else {
        Args.push_back(Arg.getType());
      }
      ArgIndex++;
    }

    FunctionType *FT = FunctionType::get(RetType, Args, false);
    Function *NewFunc = Function::Create(FT, F->getLinkage(), OrigName, M);
    llvm::SmallVector<Value *, 8> CallArgs;
    auto NewFuncArgItr = NewFunc->args().begin();
    for (auto &Arg : F->args()) {
      if (Arg.getType() != (*NewFuncArgItr).getType()) {
        if (Arg.hasByValAttr()) {
          Value *ArgLoad = ir.CreateLoad((*NewFuncArgItr).getType(), &Arg);
          CallArgs.push_back(ArgLoad);
        } else {
          Value *ArgCast = ir.CreateBitCast(&Arg, (*NewFuncArgItr).getType());
          CallArgs.push_back(ArgCast);
        }
      } else {
        CallArgs.push_back(&Arg);
      }
      NewFuncArgItr++;
    }

    Value *Res = ir.CreateCall(NewFunc, CallArgs);
    // If the return type is different to the initial function, then bitcast it.
    if (F->getReturnType() != RetType) {
      Res = ir.CreateBitCast(Res, F->getReturnType());
    }
    ir.CreateRet(Res);
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
