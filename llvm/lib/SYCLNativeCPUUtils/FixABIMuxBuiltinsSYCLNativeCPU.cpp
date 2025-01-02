//===-- FixABIMuxBuiltinsSYCLNativeCPU.cpp - Fixup mux ABI issues       ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Creates calls to shuffle up/down/xor mux builtins taking into account ABI of
// the SYCL functions. For now this only is used for vector variants.
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
  auto FunctionNeedsFixing =
      [](Function &F,
         llvm::SmallVectorImpl<std::pair<unsigned int, llvm::Type *>> &Updates,
         llvm::Type *&RetVal, std::string &MuxFuncNameToCall) {
        if (!F.isDeclaration()) {
          return false;
        }
        if (!F.getName().contains("__spirv_SubgroupShuffle")) {
          return false;
        }
        Updates.clear();
        auto LIDvPos = F.getName().find("ELIDv");
        llvm::StringRef NameToMatch;
        if (LIDvPos != llvm::StringRef::npos) {
          // Add sizeof ELIDv to get num characters to match against
          NameToMatch = F.getName().take_front(LIDvPos + 5);
        } else {
          return false;
        }

        unsigned int StartIdx = 0;
        unsigned int EndIdx = 1;
        if (NameToMatch == "_Z32__spirv_SubgroupShuffleDownINTELIDv") {
          MuxFuncNameToCall = "__mux_sub_group_shuffle_down_";
        } else if (NameToMatch == "_Z30__spirv_SubgroupShuffleUpINTELIDv") {
          MuxFuncNameToCall = "__mux_sub_group_shuffle_up_";
        } else if (NameToMatch == "_Z28__spirv_SubgroupShuffleINTELIDv") {
          MuxFuncNameToCall = "__mux_sub_group_shuffle_";
          EndIdx = 0;
        } else if (NameToMatch == "_Z31__spirv_SubgroupShuffleXorINTELIDv") {
          MuxFuncNameToCall = "__mux_sub_group_shuffle_xor_";
          EndIdx = 0;
        } else {
          return false;
        }

        // We need to create the body for this. First we need to find out what
        // the first arguments should be
        llvm::StringRef RemainingName =
            F.getName().drop_front(NameToMatch.size());
        std::string MuxFuncTypeStr = "UNKNOWN";

        unsigned int VecWidth = 0;
        if (RemainingName.consumeInteger(10, VecWidth)) {
          return false;
        }
        if (!RemainingName.consume_front("_")) {
          return false;
        }

        char TypeCh = RemainingName[0];
        Type *BaseType = nullptr;
        switch (TypeCh) {
        case 'a':
        case 'h':
          BaseType = llvm::Type::getInt8Ty(F.getContext());
          MuxFuncTypeStr = "i8";
          break;
        case 's':
        case 't':
          BaseType = llvm::Type::getInt16Ty(F.getContext());
          MuxFuncTypeStr = "i16";
          break;

        case 'i':
        case 'j':
          BaseType = llvm::Type::getInt32Ty(F.getContext());
          MuxFuncTypeStr = "i32";
          break;
        case 'l':
        case 'm':
          BaseType = llvm::Type::getInt64Ty(F.getContext());
          MuxFuncTypeStr = "i64";
          break;
        case 'f':
          BaseType = llvm::Type::getFloatTy(F.getContext());
          MuxFuncTypeStr = "f32";
          break;
        case 'd':
          BaseType = llvm::Type::getDoubleTy(F.getContext());
          MuxFuncTypeStr = "f64";
          break;
        default:
          return false;
        }
        auto *VecType = llvm::FixedVectorType::get(BaseType, VecWidth);
        RetVal = VecType;

        // Work out the mux function to call's type extension based on v##N##Sfx
        MuxFuncNameToCall += "v";
        MuxFuncNameToCall += std::to_string(VecWidth);
        MuxFuncNameToCall += MuxFuncTypeStr;

        unsigned int CurrentIndex = 0;
        for (auto &Arg : F.args()) {
          if (Arg.hasStructRetAttr()) {
            StartIdx++;
            EndIdx++;
          } else {
            if (CurrentIndex >= StartIdx && CurrentIndex <= EndIdx) {
              if (Arg.getType() != VecType) {
                Updates.push_back(std::pair<unsigned int, llvm::Type *>(
                    CurrentIndex, VecType));
              }
            }
          }
          CurrentIndex++;
        }
        return true;
      };

  llvm::SmallVector<Function *, 4> FuncsToProcess;
  for (auto &F : M.functions()) {
    FuncsToProcess.push_back(&F);
  }

  for (auto *F : FuncsToProcess) {
    llvm::SmallVector<std::pair<unsigned int, llvm::Type *>, 4> ArgUpdates;
    llvm::Type *RetType = nullptr;
    std::string MuxFuncNameToCall;
    if (!FunctionNeedsFixing(*F, ArgUpdates, RetType, MuxFuncNameToCall)) {
      continue;
    }
    if (!F->isDeclaration()) {
      continue;
    }
    Changed = true;
    IRBuilder<> IR(BasicBlock::Create(F->getContext(), "", F));

    llvm::SmallVector<Type *, 8> Args;
    unsigned int ArgIndex = 0;
    unsigned int UpdateIndex = 0;

    for (auto &Arg : F->args()) {
      if (!Arg.hasStructRetAttr()) {
        if (UpdateIndex < ArgUpdates.size() &&
            std::get<0>(ArgUpdates[UpdateIndex]) == ArgIndex) {
          Args.push_back(std::get<1>(ArgUpdates[UpdateIndex]));
          UpdateIndex++;
        } else {
          Args.push_back(Arg.getType());
        }
      }
      ArgIndex++;
    }

    FunctionType *FT = FunctionType::get(RetType, Args, false);
    Function *NewFunc =
        Function::Create(FT, F->getLinkage(), MuxFuncNameToCall, M);
    llvm::SmallVector<Value *, 8> CallArgs;
    auto NewFuncArgItr = NewFunc->args().begin();
    Argument *SretPtr = nullptr;
    for (auto &Arg : F->args()) {
      if (Arg.hasStructRetAttr()) {
        SretPtr = &Arg;
      } else {
        if (Arg.getType() != (*NewFuncArgItr).getType()) {
          if (Arg.getType()->isPointerTy()) {
            Value *ArgLoad = IR.CreateLoad((*NewFuncArgItr).getType(), &Arg);
            CallArgs.push_back(ArgLoad);
          } else {
            Value *ArgCast = IR.CreateBitCast(&Arg, (*NewFuncArgItr).getType());
            CallArgs.push_back(ArgCast);
          }
        } else {
          CallArgs.push_back(&Arg);
        }
        NewFuncArgItr++;
      }
    }

    Value *Res = IR.CreateCall(NewFunc, CallArgs);
    // If the return type is different to the initial function, then bitcast it
    // unless it's void in which case we'd expect an StructRet parameter which
    // needs stored to.
    if (F->getReturnType() != RetType) {
      if (F->getReturnType()->isVoidTy()) {
        // If we don't have an StructRet parameter then something is wrong with
        // the initial function
        if (!SretPtr) {
          llvm_unreachable(
              "No struct ret pointer for Sub group shuffle function");
        }

        IR.CreateStore(Res, SretPtr);
      } else {
        Res = IR.CreateBitCast(Res, F->getReturnType());
      }
    }
    if (F->getReturnType()->isVoidTy()) {
      IR.CreateRetVoid();
    } else {
      IR.CreateRet(Res);
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
