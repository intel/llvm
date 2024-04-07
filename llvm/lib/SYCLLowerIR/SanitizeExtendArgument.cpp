//===-- SanitizeExtendArgument.cpp - Append "__asan_launch" for sanitizer -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SanitizeExtendArgument.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/IRBuilder.h"

#define DEBUG_TYPE "SanitizeExtendArgument"

using namespace llvm;

namespace {

constexpr unsigned kSpirOffloadGlobalAS = 1;

bool isSpirFuncNeedFixup(Function &F, const TargetLibraryInfo &TLI) {
  if (F.empty())
    return false;
  if (F.getLinkage() == GlobalValue::AvailableExternallyLinkage)
    return false;

  // Leave if the function doesn't need instrumentation.
  if (!F.hasFnAttribute(Attribute::SanitizeAddress))
    return false;

  if (F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation))
    return false;

  if (F.isDeclaration())
    return false;

  if (F.getFunctionType()->isVarArg())
    return false;

  if (F.doesNotAccessMemory())
    return false;

  auto FuncName = F.getName();

  if (FuncName.starts_with("__asan_"))
    return false;

  if (FuncName.contains("__spirv_"))
    return false;

  if (FuncName.contains("__sycl_"))
    return false;

  if (FuncName.equals("_Z12get_local_idj") ||
      FuncName.equals("_Z14get_local_sizej") ||
      FuncName.equals("_Z14_get_num_groupsj"))
    return false;

  {
    LibFunc LF;
    if (TLI.getLibFunc(F, LF))
      return false;
  }

  return true;
}

static bool extendKernelArg(Module &M, FunctionAnalysisManager &FAM) {
  SmallVector<Function *> SpirFixupFuncs;
  for (Function &F : M) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      SpirFixupFuncs.emplace_back(&F);
    } else if (F.getCallingConv() == CallingConv::SPIR_FUNC) {
      const TargetLibraryInfo &TLI = FAM.getResult<TargetLibraryAnalysis>(F);
      if (isSpirFuncNeedFixup(F, TLI)) {
        SpirFixupFuncs.emplace_back(&F);
      }
    }
  }

  SmallVector<std::pair<Function *, Function *>> SpirFuncs;
  const int LongSize = M.getDataLayout().getPointerSizeInBits();
  auto *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);

  for (auto *F : SpirFixupFuncs) {
    SmallVector<Type *, 16> Types;
    for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I) {
      Types.push_back(I->getType());
    }

    // New argument type: uintptr_t as(1)*, as it's allocated in USM buffer,
    // and it can also be treated as a pointer point to the base address of
    // private shadow memory
    Types.push_back(IntptrTy->getPointerTo(kSpirOffloadGlobalAS));

    FunctionType *NewFTy = FunctionType::get(F->getReturnType(), Types, false);

    std::string OrigFuncName = F->getName().str();
    F->setName(OrigFuncName + "_del");

    Function *NewF =
        Function::Create(NewFTy, F->getLinkage(), OrigFuncName, F->getParent());

    NewF->copyAttributesFrom(F);
    NewF->setCallingConv(F->getCallingConv());
    NewF->setDSOLocal(F->isDSOLocal());
    NewF->copyMetadata(F, 0);
    F->clearMetadata();

    // Set original arguments' names.
    Function::arg_iterator NewI = NewF->arg_begin();
    for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I, ++NewI) {
      NewI->setName(I->getName());
    }
    // New argument name
    NewI->setName("__asan_launch");

    NewF->splice(NewF->begin(), F);
    assert(F->isDeclaration() &&
           "splice does not work, original function body is not empty!");

    NewF->setSubprogram(F->getSubprogram());

    NewF->setComdat(F->getComdat());
    F->setComdat(nullptr);

    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(),
                                NI = NewF->arg_begin();
         I != E; ++I, ++NI) {
      I->replaceAllUsesWith(&*NI);
    }

    // Fixup metadata
    IRBuilder<> Builder(M.getContext());

    auto FixupMetadata = [&NewF](StringRef MDName, Constant *NewV) {
      auto *Node = NewF->getMetadata(MDName);
      if (!Node)
        return;
      SmallVector<Metadata *, 8> NewMD;
      for (unsigned I = 0; I < Node->getNumOperands(); ++I) {
        NewMD.emplace_back(Node->getOperand(I));
      }
      NewMD.emplace_back(ConstantAsMetadata::get(NewV));
      NewF->setMetadata(MDName, llvm::MDNode::get(NewF->getContext(), NewMD));
    };

    FixupMetadata("kernel_arg_buffer_location", Builder.getInt32(-1));
    FixupMetadata("kernel_arg_runtime_aligned", Builder.getFalse());
    FixupMetadata("kernel_arg_exclusive_ptr", Builder.getFalse());

    SpirFuncs.emplace_back(F, NewF);
  }

  // Fixup all users
  for (auto [F, NewF] : SpirFuncs) {
    SmallVector<User *, 16> Users(F->users());
    for (User *U : Users) {
      if (auto *GA = dyn_cast<GlobalAlias>(U)) {
        auto OriginalName = GA->getName();
        GA->setName(OriginalName + "_del");
        GlobalAlias *NewGA = GlobalAlias::create(OriginalName, NewF);
        NewGA->setUnnamedAddr(GA->getUnnamedAddr());
        NewGA->setVisibility(GA->getVisibility());
        GA->replaceAllUsesWith(NewGA);
        GA->eraseFromParent();
      } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        if (CE->getOpcode() == Instruction::AddrSpaceCast) {
          auto *NewCE = ConstantExpr::getAddrSpaceCast(NewF, CE->getType());
          CE->replaceAllUsesWith(NewCE);
        }
      } else if (auto *CI = dyn_cast<CallInst>(U)) {
        if (CI->getCalledOperand() == F) {
          // Append "launch_info" into arguments of call instruction
          SmallVector<Value *, 16> Args;
          for (unsigned I = 0, E = CI->arg_size(); I != E; ++I)
            Args.push_back(CI->getArgOperand(I));
          // "launch_info" is the last argument of current function
          auto *CurF = CI->getFunction();
          Args.push_back(CurF->getArg(CurF->arg_size() - 1));

          CallInst *NewCI = CallInst::Create(NewF, Args, CI->getName(), CI);
          NewCI->setCallingConv(CI->getCallingConv());
          NewCI->setAttributes(NewF->getAttributes());
          if (CI->hasMetadata()) {
            NewCI->setDebugLoc(CI->getDebugLoc());
          }
          CI->replaceAllUsesWith(NewCI);
          CI->eraseFromParent();
        }
      }
    }
    F->removeFromParent();
  }

  // Fixup all __asan_dummy_launch functions
  //   Replace all call instructions using "__asan_dummy_launch" as a parementer
  // with "__asan_launch"
  auto *AsanDummyLaunch = M.getNamedGlobal("__asan_dummy_launch");
  SmallVector<User *, 16> Users(AsanDummyLaunch->users());
  for (auto *U : Users) {
    auto *CI = dyn_cast<CallInst>(U);
    assert(CI && "__asan_dummy_launch must be used in call instruction");

    Argument *AsanLaunchArg;
    auto *CurF = CI->getFunction();
    if (CurF->arg_size()) {
      AsanLaunchArg = CurF->getArg(CurF->arg_size() - 1);
      if (AsanLaunchArg->getName() != "__asan_launch")
        continue;
    }

    SmallVector<Value *, 16> Args;
    for (unsigned I = 0, E = CI->arg_size(); I != E; ++I) {
      auto *Argument = CI->getArgOperand(I);
      if (Argument == AsanDummyLaunch) {
        Args.push_back(AsanLaunchArg);
      } else {
        Args.push_back(CI->getArgOperand(I));
      }
    }

    CallInst *NewCI =
        CallInst::Create(CI->getCalledFunction(), Args, CI->getName(), CI);
    NewCI->setCallingConv(CI->getCallingConv());
    NewCI->setAttributes(CI->getAttributes());
    if (CI->hasMetadata()) {
      NewCI->setDebugLoc(CI->getDebugLoc());
    }
    CI->replaceAllUsesWith(NewCI);
    CI->eraseFromParent();
  }
  assert(AsanDummyLaunch->getNumUses() == 0 &&
         "__asan_dummy_launch must have no uses");
  M.removeGlobalVariable(AsanDummyLaunch);

  return true;
}

} // namespace

namespace llvm {

PreservedAnalyses SanitizeExtendArgumentPass::run(Module &M,
                                                  ModuleAnalysisManager &MAM) {
  bool Modified = false;

  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  // Append a new argument "launch_data" to user's spir_kernel & spir_func
  Modified |= extendKernelArg(M, FAM);

  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace llvm
