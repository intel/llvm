//===-- SYCLConditionalCallOnDevice.cpp - SYCLConditionalCallOnDevice Pass
//--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass performs transformations on functions which represent the conditional
// call to application's callable object. The conditional call is based on the
// SYCL device's aspects or architecture passed to the functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLConditionalCallOnDevice.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

cl::opt<std::string>
    UniquePrefixOpt("sycl-conditional-call-on-device-unique-prefix",
                    cl::Optional, cl::Hidden,
                    cl::desc("Set unique prefix for a translation unit, "
                             "required for funtions with external linkage"),
                    cl::init(""));

PreservedAnalyses
SYCLConditionalCallOnDevicePass::run(Module &M, ModuleAnalysisManager &) {
  // find call_if_on_device_conditionally function
  SmallVector<Function *, 4> FCallers;
  for (Function &F : M.functions()) {
    if (F.isDeclaration())
      continue;

    if (CallingConv::SPIR_KERNEL == F.getCallingConv())
      continue;

    if (F.hasFnAttribute("sycl-call-if-on-device-conditionally"))
      FCallers.push_back(&F);
  }

  // A vector instead of DenseMap to make LIT tests predictable
  SmallVector<std::pair<Function *, Function *>, 8> FCallersToFActions;
  for (Function *FCaller : FCallers) {
    // Find call to @CallableXXX in call_if_on_device_conditionally function
    // (FAction). FAction should be a literal (i.e. not a pointer). The
    // structure of the header file ensures that there is exactly one such
    // instruction.
    bool CallFound = false;
    for (Instruction &I : instructions(FCaller)) {
      if (auto *CI = dyn_cast<CallInst>(&I);
          CI && (Intrinsic::IndependentIntrinsics::not_intrinsic ==
                 CI->getIntrinsicID())) {
        assert(
            !CallFound &&
            "The call_if_on_device_conditionally function must have only one "
            "call instruction (w/o taking into account any calls to various "
            "intrinsics). More than one found.");
        FCallersToFActions.push_back(
            std::make_pair(FCaller, CI->getCalledFunction()));
        CallFound = true;
      }
    }
    assert(CallFound &&
           "The call_if_on_device_conditionally function must have a "
           "call instruction (w/o taking into account any calls to various "
           "intrinsics). Call not found.");
  }

  int FCallerIndex = 1;
  for (const auto &FCallerToFAction : FCallersToFActions) {
    Function *FCaller = FCallerToFAction.first;
    Function *FAction = FCallerToFAction.second;

    // Create a new function type with an additional function pointer argument
    SmallVector<Type *, 4> NewParamTypes;
    Type *FActionType = FAction->getType();
    NewParamTypes.push_back(
        PointerType::getUnqual(FActionType)); // Add function pointer to FAction
    FunctionType *OldFCallerType = FCaller->getFunctionType();
    for (Type *Ty : OldFCallerType->params())
      NewParamTypes.push_back(Ty);

    auto *NewFCallerType =
        FunctionType::get(OldFCallerType->getReturnType(), NewParamTypes,
                          OldFCallerType->isVarArg());

    // Create a new function with the updated type and rename it to
    // call_if_on_device_conditionally_GUID_N
    if (!UniquePrefixOpt.empty())
      UniquePrefix = UniquePrefixOpt;
    // Also change to external linkage
    auto *NewFCaller =
        Function::Create(NewFCallerType, Function::ExternalLinkage,
                         Twine(FCaller->getName()) + "_" + UniquePrefix + "_" +
                             Twine(FCallerIndex),
                         &M);

    NewFCaller->setCallingConv(FCaller->getCallingConv());

    DenseMap<CallInst *, CallInst *> OldCallsToNewCalls;

    // Replace all calls to the old function with the new one
    for (auto &U : FCaller->uses()) {
      auto *Call = dyn_cast<CallInst>(U.getUser());

      if (!Call)
        continue;

      SmallVector<Value *, 4> Args;
      // Add the function pointer as the first argument
      Args.push_back(FAction);
      for (unsigned I = 0; I < Call->arg_size(); ++I)
        Args.push_back(Call->getArgOperand(I));

      // Create the new call instruction
      auto *NewCall =
          CallInst::Create(NewFCaller, Args, /*	NameStr = */ "", Call);
      NewCall->setCallingConv(Call->getCallingConv());
      NewCall->setDebugLoc(Call->getDebugLoc());

      OldCallsToNewCalls[Call] = NewCall;
    }

    for (const auto &OldCallToNewCall : OldCallsToNewCalls) {
      auto *OldCall = OldCallToNewCall.first;
      auto *NewCall = OldCallToNewCall.second;

      // Replace the old call with the new call
      OldCall->replaceAllUsesWith(NewCall);
      OldCall->eraseFromParent();
    }

    // Remove the body of the new function
    NewFCaller->deleteBody();

    // Remove the old function from the module
    FCaller->eraseFromParent();

    FCallerIndex++;
  }

  return PreservedAnalyses::none();
}
