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
#include "llvm/IR/Instructions.h"

using namespace llvm;

PreservedAnalyses SYCLConditionalCallOnDevicePass::run(Module &M,
                                                   ModuleAnalysisManager &) {
  // find call_if_on_device_conditionally function
  SmallVector<Function *, 4> FCallers;
  for (Function &F : M.functions()) {
    if (F.getName().contains("call_if_on_device_conditionally_helper") ||
        !F.getName().contains("call_if_on_device_conditionally"))
      continue;

    FCallers.push_back(&F);
  }

  int FCallerIndex = 0;
  for (Function *FCaller : FCallers) {
    // Find call to @CallableXXX in call_if_on_device_conditionally function
    // (FAction). FAction should be a literal (i.e. not a pointer)
    Function *FAction = nullptr;
    for (BasicBlock &BB : *FCaller) {
      for (Instruction &I : BB) {
        if (CallInst *callInst = dyn_cast<CallInst>(&I)) {
          Value *calledValue = callInst->getCalledOperand();
          FAction = dyn_cast<Function>(calledValue);
        }
      }
    }

    if (!FAction)
      continue;

    // Create a new function type with an additional function pointer argument
    std::vector<Type *> NewParamTypes;
    Type *FActionType = FAction->getType();
    NewParamTypes.push_back(
        PointerType::getUnqual(FActionType)); // Add function pointer to FAction
    FunctionType *OldFCallerType = FCaller->getFunctionType();
    for (Type *Ty : OldFCallerType->params()) {
      NewParamTypes.push_back(Ty);
    }

    FunctionType *NewFCallerType =
        FunctionType::get(OldFCallerType->getReturnType(), NewParamTypes,
                          OldFCallerType->isVarArg());

    // Create a new function with the updated type and rename it to
    // call_if_on_device_conditionally_GUID_N
    Twine NewFCallerName = Twine(FCaller->getName()) + "_" + UniquePrefix +
                           "_" + Twine(FCallerIndex);
    // Also change to external linkage
    Function *NewFCaller = Function::Create(
        NewFCallerType, Function::ExternalLinkage, NewFCallerName, &M);

    NewFCaller->setCallingConv(FCaller->getCallingConv());

    // Replace all calls to the old function with the new one
    for (auto &U : FCaller->uses()) {
      if (auto *Call = dyn_cast<CallInst>(U.getUser())) {
        std::vector<Value *> Args;
        Args.push_back(
            FAction); // Add the function pointer as the first argument
        for (unsigned i = 0; i < Call->arg_size(); ++i) {
          Args.push_back(Call->getArgOperand(i));
        }

        // Create the new call instruction
        CallInst *NewCall =
            CallInst::Create(NewFCaller, Args, /*	NameStr = */"", Call);
        NewCall->setCallingConv(Call->getCallingConv());
        NewCall->setDebugLoc(Call->getDebugLoc());

        // Replace the old call with the new call
        Call->replaceAllUsesWith(NewCall);
        Call->eraseFromParent();
      }
    }

    // Remove the body of the new function
    NewFCaller->deleteBody();

    // Remove the old function from the module
    FCaller->eraseFromParent();

    FCallerIndex++;
  }

  return PreservedAnalyses::all();
}
