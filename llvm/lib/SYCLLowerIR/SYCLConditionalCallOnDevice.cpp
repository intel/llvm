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

    if (F.getName().contains("call_if_on_device_conditionally") &&
        !F.getName().contains("call_if_on_device_conditionally_helper"))
      FCallers.push_back(&F);
  }

  int FCallerIndex = 1;
  for (Function *FCaller : FCallers) {
    // Find call to @CallableXXX in call_if_on_device_conditionally function
    // (FAction). FAction should be a literal (i.e. not a pointer). The
    // structure of the header file ensures that there is exactly one such
    // instruction.
    Function *FAction = nullptr;
    for (Instruction &I : instructions(FCaller)) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        FAction = CI->getCalledFunction();
        break;
      }
    }

    if (!FAction)
      continue;

    // Create a new function type with an additional function pointer argument
    SmallVector<Type *, 4> NewParamTypes;
    Type *FActionType = FAction->getType();
    NewParamTypes.push_back(
        PointerType::getUnqual(FActionType)); // Add function pointer to FAction
    FunctionType *OldFCallerType = FCaller->getFunctionType();
    for (Type *Ty : OldFCallerType->params())
      NewParamTypes.push_back(Ty);

    FunctionType *NewFCallerType =
        FunctionType::get(OldFCallerType->getReturnType(), NewParamTypes,
                          OldFCallerType->isVarArg());

    // Create a new function with the updated type and rename it to
    // call_if_on_device_conditionally_GUID_N
    if (!UniquePrefixOpt.empty())
      UniquePrefix = UniquePrefixOpt;
    Twine NewFCallerName = Twine(FCaller->getName()) + "_" + UniquePrefix +
                           "_" + Twine(FCallerIndex);
    // Also change to external linkage
    Function *NewFCaller = Function::Create(
        NewFCallerType, Function::ExternalLinkage, NewFCallerName, &M);

    NewFCaller->setCallingConv(FCaller->getCallingConv());

    // Replace all calls to the old function with the new one
    for (auto &U : FCaller->uses()) {
      if (auto *Call = dyn_cast<CallInst>(U.getUser())) {
        SmallVector<Value *, 4> Args;
        // Add the function pointer as the first argument
        Args.push_back(FAction);
        for (unsigned i = 0; i < Call->arg_size(); ++i)
          Args.push_back(Call->getArgOperand(i));

        // Create the new call instruction
        auto *NewCall =
            CallInst::Create(NewFCaller, Args, /*	NameStr = */ "", Call);
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

  return PreservedAnalyses::none();
}
