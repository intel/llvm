//===------ PrepareSYCLNativeCPU.cpp - SYCL Native CPU Preparation Pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prepares the kernel for SYCL Native CPU:
// * Handles kernel calling convention and attributes.
// * Materializes spirv buitlins.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/PrepareSYCLNativeCPU.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <functional>
#include <numeric>
#include <vector>

using namespace llvm;

namespace {


void fixCallingConv(Function* F) {
  F->setCallingConv(llvm::CallingConv::C);
  // TODO: the frame-pointer=all attribute apparently makes the kernel crash at runtime
  F->setAttributes({});
}

// Clone the function and returns a new function with a new argument on type T added as 
// last argument
Function *cloneFunctionAndAddParam(Function *oldF, Type *T) {
  auto oldT = oldF->getFunctionType();
  auto retT = oldT->getReturnType();

  std::vector<Type *> args;
  for (auto arg : oldT->params()) {
    args.push_back(arg);
  }
  args.push_back(T);
  auto newT = FunctionType::get(retT, args, oldF->isVarArg());
  auto newF = Function::Create(newT, oldF->getLinkage(), oldF->getName(),
                               oldF->getParent());
  // Copy the old function's attributes
  newF->setAttributes(oldF->getAttributes());

  // Map old arguments to new arguments
  ValueToValueMapTy VMap;
  for (auto pair : llvm::zip(oldF->args(), newF->args())) {
    auto &oldA = std::get<0>(pair);
    auto &newA = std::get<1>(pair);
    VMap[&oldA] = &newA;
  }

  SmallVector<ReturnInst *, 1> ReturnInst;
  if (!oldF->isDeclaration())
    CloneFunctionInto(newF, oldF, VMap,
                      CloneFunctionChangeType::LocalChangesOnly, ReturnInst);
  return newF;
}

static std::map<std::string, std::string> BuiltinNamesMap{
    {"__spirv_BuiltInGlobalInvocationId", "_Z13get_global_idmP15nativecpu_state"}};

Function *getReplaceFunc(Module &M, Type *T, StringRef Name) {
  Function *F = M.getFunction(Name);
  assert(F && "Error retrieving replace function");
  return F;
}

Value *getStateArg(Function *F) {
  auto F_t = F->getFunctionType();
  return F->getArg(F_t->getNumParams() - 1);
}

SmallVector<Function *> getFunctionsFromUse(Use &U) {
  // This function returns a vector since an operator may be used by
  // instructions in multiple functions
  User *Usr = U.getUser();
  if (auto I = dyn_cast<Instruction>(Usr)) {
    return {I->getFunction()};
  }
  if (auto Op = dyn_cast<Operator>(Usr)) {
    SmallVector<Function *> Res;
    for (auto &Use : Op->uses()) {
      Res.push_back(dyn_cast<Instruction>(Use.getUser())->getFunction());
    }
    return Res;
  }
  return {};
}

} // namespace

PreservedAnalyses PrepareSYCLNativeCPUPass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  SmallVector<Function *> OldKernels;
  for (auto &F : M) {
    if (F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL)
      OldKernels.push_back(&F);
  }
  if(OldKernels.empty())
    return PreservedAnalyses::all();

  // Materialize builtins
  // First we add a pointer to the Native CPU state as arg to all the
  // kernels.
  Type *StateType = StructType::getTypeByName(M.getContext(), "struct.nativecpu_state");
  if (!StateType)
    report_fatal_error("Couldn't find the Native CPU state in the "
                       "module, make sure that -D __SYCL_NATIVE_CPU__ is set",
                       false);
  Type *StatePtrType = PointerType::getUnqual(StateType);
  SmallVector<Function *> NewKernels;
  for (auto &oldF : OldKernels) {
    auto newF = addArg(oldF, StatePtrType);
    newF->takeName(oldF);
    oldF->eraseFromParent();
    NewKernels.push_back(newF);
    ModuleChanged |= true;
  }

  // Then we iterate over all the supported builtins, find their uses and
  // replace them with calls to our Native CPU functions.
  for (auto &entry : BuiltinNamesMap) {
    // Kernel -> builin materialization CallInst, this is used to avoid
    // inserting multiple calls to the same builtin
    std::map<Function *, CallInst *> BuiltinCallMap;
    // Map that associates to each User of a builtin, the index of the builtin
    // in its operand list, and the callinst that will replace the builtin
    std::map<User *, std::pair<unsigned, CallInst *>> ToReplace;
    // We need to handle GEPOperator uses in a separate case since they are
    // constants
    std::set<GEPOperator *> GEPOps;
    // spirv builtins are global constants, find it in the module
    auto Glob = M.getNamedGlobal(entry.first);
    if (!Glob)
      continue;
    auto replaceFunc = getReplaceFunc(M, StatePtrType, entry.second);
    for (auto &Use : Glob->uses()) {
      auto Funcs = getFunctionsFromUse(Use);
      if (Funcs.empty()) {
        // todo: use without a parent function?
        continue;
      }
      for (auto &F : Funcs) {
        auto NewCall_it = BuiltinCallMap.find(F);
        CallInst *NewCall;
        // check if we already inserted a call to our function
        if (NewCall_it != BuiltinCallMap.end()) {
          NewCall = NewCall_it->second;
        } else {
          auto StateArg = getStateArg(F);
          NewCall = llvm::CallInst::Create(
              replaceFunc->getFunctionType(), replaceFunc, {StateArg},
              "ncpu_builtin", F->getEntryBlock().getFirstNonPHI());
          BuiltinCallMap.insert({F, NewCall});
        }
        User *Usr = Use.getUser();
        if (auto GEPOp = dyn_cast<GEPOperator>(Usr)) {
          GEPOps.insert(GEPOp);
        } else {
          // Find the index of the builtin in the user's operand list
          // We are guaranteed to find it since we are already iterating over
          // the builtin's uses.
          bool Found = false;
          unsigned Index = 0;
          for (unsigned I = 0; I < Usr->getNumOperands() && !Found; I++) {
            if (Usr->getOperand(I) == Glob) {
              Found = true;
              Index = I;
            }
          }
          assert(Found && "Unable to find builtin in operand list");
          ToReplace.insert({Usr, {Index, NewCall}});
        }
      }
    }

    // Handle the non-constant builtin uses, simply replace the builtin with the
    // return value of our function call
    for (auto &Entry : ToReplace) {
      unsigned Index = Entry.second.first;
      CallInst *NewCall = Entry.second.second;
      User *Usr = Entry.first;
      Usr->setOperand(Index, NewCall);
    }

    // Handle the constant builtin uses, we insert a non-constant GEP
    // instruction that uses the return value of our function call, and replaces
    // the original GEPOperator
    SmallVector<std::tuple<Operator *, User *, GetElementPtrInst *>>
        GEPReplaceMap;
    for (auto &OldOp : GEPOps) {
      SmallVector<Value *> Indices(OldOp->idx_begin(), OldOp->idx_end());
      for (auto &OpUse : OldOp->uses()) {
        User *Usr = OpUse.getUser();
        Instruction *I = dyn_cast<Instruction>(Usr);
        auto NewCall = BuiltinCallMap[I->getFunction()];
        GetElementPtrInst *NewGEP = GetElementPtrInst::Create(
            OldOp->getSourceElementType(), NewCall, Indices, "ncpu_gep", I);
        GEPReplaceMap.emplace_back(OldOp, Usr, NewGEP);
      }
    }
    for (auto &Entry : GEPReplaceMap) {
      auto Op = std::get<0>(Entry);
      auto Usr = std::get<1>(Entry);
      auto NewGEP = std::get<2>(Entry);
      Op->replaceUsesWithIf(NewGEP, [&](Use &U) {
        bool res = U.getUser() == Usr;
        return res;
      });
    }

    // Finally, we erase the builtin from the module
    Glob->eraseFromParent();
  }

  for (auto F : NewKernels) {
    fixCallingConv(F);
  }
  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
