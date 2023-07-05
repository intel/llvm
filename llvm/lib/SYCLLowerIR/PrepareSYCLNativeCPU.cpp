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
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <functional>
#include <numeric>
#include <set>
#include <vector>

using namespace llvm;

namespace {

void fixCallingConv(Function *F) {
  F->setCallingConv(llvm::CallingConv::C);
  // The frame-pointer=all and the "byval" attributes lead to code generation
  // that conflicts with the Kernel declaration that we emit in the Native CPU
  // helper header (in which all the kernel argument are void* or scalars).
  auto AttList = F->getAttributes();
  for (unsigned ArgNo = 0; ArgNo < F->getFunctionType()->getNumParams();
       ArgNo++) {
    if (AttList.hasParamAttr(ArgNo, Attribute::AttrKind::ByVal)) {
      AttList = AttList.removeParamAttribute(F->getContext(), ArgNo,
                                             Attribute::AttrKind::ByVal);
    }
  }
  F->setAttributes(AttList);
  F->addFnAttr("frame-pointer", "none");
  if (!F->isDeclaration())
    F->setLinkage(GlobalValue::LinkageTypes::WeakAnyLinkage);
}

// Clone the function and returns a new function with a new argument on type T
// added as last argument
Function *cloneFunctionAndAddParam(Function *OldF, Type *T) {
  auto *OldT = OldF->getFunctionType();
  auto *RetT = OldT->getReturnType();

  std::vector<Type *> Args;
  for (auto *Arg : OldT->params()) {
    Args.push_back(Arg);
  }
  Args.push_back(T);
  auto *NewT = FunctionType::get(RetT, Args, OldF->isVarArg());
  auto *NewF = Function::Create(NewT, OldF->getLinkage(), OldF->getName(),
                                OldF->getParent());
  // Copy the old function's attributes
  NewF->setAttributes(OldF->getAttributes());

  // Map old arguments to new arguments
  ValueToValueMapTy VMap;
  for (auto Pair : llvm::zip(OldF->args(), NewF->args())) {
    auto &OldA = std::get<0>(Pair);
    auto &NewA = std::get<1>(Pair);
    VMap[&OldA] = &NewA;
  }

  SmallVector<ReturnInst *, 1> ReturnInst;
  if (!OldF->isDeclaration())
    CloneFunctionInto(NewF, OldF, VMap,
                      CloneFunctionChangeType::LocalChangesOnly, ReturnInst);
  return NewF;
}

// Todo: add support for more SPIRV builtins here
static std::map<std::string, std::string> BuiltinNamesMap{
    {"__spirv_BuiltInGlobalInvocationId", "__dpcpp_nativecpu_global_id"},
    {"__spirv_BuiltInGlobalSize", "__dpcpp_nativecpu_global_range"},
    {"__spirv_BuiltInWorkgroupSize", "__dpcpp_nativecpu_get_wg_size"},
    {"__spirv_BuiltInWorkgroupId", "__dpcpp_nativecpu_get_wg_id"},
    {"__spirv_BuiltInLocalInvocationId", "__dpcpp_nativecpu_get_local_id"},
    {"__spirv_BuiltInNumWorkgroups", "__dpcpp_nativecpu_get_num_groups"},
    {"__spirv_BuiltInGlobalOffset", "__dpcpp_nativecpu_get_global_offset"}};

Function *getReplaceFunc(Module &M, Type *T, StringRef Name) {
  Function *F = M.getFunction(Name);
  assert(F && "Error retrieving replace function");
  return F;
}

Value *getStateArg(const Function *F) {
  auto *FT = F->getFunctionType();
  return F->getArg(FT->getNumParams() - 1);
}

SmallVector<Function *> getFunctionsFromUse(Use &U) {
  // This function returns a vector since an operator may be used by
  // instructions in multiple functions
  User *Usr = U.getUser();
  if (auto *I = dyn_cast<Instruction>(Usr)) {
    if (I->getParent())
      return {I->getFunction()};
  }
  if (auto *Op = dyn_cast<Operator>(Usr)) {
    SmallVector<Function *> Res;
    for (auto &Use : Op->uses()) {
      if (auto *I = dyn_cast<Instruction>(Use.getUser())) {
        if (I->getParent())
          Res.push_back(I->getFunction());
      }
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

  // Materialize builtins
  // First we add a pointer to the Native CPU state as arg to all the
  // kernels.
  Type *StateType =
      StructType::getTypeByName(M.getContext(), "struct.__nativecpu_state");
  if (!StateType)
    report_fatal_error("Couldn't find the Native CPU state in the "
                       "module, make sure that -D __SYCL_NATIVE_CPU__ is set",
                       false);
  Type *StatePtrType = PointerType::getUnqual(StateType);
  SmallVector<Function *> NewKernels;
  for (auto &OldF : OldKernels) {
    auto *NewF = cloneFunctionAndAddParam(OldF, StatePtrType);
    NewF->takeName(OldF);
    OldF->eraseFromParent();
    NewKernels.push_back(NewF);
    ModuleChanged |= true;
  }

  // Then we iterate over all the supported builtins, find their uses and
  // replace them with calls to our Native CPU functions.
  for (auto &Entry : BuiltinNamesMap) {
    // Kernel -> builtin materialization CallInst, this is used to avoid
    // inserting multiple calls to the same builtin
    std::map<Function *, CallInst *> BuiltinCallMap;
    // Map that associates to each User of a builtin, the index of the builtin
    // in its operand list, and the callinst that will replace the builtin
    std::map<User *, std::pair<unsigned, CallInst *>> ToReplace;
    // We need to handle GEPOperator uses in a separate case since they are
    // constants
    std::set<GEPOperator *> GEPOps;
    // spirv builtins are global constants, find it in the module
    auto *Glob = M.getNamedGlobal(Entry.first);
    if (!Glob)
      continue;
    auto *ReplaceFunc = getReplaceFunc(M, StatePtrType, Entry.second);
    for (auto &Use : Glob->uses()) {
      auto Funcs = getFunctionsFromUse(Use);
      // Here we check that the use comes from a kernel function
      // Todo: remove this check once this pass supports non-optimized modules
      for (auto &Func : Funcs) {
        if (!(Func->getCallingConv() == CallingConv::SPIR_KERNEL))
          report_fatal_error("SYCL Native CPU currently supports only "
                             "optimized modules, please enable optimizations "
                             "and eventually increase the inlining threshold",
                             false);
      }
      if (Funcs.empty()) {
        // todo: use without a parent function?
        continue;
      }
      for (auto &F : Funcs) {
        auto NewCallIt = BuiltinCallMap.find(F);
        CallInst *NewCall;
        // check if we already inserted a call to our function
        if (NewCallIt != BuiltinCallMap.end()) {
          NewCall = NewCallIt->second;
        } else {
          auto *StateArg = getStateArg(F);
          NewCall = llvm::CallInst::Create(
              ReplaceFunc->getFunctionType(), ReplaceFunc, {StateArg},
              "ncpu_builtin", F->getEntryBlock().getFirstNonPHI());
          BuiltinCallMap.insert({F, NewCall});
        }
        User *Usr = Use.getUser();
        if (auto *GEPOp = dyn_cast<GEPOperator>(Usr)) {
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
        if (!I) {
          continue;
        }
        auto *NewCall = BuiltinCallMap[I->getFunction()];
        auto *ArrayT = ArrayType::get(Type::getInt64Ty(M.getContext()), 3);
        GetElementPtrInst *NewGEP =
            GetElementPtrInst::Create(ArrayT, NewCall, Indices, "ncpu_gep", I);
        GEPReplaceMap.emplace_back(OldOp, Usr, NewGEP);
      }
    }
    for (auto &Entry : GEPReplaceMap) {
      auto *Op = std::get<0>(Entry);
      auto *Usr = std::get<1>(Entry);
      auto *NewGEP = std::get<2>(Entry);
      Op->replaceUsesWithIf(NewGEP, [&](Use &U) {
        bool Res = U.getUser() == Usr;
        return Res;
      });
    }

    // Finally, we erase the builtin from the module
    Glob->eraseFromParent();
  }

  for (auto &F : M) {
    fixCallingConv(&F);
  }
  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
