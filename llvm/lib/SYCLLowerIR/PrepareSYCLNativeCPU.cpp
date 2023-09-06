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
#include "llvm/IR/Constant.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"

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
#include "llvm/IR/LLVMContext.h"
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

// returns the indexes of the used arguments
SmallVector<unsigned> getUsedIndexes(const Function *F) {
  SmallVector<unsigned> Res;
  auto UsedNode = F->getMetadata("sycl_kernel_omit_args");
  if (!UsedNode) {
    // the metadata node is not available if -fenable-sycl-dae
    // was not set; set everything to true
    // Exclude one arg because we already added the state ptr
    for (unsigned I = 0; I + 1 < F->getFunctionType()->getNumParams(); I++) {
      Res.push_back(I);
    }
    return Res;
  }
  auto NumOperands = UsedNode->getNumOperands();
  for (unsigned I = 0; I < NumOperands; I++) {
    auto &Op = UsedNode->getOperand(I);
    if (auto CAM = dyn_cast<ConstantAsMetadata>(Op.get())) {
      if (auto Const = dyn_cast<ConstantInt>(CAM->getValue())) {
        auto Val = Const->getValue();
        if (!Val.getBoolValue()) {
          Res.push_back(I);
        }
      } else {
        report_fatal_error("Unable to retrieve constant int from "
                           "sycl_kernel_omit_args metadata node");
      }
    } else {
      report_fatal_error(
          "Error while processing sycl_kernel_omit_args metadata node");
    }
  }
  return Res;
}

void emitSubkernelForKernel(Function *F, Type *NativeCPUArgDescType,
                            Type *StatePtrType) {
  LLVMContext &Ctx = F->getContext();
  Type *NativeCPUArgDescPtrType = PointerType::getUnqual(NativeCPUArgDescType);

  // Create function signature
  // Todo: we need to ensure that the kernel name is not mangled as a type
  // name, otherwise this may lead to runtime failures due to *weird*
  // codegen/linking behaviour, we change the name of the kernel, and the
  // subhandler steals its name, we add a suffix to the subhandler later
  // on when lowering the device module
  std::string OldName = F->getName().str();
  std::string NewName = OldName + ".NativeCPUKernel";
  const auto SubHandlerName = OldName;
  F->setName(NewName);
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Ctx), {NativeCPUArgDescPtrType, StatePtrType}, false);
  auto SubhFCallee = F->getParent()->getOrInsertFunction(SubHandlerName, FTy);
  Function *SubhF = cast<Function>(SubhFCallee.getCallee());

  // Emit function body, unpack kernel args
  auto UsedIndexes = getUsedIndexes(F);
  auto *KernelTy = F->getFunctionType();
  // assert(UsedIndexes.size() + 1 == KernelTy->getNumParams() && "mismatch
  // between number of params and used args");
  IRBuilder<> Builder(Ctx);
  BasicBlock *Block = BasicBlock::Create(Ctx, "entry", SubhF);
  Builder.SetInsertPoint(Block);
  unsigned NumArgs = UsedIndexes.size();
  auto *BaseNativeCPUArg = SubhF->getArg(0);
  SmallVector<Value *, 5> KernelArgs;
  for (unsigned I = 0; I < NumArgs; I++) {
    auto *Arg = F->getArg(I);
    auto UsedI = UsedIndexes[I];
    // Load the correct NativeCPUDesc and load the pointer from it
    auto *Addr = Builder.CreateGEP(NativeCPUArgDescType, BaseNativeCPUArg,
                                   {Builder.getInt64(UsedI)});
    auto *Load = Builder.CreateLoad(PointerType::getUnqual(Ctx), Addr);
    if (Arg->getType()->isPointerTy()) {
      // If the arg is a pointer, just use it
      KernelArgs.push_back(Load);
    } else {
      // Otherwise, load the scalar value and use that
      auto *Scalar = Builder.CreateLoad(Arg->getType(), Load);
      KernelArgs.push_back(Scalar);
    }
  }

  // Call the kernel
  // Add the nativecpu state as arg
  KernelArgs.push_back(SubhF->getArg(1));
  Builder.CreateCall(KernelTy, F, KernelArgs);
  Builder.CreateRetVoid();

  // Add sycl-module-id attribute
  // Todo: we may want to copy other attributes to the subhandler,
  // but we can't simply use setAttributes(F->getAttributes) since
  // the function signatures are different
  if (F->hasFnAttribute(sycl::utils::ATTR_SYCL_MODULE_ID)) {
    Attribute MId = F->getFnAttribute(sycl::utils::ATTR_SYCL_MODULE_ID);
    SubhF->addFnAttr("sycl-module-id", MId.getValueAsString());
  }
}

// Clones the function and returns a new function with a new argument on type T
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
  for (const auto &Pair : llvm::zip(OldF->args(), NewF->args())) {
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
static const std::map<std::string, std::pair<std::string, unsigned int>>
    BuiltinNamesMap{
        {"_Z28__spirv_GlobalInvocationId_xv",
         {"__dpcpp_nativecpu_global_id", 0}},
        {"_Z28__spirv_GlobalInvocationId_yv",
         {"__dpcpp_nativecpu_global_id", 1}},
        {"_Z28__spirv_GlobalInvocationId_zv",
         {"__dpcpp_nativecpu_global_id", 2}},
        {"_Z20__spirv_GlobalSize_xv", {"__dpcpp_nativecpu_global_range", 0}},
        {"_Z20__spirv_GlobalSize_yv", {"__dpcpp_nativecpu_global_range", 1}},
        {"_Z20__spirv_GlobalSize_zv", {"__dpcpp_nativecpu_global_range", 2}},
        {"_Z22__spirv_GlobalOffset_xv",
         {"__dpcpp_nativecpu_get_global_offset", 0}},
        {"_Z22__spirv_GlobalOffset_yv",
         {"__dpcpp_nativecpu_get_global_offset", 1}},
        {"_Z22__spirv_GlobalOffset_zv",
         {"__dpcpp_nativecpu_get_global_offset", 2}},
        {"_Z27__spirv_LocalInvocationId_xv",
         {"__dpcpp_nativecpu_get_local_id", 0}},
        {"_Z27__spirv_LocalInvocationId_yv",
         {"__dpcpp_nativecpu_get_local_id", 1}},
        {"_Z27__spirv_LocalInvocationId_zv",
         {"__dpcpp_nativecpu_get_local_id", 2}},
        {"_Z23__spirv_NumWorkgroups_xv",
         {"__dpcpp_nativecpu_get_num_groups", 0}},
        {"_Z23__spirv_NumWorkgroups_yv",
         {"__dpcpp_nativecpu_get_num_groups", 1}},
        {"_Z23__spirv_NumWorkgroups_zv",
         {"__dpcpp_nativecpu_get_num_groups", 2}},
        {"_Z23__spirv_WorkgroupSize_xv", {"__dpcpp_nativecpu_get_wg_size", 0}},
        {"_Z23__spirv_WorkgroupSize_yv", {"__dpcpp_nativecpu_get_wg_size", 1}},
        {"_Z23__spirv_WorkgroupSize_zv", {"__dpcpp_nativecpu_get_wg_size", 2}},
        {"_Z21__spirv_WorkgroupId_xv", {"__dpcpp_nativecpu_get_wg_id", 0}},
        {"_Z21__spirv_WorkgroupId_yv", {"__dpcpp_nativecpu_get_wg_id", 1}},
        {"_Z21__spirv_WorkgroupId_zv", {"__dpcpp_nativecpu_get_wg_id", 2}}};

Function *getReplaceFunc(const Module &M, StringRef Name) {
  Function *F = M.getFunction(Name);
  assert(F && "Error retrieving replace function");
  return F;
}

Value *getStateArg(const Function *F) {
  auto *FT = F->getFunctionType();
  return F->getArg(FT->getNumParams() - 1);
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

  StructType *NativeCPUArgDescType =
      StructType::create({PointerType::getUnqual(M.getContext())});
  for (auto &NewK : NewKernels) {
    emitSubkernelForKernel(NewK, NativeCPUArgDescType, StatePtrType);
  }

  // Then we iterate over all the supported builtins, find their uses and
  // replace them with calls to our Native CPU functions.
  for (const auto &Entry : BuiltinNamesMap) {
    auto *Glob = M.getFunction(Entry.first);
    if (!Glob)
      continue;
    auto *ReplaceFunc = getReplaceFunc(M, Entry.second.first);
    SmallVector<Instruction *> ToRemove;
    for (const auto &Use : Glob->uses()) {
      auto I = dyn_cast<CallInst>(Use.getUser());
      if (!I)
        report_fatal_error("Unsupported Value in SYCL Native CPU\n");
      if (I->getFunction()->getCallingConv() != llvm::CallingConv::SPIR_KERNEL)
        report_fatal_error(
            "SYCL Native CPU currently doesn't support non-inlined "
            "functions yet, try increasing the inlining threshold. Support for "
            "non-inlined functions is planned.");
      auto *Arg = ConstantInt::get(Type::getInt32Ty(M.getContext()),
                                   Entry.second.second);
      auto *NewI = CallInst::Create(ReplaceFunc->getFunctionType(), ReplaceFunc,
                                    {Arg, getStateArg(I->getFunction())},
                                    "ncpu_call", I);
      I->replaceAllUsesWith(NewI);
      ToRemove.push_back(I);
    }

    for (auto &El : ToRemove)
      El->eraseFromParent();

    // Finally, we erase the builtin from the module
    Glob->eraseFromParent();
  }

  for (auto &F : M) {
    fixCallingConv(&F);
  }
  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
