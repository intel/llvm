//- CheckNDRangeSYCLNativeCPU.cpp - Check if a kernel uses nd_range features -//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Checks if the kernel uses features from nd_item such as:
// * local id
// * local range
// * local memory
// * work group barrier
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/CheckNDRangeSYCLNativeCPU.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/SYCLLowerIR/UtilsSYCLNativeCPU.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

static std::array<const char *, 13> NdBuiltins{
    "_Z23__spirv_WorkgroupSize_xv",     "_Z23__spirv_WorkgroupSize_yv",
    "_Z23__spirv_WorkgroupSize_zv",     "_Z23__spirv_NumWorkgroups_xv",
    "_Z23__spirv_NumWorkgroups_yv",     "_Z23__spirv_NumWorkgroups_zv",
    "_Z21__spirv_WorkgroupId_xv",       "_Z21__spirv_WorkgroupId_yv",
    "_Z21__spirv_WorkgroupId_zv",       "_Z27__spirv_LocalInvocationId_xv",
    "_Z27__spirv_LocalInvocationId_yv", "_Z27__spirv_LocalInvocationId_zv",
    "_Z22__spirv_ControlBarrierjjj"};

static void addNDRangeMetadata(Function &F, bool Value) {
  auto &Ctx = F.getContext();
  F.setMetadata("is_nd_range",
                MDNode::get(Ctx, ConstantAsMetadata::get(ConstantInt::get(
                                     Type::getInt1Ty(Ctx), Value))));
}

PreservedAnalyses
CheckNDRangeSYCLNativeCPUPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  SmallPtrSet<Function *, 5> NdFuncs; // Functions that use NDRange features
  SmallPtrSet<Function *, 5> Visited;
  SmallPriorityWorklist<Function *, 5> WorkList;

  // Add builtins to the set of functions that may use NDRange features
  for (auto &FName : NdBuiltins) {
    auto F = M.getFunction(FName);
    if (F == nullptr)
      continue;
    WorkList.insert(F);
    NdFuncs.insert(F);
  }

  // Add users of local AS global var to the set of functions that may use
  // NDRange features
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != sycl::utils::SyclNativeCpuLocalAS)
      continue;

    for (auto U : GV.users()) {
      if (auto I = dyn_cast<Instruction>(U)) {
        auto F = I->getFunction();
        if (F != nullptr && NdFuncs.insert(F).second) {
          WorkList.insert(F);
          NdFuncs.insert(F);
        }
      }
    }
  }

  // Traverse the use chain to find Functions that may use NDRange features
  // (or, recursively, Functions that call Functions that may use NDRange
  // features)
  while (!WorkList.empty()) {
    auto F = WorkList.pop_back_val();

    for (User *U : F->users()) {
      if (auto CI = dyn_cast<CallInst>(U)) {
        auto Caller = CI->getFunction();
        if (!Caller)
          continue;
        if (!Visited.contains(Caller)) {
          WorkList.insert(Caller);
          NdFuncs.insert(Caller);
        }
      }
    }
    Visited.insert(F);
  }

  for (auto &F : M) {
    if (F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
      bool IsNDRange = false;

      // Check for local memory args
      for (auto &A : F.args()) {
        if (auto Ptr = dyn_cast<PointerType>(A.getType());
            Ptr && Ptr->getAddressSpace() == 3) {
          IsNDRange = true;
        }
      }

      // Check if the kernel calls one of the ND Range builtins
      IsNDRange |= NdFuncs.contains(&F);

      addNDRangeMetadata(F, IsNDRange);
      ModuleChanged = true;
    }
  }
  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
