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
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"

using namespace llvm;

static std::array<const char *, 13> NdFunctions{
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

      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto CI = dyn_cast<CallInst>(&I)) {
            auto CalleeName = CI->getCalledFunction()->getName();
            if (std::find(NdFunctions.begin(), NdFunctions.end(), CalleeName) !=
                NdFunctions.end()) {
              IsNDRange = true;
              break;
            }
          }
        }
        if (IsNDRange) {
          break;
        }
      }

      addNDRangeMetadata(F, IsNDRange);
    }
  }
  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
