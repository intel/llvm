//===-- ConvertToMuxBuiltinsSYCLNativeCPU.cpp - Convert to Mux Builtins ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts SPIRV builtins to Mux builtins used by the oneAPI Construction
// Kit for SYCL Native CPU
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/ConvertToMuxBuiltinsSYCLNativeCPU.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"
#include <map>

using namespace llvm;

namespace {

static void fixFunctionAttributes(Function *F) {
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
}

static constexpr const char *MuxKernelAttrName = "mux-kernel";

void setIsKernelEntryPt(Function &F) {
  F.addFnAttr(MuxKernelAttrName, "entry-point");
}

} // namespace

PreservedAnalyses
ConvertToMuxBuiltinsSYCLNativeCPUPass::run(Module &M,
                                           ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  for (auto &F : M) {
    if (F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
      fixFunctionAttributes(&F);
      setIsKernelEntryPt(F);
      ModuleChanged = true;
    }
  }
  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
