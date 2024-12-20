//===-- SanitizerKernelMetadata.cpp - fix kernel medatadata for sanitizer -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass fixes attributes and metadata of global variable
// "__AsanKernelMetadata" or "__MsanKernelMetadata".
// We treat "KernelMetadata" as a device global variable, so that it
// can be read by runtime.
// "spirv.Decorations" is removed by llvm-link, so we add it here again.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SanitizerKernelMetadata.h"

#include "llvm/IR/IRBuilder.h"

#define DEBUG_TYPE "SanitizerKernelMetadata"

using namespace llvm;

namespace llvm {

constexpr StringRef SPIRV_DECOR_MD_KIND = "spirv.Decorations";
constexpr uint32_t SPIRV_HOST_ACCESS_DECOR = 6147;

PreservedAnalyses SanitizerKernelMetadataPass::run(Module &M,
                                                   ModuleAnalysisManager &MAM) {
  auto *KernelMetadata = M.getNamedGlobal("__AsanKernelMetadata");

  if (!KernelMetadata)
    KernelMetadata = M.getNamedGlobal("__MsanKernelMetadata");

  if (!KernelMetadata)
    return PreservedAnalyses::all();

  auto &DL = M.getDataLayout();
  auto &Ctx = M.getContext();

  // Fix attributes
  KernelMetadata->addAttribute(
      "sycl-device-global-size",
      std::to_string(DL.getTypeAllocSize(KernelMetadata->getValueType())));

  // Fix metadata
  unsigned MDKindID = Ctx.getMDKindID(SPIRV_DECOR_MD_KIND);

  SmallVector<Metadata *, 1> MDOps;

  SmallVector<Metadata *, 3> MD;
  auto *Ty = Type::getInt32Ty(Ctx);
  MD.push_back(ConstantAsMetadata::get(
      Constant::getIntegerValue(Ty, APInt(32, SPIRV_HOST_ACCESS_DECOR))));
  MD.push_back(
      ConstantAsMetadata::get(Constant::getIntegerValue(Ty, APInt(32, 0))));
  MD.push_back(MDString::get(Ctx, "_Z20__SanitizerKernelMetadata"));

  MDOps.push_back(MDNode::get(Ctx, MD));

  KernelMetadata->addMetadata(MDKindID, *MDNode::get(Ctx, MDOps));

  return PreservedAnalyses::none();
}

} // namespace llvm
