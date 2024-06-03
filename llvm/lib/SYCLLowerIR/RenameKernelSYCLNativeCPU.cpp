//===- RenameKernelSYCLNativeCPU.cpp - Kernel renaming for SYCL Native CPU-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass that renames the kernel names, to ensure the name
// doesn't clash with other names.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/RenameKernelSYCLNativeCPU.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"
#include "llvm/SYCLLowerIR/UtilsSYCLNativeCPU.h"
#include <set>

using namespace llvm;

static bool isSpirvSyclBuiltin(StringRef FName) {
  if (!FName.consume_front("_Z"))
    return false;
  // now skip the digits
  FName = FName.drop_while([](char C) { return std::isdigit(C); });

  return FName.starts_with("__spirv_") || FName.starts_with("__sycl_");
}

PreservedAnalyses
RenameKernelSYCLNativeCPUPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  // Add NativeCPU suffix to module exports (kernels) and make other
  // function definitions private
  for (auto &F : M) {

    // Update call sites that still have SPIR calling conventions
    const llvm::CallingConv::ID conv = F.getCallingConv();
    // assert(conv != llvm::CallingConv::SPIR_FUNC &&
    //        conv != llvm::CallingConv::SPIR_KERNEL);
    // todo: reenable assert
    for (const auto &Use : F.uses()) {
      if (auto I = dyn_cast<CallInst>(Use.getUser())) {
        if (I->getCallingConv() == llvm::CallingConv::SPIR_FUNC ||
            I->getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
          if (I->getCallingConv() == conv)
            continue;
          I->setCallingConv(conv);
          ModuleChanged = true;
        }
        assert(I->getCallingConv() == conv);
      }
    }

    if (F.hasFnAttribute(sycl::utils::ATTR_SYCL_MODULE_ID)) {
      F.setName(sycl::utils::addSYCLNativeCPUSuffix(F.getName()));
      ModuleChanged = true;
    } else if (isSpirvSyclBuiltin(F.getName()) || F.isIntrinsic())
      continue;
    else if (!F.isDeclaration()) {
      F.setComdat(nullptr);
      // todo: check what functions could be exported
      F.setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
      ModuleChanged = true;
    }
  }

  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
