//==---- SYCLKernelInfo - Analysis pass to provide access to kernel info ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_PASSES_SYCLKERNELINFO_H
#define SYCL_FUSION_PASSES_SYCLKERNELINFO_H

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "Kernel.h"

///
/// Analysis pass to make the SYCLModuleInfo available to other
/// passes for usage and updating.
class SYCLModuleInfoAnalysis
    : public llvm::AnalysisInfoMixin<SYCLModuleInfoAnalysis> {
public:
  static llvm::AnalysisKey Key;

  constexpr static llvm::StringLiteral ModuleInfoMDKey{"sycl.moduleinfo"};

  SYCLModuleInfoAnalysis(
      std::unique_ptr<jit_compiler::SYCLModuleInfo> Info = nullptr)
      : ModuleInfo{std::move(Info)} {}

  struct Result {
    jit_compiler::SYCLModuleInfo *ModuleInfo;
  };
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &);

private:
  void loadModuleInfoFromFile();
  void loadModuleInfoFromMetadata(llvm::Module &M);

  std::unique_ptr<jit_compiler::SYCLModuleInfo> ModuleInfo;
};

///
/// Simple pass to print the result of the SYCLModuleInfoAnalysis pass,
/// i.e., the SYCLModuleInfo.
struct SYCLModuleInfoPrinter
    : public llvm::PassInfoMixin<SYCLModuleInfoPrinter> {
  llvm::PreservedAnalyses run(llvm::Module &, llvm::ModuleAnalysisManager &);
  static bool isRequired() { return true; }
};

#endif // SYCL_FUSION_PASSES_SYCLKERNELINFO_H
