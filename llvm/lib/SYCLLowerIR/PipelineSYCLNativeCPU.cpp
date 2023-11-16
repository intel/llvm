//===---- PipelineSYCLNativeCPU.cpp - Pass pipeline for SYCL Native CPU ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the pass pipeline used when lowering device code for SYCL Native
// CPU.
// When NATIVECPU_USE_OCK is set, adds passes from the oneAPI Construction Kit.
//
//===----------------------------------------------------------------------===//
#include "llvm/SYCLLowerIR/ConvertToMuxBuiltinsSYCLNativeCPU.h"
#include "llvm/SYCLLowerIR/PrepareSYCLNativeCPU.h"
#include "llvm/SYCLLowerIR/RenameKernelSYCLNativeCPU.h"
#include "llvm/SYCLLowerIR/UtilsSYCLNativeCPU.h"

#ifdef NATIVECPU_USE_OCK
#include "compiler/utils/builtin_info.h"
#include "compiler/utils/sub_group_analysis.h"
#include "compiler/utils/work_item_loops_pass.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#endif

using namespace llvm;
using namespace sycl::utils;

void llvm::sycl::utils::addSYCLNativeCPUBackendPasses(llvm::ModulePassManager &MPM,
                                   ModuleAnalysisManager &MAM) {
  MPM.addPass(ConvertToMuxBuiltinsSYCLNativeCPUPass());
#ifdef NATIVECPU_USE_OCK
  // Todo set options properly
  compiler::utils::WorkItemLoopsPassOptions Opts;
  Opts.IsDebug = false;
  Opts.ForceNoTail = false;
  MAM.registerPass([&] { return compiler::utils::BuiltinInfoAnalysis(); });
  MAM.registerPass([&] { return compiler::utils::SubgroupAnalysis(); });
  MPM.addPass(compiler::utils::WorkItemLoopsPass(Opts));
  MPM.addPass(AlwaysInlinerPass());

#endif
  MPM.addPass(PrepareSYCLNativeCPUPass());
  MPM.addPass(RenameKernelSYCLNativeCPUPass());
}
