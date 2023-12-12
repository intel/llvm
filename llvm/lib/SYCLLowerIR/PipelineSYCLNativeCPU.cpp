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
#include "compiler/utils/prepare_barriers_pass.h"
#include "compiler/utils/sub_group_analysis.h"
#include "compiler/utils/work_item_loops_pass.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#endif

using namespace llvm;
using namespace sycl::utils;

cl::opt<bool>
    ForceNoTail("native-cpu-force-no-tail", cl::init(false),
                cl::desc("Never emit the peeling loop for vectorized kernels,"
                         "even when the local size is not known to be a "
                         "multiple of the vector width"));

cl::opt<bool> IsDebug(
    "native-cpu-debug", cl::init(false),
    cl::desc("Emit extra alloca instructions to preserve the value of live"
             "variables between barriers"));

void llvm::sycl::utils::addSYCLNativeCPUBackendPasses(
    llvm::ModulePassManager &MPM, ModuleAnalysisManager &MAM) {
  MPM.addPass(ConvertToMuxBuiltinsSYCLNativeCPUPass());
#ifdef NATIVECPU_USE_OCK
  compiler::utils::WorkItemLoopsPassOptions Opts;
  Opts.IsDebug = IsDebug;
  Opts.ForceNoTail = ForceNoTail;
  MAM.registerPass([&] { return compiler::utils::BuiltinInfoAnalysis(); });
  MAM.registerPass([&] { return compiler::utils::SubgroupAnalysis(); });
  MPM.addPass(compiler::utils::PrepareBarriersPass());
  MPM.addPass(compiler::utils::WorkItemLoopsPass(Opts));
  MPM.addPass(AlwaysInlinerPass());
#endif
  MPM.addPass(PrepareSYCLNativeCPUPass());
  MPM.addPass(RenameKernelSYCLNativeCPUPass());
}
