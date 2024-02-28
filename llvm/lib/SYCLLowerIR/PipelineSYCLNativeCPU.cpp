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
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/SYCLLowerIR/ConvertToMuxBuiltinsSYCLNativeCPU.h"
#include "llvm/SYCLLowerIR/PrepareSYCLNativeCPU.h"
#include "llvm/SYCLLowerIR/RenameKernelSYCLNativeCPU.h"
#include "llvm/SYCLLowerIR/UtilsSYCLNativeCPU.h"

#ifdef NATIVECPU_USE_OCK
#include "compiler/utils/builtin_info.h"
#include "compiler/utils/device_info.h"
#include "compiler/utils/prepare_barriers_pass.h"
#include "compiler/utils/sub_group_analysis.h"
#include "compiler/utils/work_item_loops_pass.h"
#include "vecz/pass.h"
#include "vecz/vecz_target_info.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#endif

using namespace llvm;
using namespace sycl::utils;

static cl::opt<bool>
    ForceNoTail("native-cpu-force-no-tail", cl::init(false),
                cl::desc("Never emit the peeling loop for vectorized kernels,"
                         "even when the local size is not known to be a "
                         "multiple of the vector width"));

static cl::opt<bool> IsDebug(
    "native-cpu-debug", cl::init(false),
    cl::desc("Emit extra alloca instructions to preserve the value of live"
             "variables between barriers"));

static cl::opt<unsigned> NativeCPUVeczWidth(
    "sycl-native-cpu-vecz-width", cl::init(8),
    cl::desc("Vector width for SYCL Native CPU vectorizer, defaults to 8"));

static cl::opt<bool>
    SYCLNativeCPUNoVecz("sycl-native-cpu-no-vecz", cl::init(false),
                        cl::desc("Disable vectorizer for SYCL Native CPU"));

void llvm::sycl::utils::addSYCLNativeCPUBackendPasses(
    llvm::ModulePassManager &MPM, ModuleAnalysisManager &MAM,
    OptimizationLevel OptLevel) {
  MPM.addPass(ConvertToMuxBuiltinsSYCLNativeCPUPass());
#ifdef NATIVECPU_USE_OCK
  // Always enable vectorizer, unless explictly disabled or -O0 is set.
  if (OptLevel != OptimizationLevel::O0 && !SYCLNativeCPUNoVecz) {
    MAM.registerPass([] { return vecz::TargetInfoAnalysis(); });
    MAM.registerPass([] { return compiler::utils::DeviceInfoAnalysis(); });
    auto QueryFunc =
        [](const llvm::Function &F, const llvm::ModuleAnalysisManager &,
           llvm::SmallVectorImpl<vecz::VeczPassOptions> &Opts) -> bool {
      if (F.getCallingConv() != llvm::CallingConv::SPIR_KERNEL) {
        return false;
      }
      compiler::utils::VectorizationFactor VF(NativeCPUVeczWidth, false);
      vecz::VeczPassOptions VPO;
      VPO.factor = std::move(VF);
      Opts.emplace_back(std::move(VPO));
      return true;
    };
    MAM.registerPass(
        [QueryFunc] { return vecz::VeczPassOptionsAnalysis(QueryFunc); });
    MPM.addPass(vecz::RunVeczPass());
  }
  compiler::utils::WorkItemLoopsPassOptions Opts;
  Opts.IsDebug = IsDebug;
  Opts.ForceNoTail = ForceNoTail;
  MAM.registerPass([] { return compiler::utils::BuiltinInfoAnalysis(); });
  MAM.registerPass([] { return compiler::utils::SubgroupAnalysis(); });
  MPM.addPass(compiler::utils::PrepareBarriersPass());
  MPM.addPass(compiler::utils::WorkItemLoopsPass(Opts));
  MPM.addPass(AlwaysInlinerPass());
#endif
  MPM.addPass(PrepareSYCLNativeCPUPass());
  MPM.addPass(RenameKernelSYCLNativeCPUPass());

  // Run optimization passes after all the changes we made to the kernels.
  // Todo: check optimization level from clang
  // Todo: maybe we could find a set of relevant passes instead of re-running
  // the full optimization pipeline.
  PassBuilder PB;
  MPM.addPass(PB.buildPerModuleDefaultPipeline(OptLevel));
}
