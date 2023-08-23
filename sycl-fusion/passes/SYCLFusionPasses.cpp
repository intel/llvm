//==------------------------ SYCLFusionPasses.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "Kernel.h"

#include "internalization/Internalization.h"
#include "kernel-fusion/SYCLKernelFusion.h"
#include "kernel-info/SYCLKernelInfo.h"
#include "syclcp/SYCLCP.h"

using namespace llvm;
using namespace jit_compiler;

cl::opt<bool>
    NoBarriers("sycl-kernel-fusion-no-barriers",
               cl::desc("Disable barrier insertion for SYCL kernel fusion."));

llvm::PassPluginLibraryInfo getSYCLKernelFusionPluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION, "SYCL-Module-Info", LLVM_VERSION_STRING,
      [](PassBuilder &PB) {
        PB.registerPipelineParsingCallback(
            [](StringRef Name, ModulePassManager &MPM,
               ArrayRef<PassBuilder::PipelineElement>) {
              if (Name == "sycl-kernel-fusion") {
                BarrierFlags BarrierFlag =
                    (NoBarriers) ? getNoBarrierFlag()
                                 : SYCLKernelFusion::DefaultBarriersFlags;
                MPM.addPass(SYCLKernelFusion(BarrierFlag));
                return true;
              }
              if (Name == "sycl-internalization") {
                MPM.addPass(SYCLInternalizer());
                return true;
              }
              if (Name == "sycl-cp") {
                MPM.addPass(SYCLCP());
                return true;
              }
              if (Name == "print-sycl-module-info") {
                MPM.addPass(SYCLModuleInfoPrinter());
                return true;
              }
              return false;
            });
        PB.registerAnalysisRegistrationCallback([](ModuleAnalysisManager &MAM) {
          MAM.registerPass([]() { return SYCLModuleInfoAnalysis{}; });
        });
      }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getSYCLKernelFusionPluginInfo();
}
