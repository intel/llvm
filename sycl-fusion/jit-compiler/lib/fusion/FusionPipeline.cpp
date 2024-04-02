//==-------------------------- FusionPipeline.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FusionPipeline.h"

#include "debug/PassDebug.h"
#include "helper/ConfigHelper.h"
#include "internalization/Internalization.h"
#include "kernel-fusion/SYCLKernelFusion.h"
#include "kernel-info/SYCLKernelInfo.h"
#include "syclcp/SYCLCP.h"

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#ifndef NDEBUG
#include "llvm/IR/Verifier.h"
#endif // NDEBUG
#include "llvm/Passes/PassBuilder.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

using namespace llvm;
using namespace jit_compiler;
using namespace jit_compiler::fusion;

static unsigned getFlatAddressSpace(Module &Mod) {
  // Ideally, we could get this information from the TargetTransformInfo, but
  // the SPIR-V backend does not yet seem to have an implementation for that.
  llvm::Triple Tri(Mod.getTargetTriple());
  if (Tri.isNVPTX() || Tri.isAMDGCN()) {
    return 0;
  }
  if (Tri.isSPIRV() || Tri.isSPIR()) {
    return 4;
  }
  // Identical to the definition of "UninitializedAddressSpace" in
  // "InferAddressSpaces.cpp".
  return std::numeric_limits<unsigned>::max();
}

std::unique_ptr<SYCLModuleInfo>
FusionPipeline::runFusionPasses(Module &Mod, SYCLModuleInfo &InputInfo,
                                BarrierFlags BarriersFlags) {
  // Perform the actual kernel fusion, i.e., generate a kernel function for the
  // fused kernel from the kernel functions of the input kernels. This is done
  // by the SYCLKernelFusion LLVM pass, which is run here through a custom LLVM
  // pass pipeline. In order to perform internalization, we run the
  // SYCLInternalizer pass.

  bool DebugEnabled = ConfigHelper::get<option::JITEnableVerbose>();
  if (DebugEnabled) {
    // Enabled debug output from the fusion passes.
    jit_compiler::PassDebug = true;
  }

  // Initialize the analysis managers with all the registered analyses.
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Make the existing SYCLModuleInfo available to the pass pipeline via the
  // corresponding analysis pass.
  MAM.registerPass([&]() {
    auto ModInfo = std::make_unique<SYCLModuleInfo>(InputInfo);
    return SYCLModuleInfoAnalysis{std::move(ModInfo)};
  });
  ModulePassManager MPM;
  // Run the fusion pass on the LLVM IR module.
  MPM.addPass(SYCLKernelFusion{BarriersFlags});
  // This pass is needed to inline remapping function calls inserted by the
  // SYCLKernelFusion pass.
  MPM.addPass(AlwaysInlinerPass{});
  {
    FunctionPassManager FPM;
    // Run loop unrolling and SROA to split the kernel functor struct into its
    // scalar parts, to avoid problems with address-spaces and enable
    // internalization.
    FPM.addPass(createFunctionToLoopPassAdaptor(IndVarSimplifyPass{}));
    LoopUnrollOptions UnrollOptions;
    FPM.addPass(LoopUnrollPass{UnrollOptions});
    FPM.addPass(SROAPass{SROAOptions::ModifyCFG});
    // Run the InferAddressSpace pass to remove as many address-space casts
    // to/from generic address-space as possible, because these hinder
    // internalization.
    // Ideally, the static compiler should have performed that job.
    const unsigned FlatAddressSpace = getFlatAddressSpace(Mod);
    FPM.addPass(InferAddressSpacesPass(FlatAddressSpace));
    // Run CFG simplification to prevent unreachable code from obscuring
    // internalization opportunities.
    FPM.addPass(SimplifyCFGPass{});
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }
  // Run dataflow internalization and runtime constant propagation.
  MPM.addPass(SYCLInternalizer{});
  MPM.addPass(SYCLCP{});
  // Run additional optimization passes after completing fusion.
  {
    FunctionPassManager FPM;
    FPM.addPass(SROAPass{SROAOptions::ModifyCFG});
    FPM.addPass(SCCPPass{});
    FPM.addPass(InstCombinePass{});
    FPM.addPass(SimplifyCFGPass{});
    FPM.addPass(SROAPass{SROAOptions::ModifyCFG});
    FPM.addPass(InstCombinePass{});
    FPM.addPass(SimplifyCFGPass{});
    FPM.addPass(ADCEPass{});
    FPM.addPass(EarlyCSEPass{/*UseMemorySSA*/ true});
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }
  MPM.run(Mod, MAM);

  if (DebugEnabled) {
    // Restore debug option
    jit_compiler::PassDebug = false;
  }

  assert(!verifyModule(Mod, &errs()) && "Invalid LLVM IR generated");

  auto NewModInfo = MAM.getResult<SYCLModuleInfoAnalysis>(Mod);
  assert(NewModInfo.ModuleInfo && "Failed to retrieve SYCL module info");

  return std::make_unique<SYCLModuleInfo>(std::move(*NewModInfo.ModuleInfo));
}
