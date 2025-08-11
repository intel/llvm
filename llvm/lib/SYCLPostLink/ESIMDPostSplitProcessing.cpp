//===-------- ESIMDPostSplitProcessing.cpp  -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLPostLink/ESIMDPostSplitProcessing.h"

#include "llvm/GenXIntrinsics/GenXSPIRVWriterAdaptor.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/SYCLLowerIR/ESIMD/LowerESIMD.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SROA.h"

#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::module_split;

namespace {

ModulePassManager buildESIMDLoweringPipeline(bool ForceDisableESIMDOpt,
                                             bool SplitESIMD) {
  ModulePassManager MPM;
  MPM.addPass(SYCLLowerESIMDPass(!SplitESIMD));

  if (!ForceDisableESIMDOpt) {
    FunctionPassManager FPM;
    FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }
  MPM.addPass(ESIMDOptimizeVecArgCallConvPass{});
  FunctionPassManager MainFPM;
  MainFPM.addPass(ESIMDLowerLoadStorePass{});

  if (!ForceDisableESIMDOpt) {
    MainFPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    MainFPM.addPass(EarlyCSEPass(true));
    MainFPM.addPass(InstCombinePass{});
    MainFPM.addPass(DCEPass{});
    MainFPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    MainFPM.addPass(EarlyCSEPass(true));
    MainFPM.addPass(InstCombinePass{});
    MainFPM.addPass(DCEPass{});
  }
  MPM.addPass(ESIMDLowerSLMReservationCalls{});
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(MainFPM)));
  MPM.addPass(GenXSPIRVWriterAdaptor(/*RewriteTypes=*/true,
                                     /*RewriteSingleElementVectorsIn*/ false));
  return MPM;
}

} // anonymous namespace

// When ESIMD code was separated from the regular SYCL code,
// we can safely process ESIMD part.
bool sycl::lowerESIMDConstructs(ModuleDesc &MD, bool ForceDisableESIMDOpt,
                                bool SplitESIMD) {
  // TODO: support options like -debug-pass, -print-[before|after], and others
  LoopAnalysisManager LAM;
  CGSCCAnalysisManager CGAM;
  FunctionAnalysisManager FAM;
  ModuleAnalysisManager MAM;

  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  std::vector<std::string> Names;
  MD.saveEntryPointNames(Names);
  ModulePassManager MPM =
      buildESIMDLoweringPipeline(ForceDisableESIMDOpt, SplitESIMD);
  PreservedAnalyses Res = MPM.run(MD.getModule(), MAM);

  // GenXSPIRVWriterAdaptor pass replaced some functions with "rewritten"
  // versions so the entry point table must be rebuilt.
  MD.rebuildEntryPoints(Names);
  return !Res.areAllPreserved();
}
