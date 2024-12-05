//===------------- ESIMD.cpp - Driver for ESIMD lowering ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ESIMD.h"

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/GenXIntrinsics/GenXSPIRVWriterAdaptor.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/SYCLLowerIR/ESIMD/LowerESIMD.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SROA.h"

using namespace llvm;

using string_vector = std::vector<std::string>;

// When ESIMD code was separated from the regular SYCL code,
// we can safely process ESIMD part.
void jit_compiler::lowerEsimdConstructs(module_split::ModuleDesc &MD,
                                        bool PerformOpts) {
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

  ModulePassManager MPM;
  MPM.addPass(SYCLLowerESIMDPass(/*ModuleContainsScalar=*/false));

  if (PerformOpts) {
    FunctionPassManager FPM;
    FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }
  MPM.addPass(ESIMDOptimizeVecArgCallConvPass{});
  FunctionPassManager MainFPM;
  MainFPM.addPass(ESIMDLowerLoadStorePass{});

  if (PerformOpts) {
    MainFPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    MainFPM.addPass(EarlyCSEPass(true));
    MainFPM.addPass(InstCombinePass{});
    MainFPM.addPass(DCEPass{});
    // TODO: maybe remove some passes below that don't affect code quality
    MainFPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    MainFPM.addPass(EarlyCSEPass(true));
    MainFPM.addPass(InstCombinePass{});
    MainFPM.addPass(DCEPass{});
  }
  MPM.addPass(ESIMDLowerSLMReservationCalls{});
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(MainFPM)));
  MPM.addPass(GenXSPIRVWriterAdaptor(/*RewriteTypes=*/true,
                                     /*RewriteSingleElementVectorsIn*/ false));
  // GenXSPIRVWriterAdaptor pass replaced some functions with "rewritten"
  // versions so the entry point table must be rebuilt.
  // TODO Change entry point search to analysis?
  std::vector<std::string> Names;
  MD.saveEntryPointNames(Names);
  MPM.run(MD.getModule(), MAM);
  MD.rebuildEntryPoints(Names);
}
