//===- MaterializerPipeline.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MaterializerPipeline.h"

#include "materializer/SYCLSpecConstMaterializer.h"

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Scalar/SROA.h"

using namespace llvm;
using namespace jit_compiler;

bool MaterializerPipeline::runMaterializerPasses(
    llvm::Module &Mod, llvm::ArrayRef<unsigned char> SpecConstData) {
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

  ModulePassManager MPM;
  // Register inserter and materializer passes.
  {
    FunctionPassManager FPM;
    MPM.addPass(SYCLSpecConstDataInserter{SpecConstData});
    FPM.addPass(SYCLSpecConstMaterializer{});
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }
  // Add generic optimizations,
  {
    FunctionPassManager FPM;
    MPM.addPass(AlwaysInlinerPass{});
    FPM.addPass(SROAPass{SROAOptions::ModifyCFG});
    FPM.addPass(SCCPPass{});
    FPM.addPass(ADCEPass{});
    FPM.addPass(EarlyCSEPass{/*UseMemorySSA*/ true});
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }
  // followed by unrolling.
  {
    FunctionPassManager FPM;
    FPM.addPass(createFunctionToLoopPassAdaptor(IndVarSimplifyPass{}));
    LoopUnrollOptions UnrollOptions;
    FPM.addPass(LoopUnrollPass{UnrollOptions});
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }

  MPM.run(Mod, MAM);

  return true;
}
