//==-------------------------- FusionPipeline.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FusionPipeline.h"

#include "helper/ConfigHelper.h"
#include "kernel-fusion/SYCLSpecConstMaterializer.h"

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

bool FusionPipeline::runMaterializerPasses(
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
