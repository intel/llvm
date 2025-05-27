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
#include "llvm/Linker/Linker.h"
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

ModulePassManager buildESIMDLoweringPipeline(bool Optimize, bool SplitESIMD) {
  ModulePassManager MPM;
  MPM.addPass(SYCLLowerESIMDPass(!SplitESIMD));

  if (Optimize) {
    FunctionPassManager FPM;
    FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }
  MPM.addPass(ESIMDOptimizeVecArgCallConvPass{});
  FunctionPassManager MainFPM;
  MainFPM.addPass(ESIMDLowerLoadStorePass{});

  if (Optimize) {
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

Expected<ModuleDesc> linkModules(ModuleDesc MD1, ModuleDesc MD2) {
  std::vector<std::string> Names;
  MD1.saveEntryPointNames(Names);
  MD2.saveEntryPointNames(Names);
  bool LinkError =
      llvm::Linker::linkModules(MD1.getModule(), MD2.releaseModulePtr());

  if (LinkError)
    return createStringError("Linking of modules failed.");

  ModuleDesc Res(MD1.releaseModulePtr(), std::move(Names));
  Res.assignMergedProperties(MD1, MD2);
  Res.Name = (Twine("linked[") + MD1.Name + "," + MD2.Name + "]").str();
  return Res;
}

} // anonymous namespace

// When ESIMD code was separated from the regular SYCL code,
// we can safely process ESIMD part.
bool sycl::lowerESIMDConstructs(ModuleDesc &MD, bool Optimize,
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
  ModulePassManager MPM = buildESIMDLoweringPipeline(Optimize, SplitESIMD);
  PreservedAnalyses Res = MPM.run(MD.getModule(), MAM);

  // GenXSPIRVWriterAdaptor pass replaced some functions with "rewritten"
  // versions so the entry point table must be rebuilt.
  MD.rebuildEntryPoints(Names);
  return !Res.areAllPreserved();
}

Expected<SmallVector<ModuleDesc, 2>> llvm::sycl::handleESIMD(
    ModuleDesc MDesc, IRSplitMode SplitMode, bool EmitOnlyKernelsAsEntryPoints,
    bool AllowDeviceImageDependencies, bool LowerESIMD, bool SplitESIMD,
    bool OptimizeESIMDModule, bool &Modified, bool &SplitOccurred) {
  SmallVector<ModuleDesc, 2> Result =
      splitByESIMD(std::move(MDesc), EmitOnlyKernelsAsEntryPoints,
                   AllowDeviceImageDependencies);

  assert(Result.size() <= 2 &&
         "Split modules aren't expected to be more than 2.");
  if (Result.size() == 2 && SplitOccurred &&
      SplitMode == module_split::SPLIT_PER_KERNEL && !SplitESIMD)
    return createStringError("SYCL and ESIMD entry points detected with "
                             "-split-mode=per-kernel and -split-esimd=false. "
                             "So -split-esimd=true is mandatory.");

  SplitOccurred |= Result.size() > 1;

  for (auto &MD : Result) {
#ifdef LLVM_ENABLE_DUMP
    dumpEntryPoints(MD.entries(), MD.Name.c_str(), 4);
#endif // LLVM_ENABLE_DUMP
    if (LowerESIMD && MD.isESIMD())
      Modified |= lowerESIMDConstructs(MD, OptimizeESIMDModule, SplitESIMD);
  }

  if (SplitESIMD || Result.size() == 1)
    return Result;

  // SYCL/ESIMD splitting is not requested, link back into single module.
  int ESIMDInd = Result[0].isESIMD() ? 0 : 1;
  int SYCLInd = 1 - ESIMDInd;
  assert(Result[SYCLInd].isSYCL() &&
         "no non-ESIMD module as a result ESIMD split?");

  // Make sure that no link conflicts occur.
  Result[ESIMDInd].renameDuplicatesOf(Result[SYCLInd].getModule(), ".esimd");
  auto LinkedOrErr = linkModules(std::move(Result[0]), std::move(Result[1]));
  if (!LinkedOrErr)
    return LinkedOrErr.takeError();

  ModuleDesc &Linked = *LinkedOrErr;
  Linked.restoreLinkageOfDirectInvokeSimdTargets();
  std::vector<std::string> Names;
  Linked.saveEntryPointNames(Names);
  // cleanup may remove some entry points, need to save/rebuild
  Linked.cleanup(AllowDeviceImageDependencies);
  Linked.rebuildEntryPoints(Names);
  Result.clear();
  Result.emplace_back(std::move(Linked));
#ifdef LLVM_ENABLE_DUMP
  dumpEntryPoints(Result.back().entries(), Result.back().Name.c_str(), 4);
#endif // LLVM_ENABLE_DUMP
  Modified = true;

  return Result;
}
