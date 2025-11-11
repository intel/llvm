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
#include "llvm/Support/FormatVariadic.h"
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

ModulePassManager
buildESIMDLoweringPipeline(const sycl::ESIMDProcessingOptions &Options) {
  ModulePassManager MPM;
  MPM.addPass(SYCLLowerESIMDPass(!Options.SplitESIMD));

  if (!Options.ForceDisableESIMDOpt) {
    FunctionPassManager FPM;
    FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }
  MPM.addPass(ESIMDOptimizeVecArgCallConvPass{});
  FunctionPassManager MainFPM;
  MainFPM.addPass(ESIMDLowerLoadStorePass{});

  if (!Options.ForceDisableESIMDOpt) {
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

Expected<std::unique_ptr<ModuleDesc>>
linkModules(std::unique_ptr<ModuleDesc> MD1, std::unique_ptr<ModuleDesc> MD2) {
  std::vector<std::string> Names;
  MD1->saveEntryPointNames(Names);
  MD2->saveEntryPointNames(Names);
  bool LinkError =
      llvm::Linker::linkModules(MD1->getModule(), MD2->releaseModulePtr());

  if (LinkError)
    return createStringError(
        formatv("link failed. Module names: {0}, {1}", MD1->Name, MD2->Name));

  auto Res =
      std::make_unique<ModuleDesc>(MD1->releaseModulePtr(), std::move(Names));
  Res->assignMergedProperties(*MD1, *MD2);
  Res->Name = (Twine("linked[") + MD1->Name + "," + MD2->Name + "]").str();
  return std::move(Res);
}

} // anonymous namespace

// When ESIMD code was separated from the regular SYCL code,
// we can safely process ESIMD part.
bool sycl::lowerESIMDConstructs(ModuleDesc &MD,
                                const sycl::ESIMDProcessingOptions &Options) {
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
  ModulePassManager MPM = buildESIMDLoweringPipeline(Options);
  PreservedAnalyses Res = MPM.run(MD.getModule(), MAM);

  // GenXSPIRVWriterAdaptor pass replaced some functions with "rewritten"
  // versions so the entry point table must be rebuilt.
  MD.rebuildEntryPoints(Names);
  return !Res.areAllPreserved();
}

Expected<SmallVector<std::unique_ptr<ModuleDesc>, 2>>
llvm::sycl::handleESIMD(std::unique_ptr<ModuleDesc> MDesc,
                        const sycl::ESIMDProcessingOptions &Options,
                        bool &Modified, bool &SplitOccurred) {
  SmallVector<std::unique_ptr<ModuleDesc>, 2> Result =
      splitByESIMD(std::move(MDesc), Options.EmitOnlyKernelsAsEntryPoints,
                   Options.AllowDeviceImageDependencies);

  assert(Result.size() <= 2 &&
         "Split modules aren't expected to be more than 2.");

  SplitOccurred |= Result.size() > 1;

  for (std::unique_ptr<ModuleDesc> &MD : Result)
    if (Options.LowerESIMD && MD->isESIMD())
      Modified |= lowerESIMDConstructs(*MD, Options);

  if (Options.SplitESIMD || Result.size() == 1)
    return std::move(Result);

  // SYCL/ESIMD splitting is not requested, link back into single module.
  int ESIMDInd = Result[0]->isESIMD() ? 0 : 1;
  int SYCLInd = 1 - ESIMDInd;
  assert(Result[SYCLInd]->isSYCL() &&
         "Result[SYCLInd]->isSYCL() expected to be true.");

  // Make sure that no link conflicts occur.
  Result[ESIMDInd]->renameDuplicatesOf(Result[SYCLInd]->getModule(), ".esimd");
  auto LinkedOrErr = linkModules(std::move(Result[0]), std::move(Result[1]));
  if (!LinkedOrErr)
    return LinkedOrErr.takeError();

  std::unique_ptr<ModuleDesc> &Linked = *LinkedOrErr;
  Linked->restoreLinkageOfDirectInvokeSimdTargets();
  std::vector<std::string> Names;
  Linked->saveEntryPointNames(Names);
  // Cleanup may remove some entry points, need to save/rebuild.
  Linked->cleanup(Options.AllowDeviceImageDependencies);
  Linked->rebuildEntryPoints(Names);
  Result.clear();
  Result.emplace_back(std::move(Linked));
  Modified = true;

  return std::move(Result);
}
