//= SpecializationConstants.cpp - Processing of SYCL Specialization Constants //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLPostLink/SpecializationConstants.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/SYCLLowerIR/SpecConstants.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"

#include <optional>

using namespace llvm;

namespace {

bool lowerSpecConstants(module_split::ModuleDesc &MD,
                        SpecConstantsPass::HandlingMode Mode) {
  ModulePassManager RunSpecConst;
  ModuleAnalysisManager MAM;
  SpecConstantsPass SCP(Mode);
  // Register required analysis.
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  RunSpecConst.addPass(std::move(SCP));

  // Perform the spec constant intrinsics transformation on resulting module.
  PreservedAnalyses Res = RunSpecConst.run(MD.getModule(), MAM);
  MD.Props.SpecConstsMet = !Res.areAllPreserved();
  return MD.Props.SpecConstsMet;
}

/// Function generates the copy of the given \p MD where all uses of
/// Specialization Constants are replaced by corresponding default values.
/// If the Module in \p MD doesn't contain specialization constants then
/// std::nullopt is returned.
std::optional<module_split::ModuleDesc>
cloneModuleWithSpecConstsReplacedByDefaultValues(
    const module_split::ModuleDesc &MD) {
  std::optional<module_split::ModuleDesc> NewMD;
  if (!checkModuleContainsSpecConsts(MD.getModule()))
    return NewMD;

  NewMD = MD.clone();
  NewMD->setSpecConstantDefault(true);

  ModulePassManager MPM;
  ModuleAnalysisManager MAM;
  SpecConstantsPass SCP(SpecConstantsPass::HandlingMode::default_values);
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  MPM.addPass(std::move(SCP));
  MPM.addPass(StripDeadPrototypesPass());

  PreservedAnalyses Res = MPM.run(NewMD->getModule(), MAM);
  NewMD->Props.SpecConstsMet = !Res.areAllPreserved();
  assert(NewMD->Props.SpecConstsMet &&
         "This property should be true since the presence of SpecConsts "
         "has been checked before the run of the pass");
  NewMD->rebuildEntryPoints();
  return NewMD;
}

} // namespace

bool llvm::sycl::handleSpecializationConstants(
    SmallVectorImpl<module_split::ModuleDesc> &MDs,
    std::optional<SpecConstantsPass::HandlingMode> Mode,
    bool GenerateModuleDescWithDefaultSpecConsts,
    SmallVectorImpl<module_split::ModuleDesc> *NewModuleDescs) {
  [[maybe_unused]] bool AreArgumentsCompatible =
      (GenerateModuleDescWithDefaultSpecConsts && NewModuleDescs) ||
      (!GenerateModuleDescWithDefaultSpecConsts && !NewModuleDescs);
  assert(AreArgumentsCompatible &&
         "NewModuleDescs pointer is nullptr iff "
         "GenerateModuleDescWithDefaultSpecConsts is false.");

  bool Modified = false;
  for (module_split::ModuleDesc &MD : MDs) {
    if (GenerateModuleDescWithDefaultSpecConsts)
      if (std::optional<module_split::ModuleDesc> NewMD =
              cloneModuleWithSpecConstsReplacedByDefaultValues(MD);
          NewMD)
        NewModuleDescs->push_back(std::move(*NewMD));

    if (Mode)
      Modified |= lowerSpecConstants(MD, *Mode);
  }

  return Modified;
}
