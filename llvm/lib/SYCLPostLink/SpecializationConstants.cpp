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
using namespace llvm::module_split;

namespace {

bool lowerSpecConstants(ModuleDesc &MD, SpecConstantsPass::HandlingMode Mode) {
  ModulePassManager RunSpecConst;
  ModuleAnalysisManager MAM;
  SpecConstantsPass SCP(Mode);
  // Register required analysis.
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  RunSpecConst.addPass(std::move(SCP));

  // Perform the specialization constant intrinsics transformation on resulting
  // module.
  PreservedAnalyses Res = RunSpecConst.run(MD.getModule(), MAM);
  MD.Props.SpecConstsMet = !Res.areAllPreserved();
  return MD.Props.SpecConstsMet;
}

/// Function generates the copy of the given \p MD where all uses of
/// Specialization constants are replaced by corresponding default values.
/// If the Module in \p MD doesn't contain specialization constants then
/// std::nullopt is returned.
std::optional<std::unique_ptr<ModuleDesc>>
cloneModuleWithSpecConstsReplacedByDefaultValues(const ModuleDesc &MD) {
  if (!checkModuleContainsSpecConsts(MD.getModule()))
    return std::nullopt;

  std::unique_ptr<ModuleDesc> NewMD = MD.clone();
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
         "SpecConstsMet should be true since the presence of SpecConsts "
         "has been checked before the run of the pass");
  NewMD->rebuildEntryPoints();
  return std::move(NewMD);
}

} // namespace

bool llvm::sycl::handleSpecializationConstants(
    SmallVectorImpl<std::unique_ptr<ModuleDesc>> &MDs,
    std::optional<SpecConstantsPass::HandlingMode> Mode,
    SmallVectorImpl<std::unique_ptr<ModuleDesc>> &NewModuleDescs,
    bool GenerateModuleDescWithDefaultSpecConsts) {
  bool Modified = false;
  for (std::unique_ptr<ModuleDesc> &MD : MDs) {
    if (GenerateModuleDescWithDefaultSpecConsts)
      if (std::optional<std::unique_ptr<ModuleDesc>> NewMD =
              cloneModuleWithSpecConstsReplacedByDefaultValues(*MD))
        NewModuleDescs.push_back(std::move(*NewMD));

    if (Mode)
      Modified |= lowerSpecConstants(*MD, *Mode);
  }

  return Modified;
}
