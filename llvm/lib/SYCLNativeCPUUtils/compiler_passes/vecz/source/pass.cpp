// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "vecz/pass.h"

#include <compiler/utils/attributes.h>
#include <compiler/utils/builtin_info.h>
#include <compiler/utils/device_info.h>
#include <compiler/utils/metadata.h>
#include <compiler/utils/sub_group_analysis.h>
#include <compiler/utils/vectorization_factor.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>

#include <cstdlib>
#include <functional>
#include <optional>
#include <tuple>

#include "vectorization_context.h"
#include "vectorization_helpers.h"
#include "vectorization_unit.h"
#include "vectorizer.h"
#include "vecz/vecz_choices.h"
#include "vecz/vecz_target_info.h"
#include "vecz_pass_builder.h"

#define DEBUG_TYPE "vecz"

using namespace llvm;

/// @brief Provide debug logging for Vecz's PassManager
///
/// This flag is intended for testing and debugging purposes.
static cl::opt<bool> DebugVeczPipeline(
    "debug-vecz-pipeline",
    cl::desc("Enable debug logging of the vecz PassManager"));

/// @brief Provide debug logging for Vecz's PassManager
///
/// This flag specifies a textual description of the optimization pass pipeline
/// to run over the kernel.
static cl::opt<std::string> VeczPassPipeline(
    "vecz-passes",
    cl::desc(
        "A textual description of the pass pipeline. To have analysis passes "
        "available before a certain pass, add 'require<foo-analysis>'."));

namespace vecz {
using FnVectorizationResult =
    std::pair<Function *, compiler::utils::VectorizationFactor>;
AnalysisKey VeczPassOptionsAnalysis::Key;

PreservedAnalyses RunVeczPass::run(Module &M, ModuleAnalysisManager &MAM) {
  auto getVeczOptions = MAM.getResult<VeczPassOptionsAnalysis>(M);
  auto preserved = PreservedAnalyses::none();
  // Cache the current set of functions as the vectorizer will insert new ones,
  // which we don't want to revisit.
  SmallVector<std::pair<Function *, llvm::SmallVector<VeczPassOptions, 1>>, 4>
      FnOpts;
  for (auto &Fn : M.functions()) {
    llvm::SmallVector<VeczPassOptions, 1> Opts;
    if (!getVeczOptions(Fn, MAM, Opts)) {
      continue;
    }
    FnOpts.emplace_back(std::make_pair(&Fn, std::move(Opts)));
  }

  ModulePassManager PM;

  auto &device_info = MAM.getResult<compiler::utils::DeviceInfoAnalysis>(M);
  TargetInfo *target_info = MAM.getResult<vecz::TargetInfoAnalysis>(M);
  assert(target_info && "Missing TargetInfo");
  auto &builtin_info = MAM.getResult<compiler::utils::BuiltinInfoAnalysis>(M);

  VectorizationContext Ctx(M, *target_info, builtin_info);
  VeczPassMachinery Mach(M.getContext(), target_info->getTargetMachine(), Ctx,
                         /*verifyEach*/ false,
                         DebugVeczPipeline
                             ? compiler::utils::DebugLogging::Normal
                             : compiler::utils::DebugLogging::None);
  Mach.initializeStart();
  Mach.getMAM().registerPass([&device_info] {
    return compiler::utils::DeviceInfoAnalysis(device_info);
  });
  Mach.initializeFinish();

  // Forcibly compute the DeviceInfoAnalysis so that cached retrievals work.
  PM.addPass(
      RequireAnalysisPass<compiler::utils::DeviceInfoAnalysis, Module>());

  const bool Check = VeczPassPipeline.empty();
  if (Check) {
    if (!buildPassPipeline(PM)) {
      return PreservedAnalyses::all();
    }
  } else {
    if (auto Err = Mach.getPB().parsePassPipeline(PM, VeczPassPipeline)) {
      // NOTE this is a command line user error print, not a debug print.
      // We may want to hoist this out of Vecz once CA-4134 is resolved.
      errs() << "vecz pipeline: " << toString(std::move(Err)) << "\n";
      return PreservedAnalyses::all();
    }
  }

  // Create the vectorization units and clone the kernels
  using ResultTy =
      SmallVector<std::pair<VectorizationUnit *, VeczPassOptions *>, 2>;
  SmallDenseMap<Function *, ResultTy, 2> Results;
  for (auto &P : FnOpts) {
    Function *Fn = P.first;
    ResultTy T;
    Results.insert(std::make_pair(Fn, std::move(T)));
    for (auto &Opts : P.second) {
      // If we've been given an auto width, try and fit it to any requirements
      // that the kernel/device places on its sub-groups.
      if (Opts.vecz_auto) {
        if (auto AutoSGOpts = getAutoSubgroupSizeOpts(*Fn, MAM)) {
          Opts = *AutoSGOpts;
        }
      }

      auto *const VU =
          createVectorizationUnit(Ctx, Fn, Opts, Mach.getFAM(), Check);
      if (!VU) {
        LLVM_DEBUG(llvm::dbgs() << Fn->getName() << " was not vectorized\n");
        continue;
      }
      Results[Fn].emplace_back(std::make_pair(VU, &Opts));

      if (auto *const VecFn = vecz::cloneFunctionToVector(*VU)) {
        VU->setVectorizedFunction(VecFn);

        // Allows the Vectorization Unit Analysis to work on the vector kernel
        Ctx.setActiveVU(VecFn, VU);
      } else {
        LLVM_DEBUG(llvm::dbgs() << Fn->getName() << " could not be cloned\n");
      }
    }
  }

  // Vectorize everything
  PM.run(M, Mach.getMAM());

  auto AllOnModule = llvm::PreservedAnalyses::allInSet<AllAnalysesOn<Module>>();
  auto eraseFailed = [&](VectorizationUnit *VU) {
    Function *VectorizedFn = VU->vectorizedFunction();
    if (VectorizedFn) {
      // If we fail to vectorize a function, we still cloned and then
      // deleted it which affects internal addresses. The module has changed
      // and we can't cache any analyses.
      Mach.getFAM().invalidate(*VectorizedFn, llvm::PreservedAnalyses::none());
      // Remove the partially-vectorized function if something went wrong.
      Ctx.clearActiveVU(VectorizedFn);
      VU->setVectorizedFunction(nullptr);
      VectorizedFn->eraseFromParent();
    }
    MAM.invalidate(M, AllOnModule);
  };

  // Fix up the metadata and clean out any dead kernels
  for (auto &P : Results) {
    auto &Result = P.second;
    for (auto &R : Result) {
      VectorizationUnit *VU = R.first;
      trackVeczSuccessFailure(*VU);
      if (!createVectorizedFunctionMetadata(*VU)) {
        LLVM_DEBUG(dbgs() << P.first->getName() << " failed to vectorize\n");
        eraseFailed(VU);
      }
    }
  }
  return PreservedAnalyses::none();
}

PreservedAnalyses VeczPassOptionsPrinterPass::run(Module &M,
                                                  ModuleAnalysisManager &MAM) {
  auto getVeczOptions = MAM.getResult<VeczPassOptionsAnalysis>(M);
  for (auto &F : M.functions()) {
    OS << "Function '" << F.getName() << "'";
    llvm::SmallVector<VeczPassOptions, 1> Opts;
    if (!getVeczOptions(F, MAM, Opts)) {
      OS << " will not be vectorized\n";
      continue;
    }

    OS << " will be vectorized {\n";
    for (auto &O : Opts) {
      OS << "  VF = ";
      if (O.factor.isScalable()) {
        OS << "vscale x ";
      }
      OS << O.factor.getKnownMin();

      if (O.vecz_auto) {
        OS << ", (auto)";
      }

      OS << ", vec-dim = " << O.vec_dim_idx;

      if (O.local_size) {
        OS << ", local-size = " << O.local_size;
      }

      OS << ", choices = [";
      OS.tell();
      auto AvailChoices = VectorizationChoices::queryAvailableChoices();
      unsigned NumChoices = 0;

      for (auto &C : AvailChoices) {
        if (!O.choices.isEnabled(C.number)) {
          continue;
        }
        if (!NumChoices) {
          OS << "\n    ";
        } else {
          OS << ",";
        }
        OS << C.name;
        NumChoices++;
      }
      // Pretty-print the list of choices on one line if empty, else formatted
      // across several lines. Always end with a newline, meaning the options
      // are closed off with a '}' on the first column.
      if (NumChoices) {
        OS << "\n  ]\n";
      } else {
        OS << "]\n";
      }
    }
    OS << "}\n";
  }

  return PreservedAnalyses::all();
}

std::optional<VeczPassOptions> getReqdSubgroupSizeOpts(Function &F) {
  if (auto reqd_sg_size = compiler::utils::getReqdSubgroupSize(F)) {
    vecz::VeczPassOptions vecz_opts;
    // Disable auto - we want a specific width
    vecz_opts.vecz_auto = false;
    vecz_opts.vec_dim_idx = 0;
    // If we can't vectorize to the required sub-group size then we must bail.
    if (*reqd_sg_size % compiler::utils::getMuxSubgroupSize(F)) {
      return std::nullopt;
    }
    // Else we must vectorize such that we multiply the existing mux sub-group
    // size up to the required one.
    vecz_opts.factor = compiler::utils::VectorizationFactor::getFixedWidth(
        *reqd_sg_size / compiler::utils::getMuxSubgroupSize(F));
    vecz_opts.choices.enable(vecz::VectorizationChoices::eDivisionExceptions);
    return vecz_opts;
  }
  return std::nullopt;
}

std::optional<VeczPassOptions> getAutoSubgroupSizeOpts(
    Function &F, ModuleAnalysisManager &AM) {
  // If there's a required sub-group size, we must return a vectorization
  // factor that gets us there.
  if (auto opts = getReqdSubgroupSizeOpts(F)) {
    return opts;
  }

  auto &M = *F.getParent();
  const auto &GSGI = AM.getResult<compiler::utils::SubgroupAnalysis>(M);

  // If the function doesn't use sub-groups (from the user's perspective) then
  // we don't need to adhere to a specific sub-group size.
  if (!GSGI.usesSubgroups(F)) {
    return std::nullopt;
  }

  // Use the device's sub-group sizes to determine which to vectorize to.
  auto &DI = AM.getResult<compiler::utils::DeviceInfoAnalysis>(M);

  // We don't force devices to support any sub-group sizes.
  if (DI.reqd_sub_group_sizes.empty()) {
    return std::nullopt;
  }

  vecz::VeczPassOptions vecz_opts;
  vecz_opts.vec_dim_idx = 0;
  // Disable auto - we want a specific width
  vecz_opts.vecz_auto = false;
  // Enable some default choices
  vecz_opts.choices.enable(vecz::VectorizationChoices::eDivisionExceptions);

  // Now try and choose the best width.
  std::optional<unsigned> best_width;
  const auto mux_sub_group_size = compiler::utils::getMuxSubgroupSize(F);

  auto can_produce_legal_width = [&mux_sub_group_size](unsigned size) {
    // We only support vectorization widths where there's a clean multiple, and
    // we can vectorize *up* to the desired size - we can't shrink the
    // sub-group size by vectorizing.
    return size >= mux_sub_group_size && (size % mux_sub_group_size) == 0;
  };

  for (auto size : DI.reqd_sub_group_sizes) {
    if (!can_produce_legal_width(size)) {
      continue;
    }
    const unsigned candidate_width = size / mux_sub_group_size;
    // Try and choose at least one width.
    if (!best_width) {
      best_width = candidate_width;
      continue;
    }

    // Prefer non-scalar widths.
    if (best_width == 1 && candidate_width > 1) {
      best_width = candidate_width;
      continue;
    }

    // If we have a required work-group size, prefer one that will fit well
    // with that.
    if (auto wgs = compiler::utils::parseRequiredWGSMetadata(F)) {
      const uint64_t local_size_x = wgs.value()[0];
      const bool best_fits = !(local_size_x % *best_width);
      const bool cand_fits = !(local_size_x % candidate_width);
      if (!best_fits && cand_fits) {
        best_width = candidate_width;
        continue;
      } else if (best_fits && !cand_fits) {
        continue;
      }
    }

    // Else, prefer powers of two.
    if (!isPowerOf2_32(*best_width) && isPowerOf2_32(candidate_width)) {
      best_width = candidate_width;
      continue;
    }
  }

  // Return nothing if we couldn't find a good, legal, width.
  if (!best_width) {
    return std::nullopt;
  }

  vecz_opts.factor =
      compiler::utils::VectorizationFactor::getFixedWidth(*best_width);

  return vecz_opts;
}

}  // namespace vecz
