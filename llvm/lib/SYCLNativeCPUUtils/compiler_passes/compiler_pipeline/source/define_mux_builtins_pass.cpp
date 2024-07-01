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

#include <compiler/utils/builtin_info.h>
#include <compiler/utils/define_mux_builtins_pass.h>
#include <llvm/IR/Module.h>

#define DEBUG_TYPE "define-mux-builtins"

using namespace llvm;

PreservedAnalyses compiler::utils::DefineMuxBuiltinsPass::run(
    Module &M, ModuleAnalysisManager &AM) {
  bool Changed = false;
  auto &BI = AM.getResult<BuiltinInfoAnalysis>(M);

  auto functionNeedsDefining = [&BI](Function &F) {
    return F.isDeclaration() && !F.isIntrinsic() &&
           BI.isMuxBuiltinID(BI.analyzeBuiltin(F).ID);
  };

  // Define all mux builtins
  for (auto &F : M.functions()) {
    if (!functionNeedsDefining(F)) {
      continue;
    }
    LLVM_DEBUG(dbgs() << "  Defining mux builtin: " << F.getName() << "\n";);

    // Define the builtin. If it declares any new dependent builtins, those
    // will be appended to the module's function list and so will be
    // encountered by later iterations.
    auto Builtin = BI.analyzeBuiltin(F);
    if (BI.defineMuxBuiltin(Builtin.ID, M, Builtin.mux_overload_info)) {
      Changed = true;
    }
  }

  // While declaring any builtins should go to the end of the module's list of
  // functions, it's not technically impossible for something else to happen.
  // As such, assert that we are leaving the module in the state we are
  // contractually obliged to: with all functions that need defining having
  // been defined.
  assert(all_of(M.functions(),
                [&](Function &F) {
                  return F.isDeclaration() || !functionNeedsDefining(F);
                }) &&
         "Did not define a function that requires it");

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
