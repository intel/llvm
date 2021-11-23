//===---- LowerInvokeSimd.h - lower invoke_simd calls ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass brings __builtin_invoke_simd intrinsic call the the form consumable
// by the back ends:
// - determines the "invokee" (call target) - actual function address link-time
//   constant (it can be represented as an SSA value in the input IR)
//===----------------------------------------------------------------------===//

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class SYCLLowerInvokeSimdPass : public PassInfoMixin<SYCLLowerInvokeSimdPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createSYCLLowerInvokeSimdPass();
void initializeSYCLLowerInvokeSimdLegacyPassPass(PassRegistry &);
}
