//===---- LowerInvokeSimd.h - lower invoke_simd calls ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass brings __builtin_invoke_simd intrinsic call to the form consumable
// by the back ends:
// - determines the "invokee" (call target) - actual function address link-time
//   constant (it can be represented as an SSA value in the input IR)
// See more comments in the implementation.
//===----------------------------------------------------------------------===//

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {
class SYCLLowerInvokeSimdPass : public PassInfoMixin<SYCLLowerInvokeSimdPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createSYCLLowerInvokeSimdPass();
void initializeSYCLLowerInvokeSimdLegacyPassPass(PassRegistry &);

// Attribute added to functions which are known to be invoke_simd targets.
constexpr char INVOKE_SIMD_DIRECT_TARGET_ATTR[] = "__invoke_simd_target";

} // namespace llvm
