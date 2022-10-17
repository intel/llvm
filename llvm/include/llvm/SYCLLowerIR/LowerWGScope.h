//===-- LowerWGScope.h - lower work group-scope code ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CODEGEN_SYCLLOWERIR_LOWERWGCODE_H
#define CLANG_LIB_CODEGEN_SYCLLOWERIR_LOWERWGCODE_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class FunctionPass;

/// SPIRV target specific pass to transform work group-scope code to match SIMT
/// execution model semantics - this code must be executed once per work group.
class SYCLLowerWGScopePass : public PassInfoMixin<SYCLLowerWGScopePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &);
};

FunctionPass *createSYCLLowerWGScopePass();

} // namespace llvm

#endif // CLANG_LIB_CODEGEN_SYCLLOWERIR_LOWERWGCODE_H
