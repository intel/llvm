//===--------- ESIMDVerifier.h - ESIMD-specific IR verification -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ESIMD verification pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCLLOWERIR_ESIMDVERIFIER_H
#define LLVM_SYCLLOWERIR_ESIMDVERIFIER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class ModulePass;

struct ESIMDVerifierPass : public PassInfoMixin<ESIMDVerifierPass> {
  ESIMDVerifierPass() {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
  static bool isRequired() { return true; }
};

ModulePass *createESIMDVerifierPass();

} // namespace llvm

#endif // LLVM_SYCLLOWERIR_ESIMDVERIFIER_H
