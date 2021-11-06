//===---- DelimitESIMDandSYCL.h - delimit ESIMD and SYCL code -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass "delimits" SYCL and ESIMD code within a module - makes sure
// callgraphs starting from SYCL and ESIMD roots do not intersect, cloning
// functions participating in both callgraphs. Functions from ESIMD callgrpahs
// are also marked with "sycl_explicit_simd" metadata. This is done for easier
// JIT-time SYCL/ESIMD device code redirection to scalar/vector backends.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCLLOWERIR_DELIMITESIMDANDSYCL_H
#define LLVM_SYCLLOWERIR_DELIMITESIMDANDSYCL_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class DelimitESIMDandSYCLPass : public PassInfoMixin<DelimitESIMDandSYCLPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

ModulePass *createDelimitESIMDandSYCLPass();
void initializeDelimitESIMDandSYCLLegacyPassPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_SYCLLOWERIR_DELIMITESIMDANDSYCL_H
