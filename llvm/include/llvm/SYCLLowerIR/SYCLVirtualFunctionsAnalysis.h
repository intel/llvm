//===---------------- SYCLVirtualFunctionsAnalysis.h ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emits diagnostics for improper use of virtual functions in SYCL device code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_VIRTUAL_FUNCTIONS_ANALYSIS_H
#define LLVM_SYCL_VIRTUAL_FUNCTIONS_ANALYSIS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLVirtualFunctionsAnalysisPass
    : public PassInfoMixin<SYCLVirtualFunctionsAnalysisPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_SYCL_VIRTUAL_FUNCTIONS_ANALYSIS_H
