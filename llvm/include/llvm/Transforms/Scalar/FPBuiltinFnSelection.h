//===- FPBuiltinFnSelection.h - fpbuiltin intrinsic lowering pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements alternate math library implementation selection for
// llvm.fpbuiltin.* intrinsics.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_SCALAR_FPBUILTINFNSELECTION_H
#define LLVM_TRANSFORMS_SCALAR_FPBUILTINFNSELECTION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;

struct FPBuiltinFnSelectionPass : PassInfoMixin<FPBuiltinFnSelectionPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_FPBUILTINFNSELECTION_H
