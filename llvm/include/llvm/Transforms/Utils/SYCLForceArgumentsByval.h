//===- SYCLForceArgumentsByval.h - forces kernel arguments to be by value -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_TRANSFORMS_UTILS_SYCLFORCEARGUMENTSBYVAL_H
#define CLANG_TRANSFORMS_UTILS_SYCLFORCEARGUMENTSBYVAL_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLForceArgumentsByvalPass
    : public PassInfoMixin<SYCLForceArgumentsByvalPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &);
};

FunctionPass *createSYCLForceArgumentsByvalPass();

} // namespace llvm

#endif // CLANG_TRANSFORMS_UTILS_SYCLFORCEARGUMENTSBYVAL_H
