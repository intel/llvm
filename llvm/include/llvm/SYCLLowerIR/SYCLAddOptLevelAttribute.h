//===----- SYCLAddOptLevelAttribute.h - SYCLAddOptLevelAttribute Pass -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass adds 'sycl-optlevel' function attribute based on optimization level
// passed in.
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_SYCL_ADD_OPT_LEVEL_ATTRIBUTE_H
#define LLVM_SYCL_ADD_OPT_LEVEL_ATTRIBUTE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLAddOptLevelAttributePass
    : public PassInfoMixin<SYCLAddOptLevelAttributePass> {
public:
  SYCLAddOptLevelAttributePass(int OptLevel = -1) : OptLevel{OptLevel} {};
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

private:
  int OptLevel;
};

} // namespace llvm

#endif // LLVM_SYCL_ADD_OPT_LEVEL_ATTRIBUTE_H
