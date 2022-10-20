//===---- SYCLPropagateAspectsUsage.cpp - SYCLPropagateAspectsUsage Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass propagates optional kernel features metadata through a module call graph
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_SYCL_PROPAGATE_ASPECTS_USAGE_H
#define LLVM_SYCL_PROPAGATE_ASPECTS_USAGE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

struct SPAUOptions {
  SmallVector<StringRef, 8> Targets;
};

class SYCLPropagateAspectsUsagePass
    : public PassInfoMixin<SYCLPropagateAspectsUsagePass> {
public:
  SYCLPropagateAspectsUsagePass(StringRef Opts = {}) {
    this->Opts = parseOpts(Opts);
  };
  SPAUOptions parseOpts(StringRef Params);
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

private:
  SPAUOptions Opts;
};

} // namespace llvm

#endif // LLVM_SYCL_PROPAGATE_ASPECTS_USAGE_H
