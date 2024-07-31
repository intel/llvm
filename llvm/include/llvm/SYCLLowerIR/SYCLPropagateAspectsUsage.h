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

#include <set>

namespace llvm {

class SYCLPropagateAspectsUsagePass
    : public PassInfoMixin<SYCLPropagateAspectsUsagePass> {
public:
  SYCLPropagateAspectsUsagePass(bool FP64ConvEmu = false,
                                std::set<StringRef> ExcludeAspects = {},
                                bool ValidateAspects = true,
                                StringRef OptionsString = {})
      : FP64ConvEmu{FP64ConvEmu}, ExcludedAspects{std::move(ExcludeAspects)},
        ValidateAspectUsage{ValidateAspects} {
    OptionsString.split(this->TargetFixedAspects, ',', /*MaxSplit=*/-1,
                        /*KeepEmpty=*/false);
  };
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

private:
  bool FP64ConvEmu;
  std::set<StringRef> ExcludedAspects;
  const bool ValidateAspectUsage;
  SmallVector<StringRef, 8> TargetFixedAspects;
};

} // namespace llvm

#endif // LLVM_SYCL_PROPAGATE_ASPECTS_USAGE_H
