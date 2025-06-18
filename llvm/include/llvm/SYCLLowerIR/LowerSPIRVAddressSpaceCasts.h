//===- LowerSPIRVAddressSpaceCasts.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCLLOWERIR_LOWERSPIRVADDRESSSPACECASTS_H
#define LLVM_SYCLLOWERIR_LOWERSPIRVADDRESSSPACECASTS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class LowerSPIRVAddressSpaceCastsPass
    : public PassInfoMixin<LowerSPIRVAddressSpaceCastsPass> {
public:
  PreservedAnalyses run(Function &M, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_SYCLLOWERIR_LOWERSPIRVADDRESSSPACECASTS_H
