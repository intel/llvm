//===- SYCLLegalizeNonStandardIntegers.h - Legalize int types -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements legalization of non-standard integer types (i24, i48,
// etc.) by widening them to the next power-of-2 width.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_LEGALIZE_NON_STANDARD_INTEGERS_H
#define LLVM_SYCL_LEGALIZE_NON_STANDARD_INTEGERS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLLegalizeNonStandardIntegersPass
    : public PassInfoMixin<SYCLLegalizeNonStandardIntegersPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_SYCL_LEGALIZE_NON_STANDARD_INTEGERS_H
