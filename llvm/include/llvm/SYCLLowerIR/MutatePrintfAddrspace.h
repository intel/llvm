//===------- MutatePrintfAddrspace.h - SYCL printf AS mutation Pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which detects non-constant address space
// literals usage for the first argument of SYCL experimental printf
// function, and moves the string literal to constant address
// space. This a temporary solution for printf's support of generic
// address space literals; the pass should be dropped once SYCL device
// backends learn to handle the generic address-spaced argument properly.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLMutatePrintfAddrspacePass
    : public PassInfoMixin<SYCLMutatePrintfAddrspacePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createSYCLMutatePrintfAddrspaceLegacyPass();

} // namespace llvm
