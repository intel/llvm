//===---- FixABIMuxBuiltins.h - Fixup ABI issues with called mux builtins ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Creates calls to shuffle up/down/xor mux builtins taking into account ABI of the
// SYCL functions. For now this only is used for vector variants.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"


namespace llvm {

class FixABIMuxBuiltinsPass final
    : public llvm::PassInfoMixin<FixABIMuxBuiltinsPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Module &, llvm::ModuleAnalysisManager &);
};

} // namespace llvm

