//===---- FixABIMuxBuiltins.h - Fixup ABI issues with called mux builtins ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fixes up the ABI for any mux builtins which meet the architecture
// ABI but not the mux_* usage. For now this is restricted to mux_shuffle* 
// builtins which take a float2 input.
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

