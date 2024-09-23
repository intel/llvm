//===---- ConvertToMuxBuiltinsSYCLNativeCPU.h - Convert to Mux Builtins ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts SPIRV builtins to Mux builtins used by the oneAPI Construction
// Kit for SYCL Native CPU
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class ModulePass;

class ConvertToMuxBuiltinsSYCLNativeCPUPass
    : public PassInfoMixin<ConvertToMuxBuiltinsSYCLNativeCPUPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm
