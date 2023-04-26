//===------- EmitSYCLHCHeader.h - Emits the SYCL Native CPU helper header
//-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass emits the SYCL Native CPU helper header.
// The header mainly contains the definition for the handler function which
// allows to call the kernel extracted by the device compiler from the host
// runtime.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class ModulePass;

class EmitSYCLNativeCPUHeaderPass
    : public PassInfoMixin<EmitSYCLNativeCPUHeaderPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  EmitSYCLNativeCPUHeaderPass(const std::string &FileName)
      : NativeCPUHeaderName(FileName) {}
  EmitSYCLNativeCPUHeaderPass() = default;

private:
  std::string NativeCPUHeaderName;
};

} // namespace llvm
