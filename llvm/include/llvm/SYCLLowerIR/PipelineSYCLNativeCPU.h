//===----- PipelineSYCLNativeCPU.h - Pass pipeline for SYCL Native CPU ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the pass pipeline used when lowering device code for SYCL Native 
// CPU. 
//
//===----------------------------------------------------------------------===//
#include "llvm/Target/TargetMachine.h"

namespace llvm {
void addSYCLNativeCPUBackendPasses(ModulePassManager &MPM,
                                   ModuleAnalysisManager &MAM);
} // namespace llvm
