//===----- SYCLITTAnnotations.h - SYCL Instrumental Annotations Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which adds instrumental calls to annotate SYCL
// synchronization instrucations. This can be used for kernel profiling.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLITTAnnotationsPass : public PassInfoMixin<SYCLITTAnnotationsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createSYCLITTAnnotationsPass();

} // namespace llvm
