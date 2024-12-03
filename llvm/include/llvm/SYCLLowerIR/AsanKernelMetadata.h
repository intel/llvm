//===-- AsanKernelMetadata.h - fix kernel medatadata for sanitizer ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass fixes attributes and metadata of the global variable
// "__AsanKernelMetadata"
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/PassManager.h"

namespace llvm {

class AsanKernelMetadataPass : public PassInfoMixin<AsanKernelMetadataPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm
