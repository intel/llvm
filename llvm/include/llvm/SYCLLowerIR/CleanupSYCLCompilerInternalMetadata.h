//===- CleanupSYCLCompilerInternalMetadata.h - SYCLRemoveAspectsMD Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Cleanup SYCL compiler internal metadata (usually inserted by sycl-post-link)
// as it will never be used in the compilation ever again
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_CLEANUP_SYCL_COMPILER_INTERNAL_METADATA
#define LLVM_CLEANUP_SYCL_COMPILER_INTERNAL_METADATA

#include "llvm/IR/PassManager.h"

#include <set>

namespace llvm {

class CleanupSYCLCompilerInternalMetadataPass
    : public PassInfoMixin<CleanupSYCLCompilerInternalMetadataPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_CLEANUP_SYCL_COMPILER_INTERNAL_METADATA
