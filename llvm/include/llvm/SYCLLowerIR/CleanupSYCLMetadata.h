//===---------- CleanupSYCLMetadata.h - CleanupSYCLMetadata Pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Cleanup SYCL compiler internal metadata inserted by the frontend as it will
// never be used in the compilation ever again
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_CLEANUP_SYCL_METADATA
#define LLVM_CLEANUP_SYCL_METADATA

#include "llvm/IR/PassManager.h"

namespace llvm {

class CleanupSYCLMetadataPass : public PassInfoMixin<CleanupSYCLMetadataPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_CLEANUP_SYCL_METADATA
