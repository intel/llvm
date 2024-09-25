//===- SYCLCreateNVVMAnnotations.h - SYCLCreateNVVMAnnotationsPass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers function metadata to NVVM annotations
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_SYCL_CREATE_NVVM_ANNOTATIONS_H
#define LLVM_SYCL_CREATE_NVVM_ANNOTATIONS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLCreateNVVMAnnotationsPass
    : public PassInfoMixin<SYCLCreateNVVMAnnotationsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_SYCL_CREATE_NVVM_ANNOTATIONS_H
