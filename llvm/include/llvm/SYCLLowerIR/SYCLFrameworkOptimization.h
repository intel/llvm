//== SYCLFrameworkOptimization.h - Utility Pass for SYCL framework optimization
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This header defines utility pass for SYCL framework optimization removing
/// optnone and noinline attributes from function marked with
/// "!sycl-framework" metadata.
///
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_SYCL_FRAMEWORK_OPTIMIZATION_H
#define LLVM_SYCL_FRAMEWORK_OPTIMIZATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {
namespace sycl {

class RemoveFuncAttrsFromSYCLFrameworkFuncs
    : public PassInfoMixin<RemoveFuncAttrsFromSYCLFrameworkFuncs> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace sycl
} // namespace llvm

#endif // LLVM_SYCL_FRAMEWORK_OPTIMIZATION_H
