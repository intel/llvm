//===----- UtilsSYCLNativeCPU.h - Pass pipeline for SYCL Native CPU ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions and constants for SYCL Native CPU.
//
//===----------------------------------------------------------------------===//
#pragma once
#include "llvm/ADT/Twine.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/OptimizationLevel.h"

namespace llvm {
namespace sycl {
namespace utils {

void addSYCLNativeCPUBackendPasses(ModulePassManager &MPM,
                                   ModuleAnalysisManager &MAM,
                                   OptimizationLevel OptLevel);

constexpr char SYCLNATIVECPUSUFFIX[] = ".SYCLNCPU";
constexpr char SYCLNATIVECPUKERNEL[] = ".NativeCPUKernel";
constexpr char SYCLNATIVECPUPREFIX[] = "__dpcpp_nativecpu";
inline llvm::Twine addSYCLNativeCPUSuffix(StringRef S) {
  if (S.starts_with(SYCLNATIVECPUPREFIX) || S.ends_with(SYCLNATIVECPUKERNEL))
    return S;
  return llvm::Twine(S, SYCLNATIVECPUSUFFIX);
}

inline bool isSYCLNativeCPU(const Module &M) {
  return M.getModuleFlag("is-native-cpu") != nullptr;
}

} // namespace utils
} // namespace sycl
} // namespace llvm
