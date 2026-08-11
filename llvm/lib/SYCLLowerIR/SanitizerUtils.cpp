//===------------ SanitizerUtils.cpp - sanitizer utility functions ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for device sanitizers.
//===----------------------------------------------------------------------===//
#include "llvm/SYCLLowerIR/SanitizerUtils.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

namespace llvm {
namespace sycl {

bool isModuleUsingAsan(const Module &M) {
  return any_of(M.globals(), [](const GlobalVariable &GV) {
    return GV.getName().starts_with(ASAN_KERNEL_METADATA_PREFIX);
  });
}

bool isModuleUsingMsan(const Module &M) {
  return any_of(M.globals(), [](const GlobalVariable &GV) {
    return GV.getName().starts_with(MSAN_KERNEL_METADATA_PREFIX);
  });
}

bool isModuleUsingTsan(const Module &M) {
  return any_of(M.globals(), [](const GlobalVariable &GV) {
    return GV.getName().starts_with(TSAN_KERNEL_METADATA_PREFIX);
  });
}

} // namespace sycl
} // namespace llvm
