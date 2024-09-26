//==-- ErrorHandling.h - Helpers for error handling in the JIT compiler ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_HELPER_ERRORHANDLING_H
#define SYCL_FUSION_JIT_COMPILER_HELPER_ERRORHANDLING_H

#include "ConfigHelper.h"
#include "Options.h"
#include <llvm/Support/Error.h>

///
/// Call a function returning llvm::Expected and propagate the error if it
/// fails.
///
/// @param Var The variable the value should be assigned to if the call does not
/// fail.
/// @param F The function call.
#define PROPAGATE_ERROR(Var, F)                                                \
  auto VarOrErr = F;                                                           \
  if (auto Err = VarOrErr.takeError()) {                                       \
    return std::move(Err);                                                     \
  }                                                                            \
  auto &Var = *VarOrErr;

namespace helper {

static inline void printDebugMessage(llvm::StringRef Message) {
  if (jit_compiler::ConfigHelper::get<
          jit_compiler::option::JITEnableVerbose>()) {
    llvm::errs() << "JIT DEBUG: " << Message << "\n";
  }
}

} // namespace helper

#endif // SYCL_FUSION_JIT_COMPILER_HELPER_ERRORHANDLING_H
