//==---- PassDebug.h - Helper for debug output from kernel fusion passes ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_PASSES_DEBUG_H
#define SYCL_FUSION_PASSES_DEBUG_H

namespace jit_compiler {

extern bool PassDebug;

} // namespace jit_compiler

#define FUSION_DEBUG(X)                                                        \
  if (::jit_compiler::PassDebug) {                                             \
    X;                                                                         \
  } else {                                                                     \
    LLVM_DEBUG(X);                                                             \
  }

#endif // SYCL_FUSION_PASSES_DEBUG_H
