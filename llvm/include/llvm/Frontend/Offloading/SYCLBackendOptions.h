//===-- SYCLBackendOptions.h - SYCL backend option mapping -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OFFLOADING_SYCLBACKENDOPTIONS_H
#define LLVM_FRONTEND_OFFLOADING_SYCLBACKENDOPTIONS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm::offloading {

/// Option namespace shared by the driver and linker wrapper for SYCL SPIR
/// backends. AOT tools have one namespace for compile and link options; JIT
/// uses separate namespaces. Other targets use clang and need no prefix.
inline StringRef getSYCLBackendOptionPrefix(const Triple &TargetTriple,
                                            bool IsLink) {
  if (TargetTriple.isSPIRAOT())
    return TargetTriple.getSubArch() == Triple::SPIRSubArch_gen
               ? "--ocloc-options="
               : "--opencl-aot-options=";
  if (TargetTriple.isSPIROrSPIRV())
    return IsLink ? "--jit-linker-options=" : "--jit-compiler-options=";
  return {};
}

} // namespace llvm::offloading

#endif // LLVM_FRONTEND_OFFLOADING_SYCLBACKENDOPTIONS_H
