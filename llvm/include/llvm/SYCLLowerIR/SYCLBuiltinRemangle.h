//===------- SYCLBuiltinRemangle.h - SYCLBuiltinRemangle Pass -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Remangle SPIR-V built-ins in user SYCL device code match OpenCL C mangling
// used in libspirv. This enables linking with unmodified libspirv.
//
// Type mappings applied (target -> OpenCL):
// - long long -> long
// - long -> int (Windows or 32-bit only)
// - signed char -> char
// - char -> unsigned char (only if char is signed on the host)
// - _Float16 -> half
// - Pointer address space adjustments if target's default addrspace is private.

#ifndef LLVM_SYCL_BUILTIN_REMANGLE_H
#define LLVM_SYCL_BUILTIN_REMANGLE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLBuiltinRemanglePass : public PassInfoMixin<SYCLBuiltinRemanglePass> {
public:
  SYCLBuiltinRemanglePass(bool CharIsSigned = true)
      : CharIsSigned(CharIsSigned) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

private:
  bool CharIsSigned;
};

} // namespace llvm

#endif // LLVM_SYCL_BUILTIN_REMANGLE_H
