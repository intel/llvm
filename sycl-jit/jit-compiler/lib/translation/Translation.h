//===- Translation.h - Translate SYCL images between different formats ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "JITBinaryInfo.h"
#include "JITContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace jit_compiler {

class Translator {

public:
  /// Translate `Mod` to the given target format. If the format is PTX or
  /// AMDGCN, the optional `KernelName` will be used to determine additional
  /// target attributes from the kernel function.
  static llvm::Expected<JITBinaryInfo>
  translate(llvm::Module &Mod, JITContext &JITCtx, BinaryFormat Format,
            const char *KernelName = nullptr);

private:
  /// Pair of address and size to represent a binary blob.
  using BinaryBlob = std::pair<BinaryAddress, size_t>;

  static llvm::Expected<JITBinary *> translateToSPIRV(llvm::Module &Mod,
                                                      JITContext &JITCtx);

  static llvm::Expected<JITBinary *>
  translateToPTX(llvm::Module &Mod, JITContext &JITCtx, const char *KernelName);

  static llvm::Expected<JITBinary *> translateToAMDGCN(llvm::Module &Mod,
                                                       JITContext &JITCtx,
                                                       const char *KernelName);
};
} // namespace jit_compiler
