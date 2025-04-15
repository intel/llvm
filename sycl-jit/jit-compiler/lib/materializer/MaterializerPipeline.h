//===- MaterializerPipeline.h - LLVM pass pipeline for materializer -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/Module.h"

namespace jit_compiler {

class MaterializerPipeline {
public:
  /// Run the necessary passes in a custom pass pipeline to perform
  /// materialization of kernel specialization constants.
  static bool
  runMaterializerPasses(llvm::Module &Mod,
                        llvm::ArrayRef<unsigned char> SpecConstData);
};

} // namespace jit_compiler
