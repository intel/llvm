//===- SPIRVLLVMTranslation.h - Translation between LLVM IR and SPIR-V ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "JITContext.h"
#include "LLVMSPIRVOpts.h"
#include "llvm/IR/Module.h"
#include <llvm/Support/Error.h>

namespace jit_compiler {

class SPIRVLLVMTranslator {
public:
  /// Translate the LLVM IR module Mod to SPIR-V, store it in the JITContext and
  /// return a pointer to its container.
  static llvm::Expected<JITBinary *> translateLLVMtoSPIRV(llvm::Module &Mod,
                                                          JITContext &JITCtx);

private:
  /// Default settings for the SPIRV translation options.
  static SPIRV::TranslatorOpts &translatorOpts();
};

} // namespace jit_compiler
