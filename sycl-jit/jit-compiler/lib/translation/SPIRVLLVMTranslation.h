//==--- SPIRVLLVMTranslation.h - Translation between LLVM IR and SPIR-V ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Kernel.h"
#include "LLVMSPIRVOpts.h"
#include "helper/JITContext.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <llvm/Support/Error.h>
#include <vector>

namespace jit_compiler {
namespace translation {

class SPIRVLLVMTranslator {
public:
  ///
  /// Load a list of SPIR-V kernels into a single LLVM module.
  static llvm::Expected<std::unique_ptr<llvm::Module>>
  loadSPIRVKernel(llvm::LLVMContext &LLVMCtx, const JITBinaryInfo &BinaryInfo);

  ///
  /// Translate the LLVM IR module Mod to SPIR-V, store it in the JITContext and
  /// return a pointer to its container.
  static llvm::Expected<JITBinary *> translateLLVMtoSPIRV(llvm::Module &Mod,
                                                          JITContext &JITCtx);

private:
  ///
  /// Default settings for the SPIRV translation options.
  static SPIRV::TranslatorOpts &translatorOpts();
};

} // namespace translation
} // namespace jit_compiler
