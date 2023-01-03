//==--- SPIRVLLVMTranslation.h - Translation between LLVM IR and SPIR-V ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_TRANSLATION_SPIRVLLVMTRANSLATION_H
#define SYCL_FUSION_JIT_COMPILER_TRANSLATION_SPIRVLLVMTRANSLATION_H

#include "JITContext.h"
#include "Kernel.h"
#include "LLVMSPIRVOpts.h"
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
  loadSPIRVKernels(llvm::LLVMContext &LLVMCtx,
                   std::vector<SYCLKernelInfo> &Kernels);

  ///
  /// Translate the LLVM IR module Mod to SPIR-V, store it in the JITContext and
  /// return a pointer to its container.
  static llvm::Expected<SPIRVBinary *> translateLLVMtoSPIRV(llvm::Module &Mod,
                                                            JITContext &JITCtx);

private:
  ///
  /// Pair of address and size to represent a binary blob.
  using BinaryBlob = std::pair<BinaryAddress, size_t>;

  ///
  /// Get an attribute value consisting of NumValues scalar constant integers
  /// from the MDNode.
  static void getAttributeValues(std::vector<std::string> &Values,
                                 llvm::MDNode *MD, size_t NumValues);

  ///
  /// Restore kernel attributes for the kernel in Info from the metadata
  /// attached to its kernel function in the LLVM module Mod.
  /// Currently supported attributes:
  ///   - reqd_work_group_size
  ///   - work_group_size_hint
  static void restoreKernelAttributes(llvm::Module *Mod, SYCLKernelInfo &Info);

  ///
  /// Read the given SPIR-V binary and translate it to a new LLVM module
  /// associated with the given context.
  static llvm::Expected<std::unique_ptr<llvm::Module>>
  readAndTranslateSPIRV(llvm::LLVMContext &LLVMCtx, BinaryBlob Input);

  ///
  /// Default settings for the SPIRV translation options.
  static SPIRV::TranslatorOpts &translatorOpts();
};

} // namespace translation
} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_TRANSLATION_SPIRVLLVMTRANSLATION_H
