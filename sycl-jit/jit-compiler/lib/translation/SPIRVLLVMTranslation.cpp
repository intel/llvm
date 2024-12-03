//==---------------------- SPIRVLLVMTranslation.cpp ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVLLVMTranslation.h"

#include "Kernel.h"
#include "LLVMSPIRVLib.h"
#include "helper/ErrorHandling.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <sstream>

using namespace jit_compiler;
using namespace jit_compiler::translation;
using namespace llvm;

SPIRV::TranslatorOpts &SPIRVLLVMTranslator::translatorOpts() {
  static auto Opts = []() -> SPIRV::TranslatorOpts {
    // Options for translation between SPIR-V and LLVM IR.
    // Set SPIRV-V 1.4 as the maximum version number for now.
    // Note that some parts of the code depend on the available builtins, e.g.,
    // passes/kernel-fusion/Builtins.cpp, so updating the SPIR-V version should
    // involve revisiting that code.
    SPIRV::TranslatorOpts TransOpt{SPIRV::VersionNumber::SPIRV_1_4};
    // Enable attachment of kernel arg names as metadata.
    TransOpt.enableGenArgNameMD();
    // Enable mem2reg.
    TransOpt.setMemToRegEnabled(true);
    // Enable all extensions.
    // TODO: Specifically enable only the
    // extensions listed in the KernelInfo.
    // FIXME: Because there's no size provided,
    // there's currently no obvious way to iterate the
    // array of extensions in KernelInfo.
    TransOpt.enableAllExtensions();
    // TODO: Remove this workaround.
    TransOpt.setAllowedToUseExtension(
        SPIRV::ExtensionID::SPV_KHR_untyped_pointers, false);
    TransOpt.setDesiredBIsRepresentation(
        SPIRV::BIsRepresentation::SPIRVFriendlyIR);
    // TODO: We need to take care of specialization constants, either by
    // instantiating them by the user-supplied value from the SYCL runtime or by
    // making sure they are correctly represented in the output of the fusion
    // process.
    return TransOpt;
  }();
  return Opts;
}

Expected<std::unique_ptr<llvm::Module>>
SPIRVLLVMTranslator::loadSPIRVKernel(llvm::LLVMContext &LLVMCtx,
                                     SYCLKernelInfo &Kernel) {
  std::unique_ptr<Module> Result{nullptr};

  SYCLKernelBinaryInfo &BinInfo = Kernel.BinaryInfo;
  assert(BinInfo.Format == BinaryFormat::SPIRV &&
         "Only SPIR-V supported as input");

  // Create an input stream for the SPIR-V binary.
  std::stringstream SPIRStream(
      std::string(reinterpret_cast<const char *>(BinInfo.BinaryStart),
                  BinInfo.BinarySize),
      std::ios_base::in | std::ios_base::binary);
  std::string ErrMsg;
  // Create a raw pointer. readSpirv accepts a reference to a pointer,
  // so it will reset the pointer to point to an actual LLVM module.
  Module *LLVMMod;
  auto Success =
      llvm::readSpirv(LLVMCtx, translatorOpts(), SPIRStream, LLVMMod, ErrMsg);
  if (!Success) {
    return createStringError(
        inconvertibleErrorCode(),
        "Failed to load and translate SPIR-V module with error %s",
        ErrMsg.c_str());
  }
  std::unique_ptr<Module> NewMod{LLVMMod};

  return std::move(NewMod);
}

Expected<jit_compiler::KernelBinary *>
SPIRVLLVMTranslator::translateLLVMtoSPIRV(Module &Mod, JITContext &JITCtx) {
  std::ostringstream BinaryStream;
  std::string ErrMsg;
  auto Success = llvm::writeSpirv(&Mod, translatorOpts(), BinaryStream, ErrMsg);
  if (!Success) {
    return createStringError(
        inconvertibleErrorCode(),
        "Translation of LLVM IR to SPIR-V failed with error %s",
        ErrMsg.c_str());
  }
  return &JITCtx.emplaceKernelBinary(BinaryStream.str(), BinaryFormat::SPIRV);
}
