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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <sstream>

using namespace jit_compiler;
using namespace jit_compiler::translation;
using namespace llvm;

void SPIRVLLVMTranslator::getAttributeValues(std::vector<std::string> &Values,
                                             MDNode *MD, size_t NumValues) {
  assert(MD->getNumOperands() == NumValues && "Incorrect number of values");
  for (const auto &MDOp : MD->operands()) {
    auto *ConstantMD = cast<ConstantAsMetadata>(MDOp);
    auto *ConstInt = cast<ConstantInt>(ConstantMD->getValue());
    Values.push_back(std::to_string(ConstInt->getZExtValue()));
  }
}

// NOLINTNEXTLINE(readability-identifier-naming)
static const char *REQD_WORK_GROUP_SIZE_ATTR = "reqd_work_group_size";
// NOLINTNEXTLINE(readability-identifier-naming)
static const char *WORK_GROUP_SIZE_HINT_ATTR = "work_group_size_hint";

void SPIRVLLVMTranslator::restoreKernelAttributes(Module *Mod,
                                                  SYCLKernelInfo &Info) {
  auto *KernelFunction = Mod->getFunction(Info.Name);
  assert(KernelFunction && "Kernel function not present in module");
  if (auto *MD = KernelFunction->getMetadata(REQD_WORK_GROUP_SIZE_ATTR)) {
    SYCLKernelAttribute ReqdAttr{REQD_WORK_GROUP_SIZE_ATTR};
    getAttributeValues(ReqdAttr.Values, MD, 3);
    Info.Attributes.push_back(ReqdAttr);
  }
  if (auto *MD = KernelFunction->getMetadata(WORK_GROUP_SIZE_HINT_ATTR)) {
    SYCLKernelAttribute HintAttr{WORK_GROUP_SIZE_HINT_ATTR};
    getAttributeValues(HintAttr.Values, MD, 3);
    Info.Attributes.push_back(HintAttr);
  }
}

SPIRV::TranslatorOpts &SPIRVLLVMTranslator::translatorOpts() {
  static auto Opts = []() -> SPIRV::TranslatorOpts {
    // Options for translation between SPIR-V and LLVM IR.
    // Set SPIRV-V 1.2 as the maximum version number for now.
    SPIRV::TranslatorOpts TransOpt{SPIRV::VersionNumber::SPIRV_1_2};
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

Expected<std::unique_ptr<Module>>
SPIRVLLVMTranslator::readAndTranslateSPIRV(LLVMContext &LLVMCtx,
                                           BinaryBlob Input) {
  // Create an input stream for the binary blob.
  std::stringstream SPIRStream(
      std::string(reinterpret_cast<const char *>(Input.first), Input.second),
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
  return std::unique_ptr<Module>(LLVMMod);
}

Expected<std::unique_ptr<llvm::Module>>
SPIRVLLVMTranslator::loadSPIRVKernels(llvm::LLVMContext &LLVMCtx,
                                      std::vector<SYCLKernelInfo> &Kernels) {
  std::unique_ptr<Module> Result{nullptr};
  bool First = true;
  DenseSet<BinaryBlob> ParsedSPIRVModules;
  size_t AddressBits = 0;
  for (auto &Kernel : Kernels) {
    // FIXME: Currently, we use the front of the list.
    // Do we need to iterate to find the most suitable
    // SPIR-V module?
    SYCLKernelBinaryInfo &BinInfo = Kernel.BinaryInfo;
    // TODO(Lukas, ONNX-399): Also support LLVM IR as input but simply skipping
    // the translation from SPIR-V to LLVM.
    assert(BinInfo.Format == BinaryFormat::SPIRV &&
           "Only SPIR-V supported as input");
    const unsigned char *SPRModulePtr = BinInfo.BinaryStart;
    size_t SPRModuleSize = BinInfo.BinarySize;
    BinaryBlob BinBlob{SPRModulePtr, SPRModuleSize};
    if (ParsedSPIRVModules.contains(BinBlob)) {
      // Multiple kernels can be stored in the same SPIR-V module.
      // If we encountered the same SPIR-V module before, skip.
      // NOTE: We compare the pointer as well as the size, in case
      // a previous kernel only referenced part of the SPIR-V module.
      // Not sure this can actually happen, but better safe than sorry.
      continue;
    }
    // Simply load and translate the SPIR-V into the currently still empty
    // module.
    PROPAGATE_ERROR(NewMod, readAndTranslateSPIRV(LLVMCtx, BinBlob));

    // We do not assume that the input binary information has the address bits
    // set, but rather retrieve this information from the SPIR-V/LLVM module's
    // data-layout.
    BinInfo.AddressBits = NewMod->getDataLayout().getPointerSizeInBits();
    assert((First || BinInfo.AddressBits == AddressBits) &&
           "Address bits do not match");
    // Restore SYCL/OpenCL kernel attributes such as 'reqd_work_group_size' or
    // 'work_group_size_hint' from metadata attached to the kernel function and
    // store it in the SYCLKernelInfo.
    // TODO(Lukas, ONNX-399): Validate that DPC++ used metadata to represent
    // that information.
    restoreKernelAttributes(NewMod.get(), Kernel);

    if (First) {
      // We can simply assign the module we just loaded from SPIR-V to the
      // empty pointer on the first iteration.
      Result = std::move(NewMod);
      // The first module will dictate the address bits for the remaining.
      AddressBits = BinInfo.AddressBits;
      First = false;
    } else {
      // We have already loaded some module, so now we need to
      // link the module we just loaded with the result so far.
      // FIXME: We allow duplicates to be overridden by the module
      // read last. This could cause problems if different modules contain
      // definitions with the same name, but different body/content.
      // Check that this is not problematic.
      Linker::linkModules(*Result, std::move(NewMod),
                          Linker::Flags::OverrideFromSrc);
      if (AddressBits != BinInfo.AddressBits) {
        return createStringError(
            inconvertibleErrorCode(),
            "Number of address bits between SPIR-V modules does not match");
      }
    }
  }
  return std::move(Result);
}

Expected<jit_compiler::SPIRVBinary *>
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
  return &JITCtx.emplaceSPIRVBinary(BinaryStream.str());
}
