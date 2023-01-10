//==-------------------------- LoadKernels.cpp  ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoadKernels.h"
#include "SPIRVLLVMTranslation.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace jit_compiler;
using namespace jit_compiler::translation;
using namespace llvm;

llvm::Expected<std::unique_ptr<llvm::Module>>
KernelLoader::loadKernels(llvm::LLVMContext &LLVMCtx,
                          std::vector<SYCLKernelInfo> &Kernels) {
  std::unique_ptr<Module> Result{nullptr};
  bool First = true;
  DenseSet<BinaryBlob> ParsedBinaries;
  size_t AddressBits = 0;
  for (auto &Kernel : Kernels) {
    // FIXME: Currently, we use the front of the list.
    // Do we need to iterate to find the most suitable
    // SPIR-V module?
    SYCLKernelBinaryInfo &BinInfo = Kernel.BinaryInfo;

    const unsigned char *ModulePtr = BinInfo.BinaryStart;
    size_t ModuleSize = BinInfo.BinarySize;
    BinaryBlob BinBlob{ModulePtr, ModuleSize};
    if (ParsedBinaries.contains(BinBlob)) {
      // Multiple kernels can be stored in the same SPIR-V or LLVM IR module.
      // If we encountered the same binary module before, skip.
      // NOTE: We compare the pointer as well as the size, in case
      // a previous kernel only referenced part of the SPIR-V/LLVM IR module.
      // Not sure this can actually happen, but better safe than sorry.
      continue;
    }
    // Simply load and translate the SPIR-V into the currently still empty
    // module.
    std::unique_ptr<llvm::Module> NewMod;

    switch (BinInfo.Format) {
    case BinaryFormat::LLVM: {
      auto ModOrError = loadLLVMKernel(LLVMCtx, Kernel);
      if (auto Err = ModOrError.takeError()) {
        return std::move(Err);
      }
      NewMod = std::move(*ModOrError);
      break;
    }
    case BinaryFormat::SPIRV: {
      auto ModOrError = loadSPIRVKernel(LLVMCtx, Kernel);
      if (auto Err = ModOrError.takeError()) {
        return std::move(Err);
      }
      NewMod = std::move(*ModOrError);
      break;
    }
    default: {
      return createStringError(
          inconvertibleErrorCode(),
          "Failed to load kernel from unsupported input format");
    }
    }

    // We do not assume that the input binary information has the address bits
    // set, but rather retrieve this information from the SPIR-V/LLVM module's
    // data-layout.
    BinInfo.AddressBits = NewMod->getDataLayout().getPointerSizeInBits();

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

llvm::Expected<std::unique_ptr<llvm::Module>>
KernelLoader::loadLLVMKernel(llvm::LLVMContext &LLVMCtx,
                             SYCLKernelInfo &Kernel) {
  auto &BinInfo = Kernel.BinaryInfo;
  llvm::StringRef RawData(reinterpret_cast<const char *>(BinInfo.BinaryStart),
                          BinInfo.BinarySize);
  return llvm::parseBitcodeFile(
      MemoryBuffer::getMemBuffer(RawData)->getMemBufferRef(), LLVMCtx);
}

llvm::Expected<std::unique_ptr<llvm::Module>>
KernelLoader::loadSPIRVKernel(llvm::LLVMContext &LLVMCtx,
                              SYCLKernelInfo &Kernel) {
  return SPIRVLLVMTranslator::loadSPIRVKernel(LLVMCtx, Kernel);
}
