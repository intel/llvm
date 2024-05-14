//==----------------------- KernelTranslation.cpp  -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KernelTranslation.h"

#include "SPIRVLLVMTranslation.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace jit_compiler;
using namespace jit_compiler::translation;
using namespace llvm;

///
/// Get an `Indices` object from the MDNode's three constant integer operands.
static Indices getAttributeValues(MDNode *MD) {
  assert(MD->getNumOperands() == Indices::size());
  Indices Res;
  std::transform(MD->op_begin(), MD->op_end(), Res.begin(),
                 [](const auto &MDOp) {
                   auto *ConstantMD = cast<ConstantAsMetadata>(MDOp);
                   auto *ConstInt = cast<ConstantInt>(ConstantMD->getValue());
                   return ConstInt->getZExtValue();
                 });
  return Res;
}

///
/// Restore kernel attributes for the kernel in Info from the metadata
/// attached to its kernel function in the LLVM module Mod.
/// Currently supported attributes:
///   - reqd_work_group_size
///   - work_group_size_hint
static void restoreKernelAttributes(Module *Mod, SYCLKernelInfo &Info) {
  auto *KernelFunction = Mod->getFunction(Info.Name.c_str());
  assert(KernelFunction && "Kernel function not present in module");
  SmallVector<SYCLKernelAttribute, 2> Attrs;
  using AttrKind = SYCLKernelAttribute::AttrKind;
  if (auto *MD = KernelFunction->getMetadata(
          SYCLKernelAttribute::ReqdWorkGroupSizeName)) {
    Attrs.emplace_back(AttrKind::ReqdWorkGroupSize, getAttributeValues(MD));
  }
  if (auto *MD = KernelFunction->getMetadata(
          SYCLKernelAttribute::WorkGroupSizeHintName)) {
    Attrs.emplace_back(AttrKind::WorkGroupSizeHint, getAttributeValues(MD));
  }
  if (Attrs.empty())
    return;
  Info.Attributes = SYCLAttributeList{Attrs.size()};
  llvm::copy(Attrs, Info.Attributes.begin());
}

llvm::Expected<std::unique_ptr<llvm::Module>>
KernelTranslator::loadKernels(llvm::LLVMContext &LLVMCtx,
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
    if (!ParsedBinaries.contains(BinBlob)) {
      // Multiple kernels can be stored in the same SPIR-V or LLVM IR module.
      // We only load if we did not encounter the same binary module before.
      // NOTE: We compare the pointer as well as the size, in case
      // a previous kernel only referenced part of the SPIR-V/LLVM IR module.
      // Not sure this can actually happen, but better safe than sorry.
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
      ParsedBinaries.insert(BinBlob);
    }
    // Restore SYCL/OpenCL kernel attributes such as 'reqd_work_group_size' or
    // 'work_group_size_hint' from metadata attached to the kernel function and
    // store it in the SYCLKernelInfo.
    restoreKernelAttributes(Result.get(), Kernel);
  }
  return std::move(Result);
}

llvm::Expected<std::unique_ptr<llvm::Module>>
KernelTranslator::loadLLVMKernel(llvm::LLVMContext &LLVMCtx,
                                 SYCLKernelInfo &Kernel) {
  auto &BinInfo = Kernel.BinaryInfo;
  llvm::StringRef RawData(reinterpret_cast<const char *>(BinInfo.BinaryStart),
                          BinInfo.BinarySize);
  return llvm::parseBitcodeFile(
      MemoryBuffer::getMemBuffer(RawData, Kernel.Name.c_str(),
                                 /* RequiresNullTermnator*/ false)
          ->getMemBufferRef(),
      LLVMCtx);
}

llvm::Expected<std::unique_ptr<llvm::Module>>
KernelTranslator::loadSPIRVKernel(llvm::LLVMContext &LLVMCtx,
                                  SYCLKernelInfo &Kernel) {
  return SPIRVLLVMTranslator::loadSPIRVKernel(LLVMCtx, Kernel);
}

llvm::Error KernelTranslator::translateKernel(SYCLKernelInfo &Kernel,
                                              llvm::Module &Mod,
                                              JITContext &JITCtx,
                                              BinaryFormat Format) {

  KernelBinary *KernelBin = nullptr;
  switch (Format) {
  case BinaryFormat::SPIRV: {
    llvm::Expected<KernelBinary *> BinaryOrError =
        translateToSPIRV(Mod, JITCtx);
    if (auto Error = BinaryOrError.takeError()) {
      return Error;
    }
    KernelBin = *BinaryOrError;
    break;
  }
  case BinaryFormat::PTX: {
    llvm::Expected<KernelBinary *> BinaryOrError =
        translateToPTX(Kernel, Mod, JITCtx);
    if (auto Error = BinaryOrError.takeError()) {
      return Error;
    }
    KernelBin = *BinaryOrError;
    break;
  }
  case BinaryFormat::AMDGCN: {
    llvm::Expected<KernelBinary *> BinaryOrError =
        translateToAMDGCN(Kernel, Mod, JITCtx);
    if (auto Error = BinaryOrError.takeError())
      return Error;
    KernelBin = *BinaryOrError;
    break;
  }
  default: {
    return createStringError(
        inconvertibleErrorCode(),
        "Failed to translate kernel to unsupported output format");
  }
  }

  // Update the KernelInfo for the fused kernel with the address and size of the
  // SPIR-V binary resulting from translation.
  SYCLKernelBinaryInfo &FusedBinaryInfo = Kernel.BinaryInfo;
  FusedBinaryInfo.Format = Format;
  // Output SPIR-V should use the same number of address bits as the input
  // SPIR-V. SPIR-V translation requires all modules to use the same number of
  // address bits, so it's safe to take the value from the first one.
  FusedBinaryInfo.AddressBits = Mod.getDataLayout().getPointerSizeInBits();
  FusedBinaryInfo.BinaryStart = KernelBin->address();
  FusedBinaryInfo.BinarySize = KernelBin->size();
  return Error::success();
}

llvm::Expected<KernelBinary *>
KernelTranslator::translateToSPIRV(llvm::Module &Mod, JITContext &JITCtx) {
  return SPIRVLLVMTranslator::translateLLVMtoSPIRV(Mod, JITCtx);
}

llvm::Expected<KernelBinary *>
KernelTranslator::translateToPTX(SYCLKernelInfo &KernelInfo, llvm::Module &Mod,
                                 JITContext &JITCtx) {
#ifndef FUSION_JIT_SUPPORT_PTX
  (void)KernelInfo;
  (void)Mod;
  (void)JITCtx;
  return createStringError(inconvertibleErrorCode(),
                           "PTX translation not supported in this build");
#else  // FUSION_JIT_SUPPORT_PTX
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXAsmPrinter();
  LLVMInitializeNVPTXTargetMC();

  static const char *TARGET_CPU_ATTRIBUTE = "target-cpu";
  static const char *TARGET_FEATURE_ATTRIBUTE = "target-features";

  std::string TargetTriple{"nvptx64-nvidia-cuda"};

  std::string ErrorMessage;
  const auto *Target =
      llvm::TargetRegistry::lookupTarget(TargetTriple, ErrorMessage);

  if (!Target) {
    return createStringError(
        inconvertibleErrorCode(),
        "Failed to load and translate PTX LLVM IR module with error %s",
        ErrorMessage.c_str());
  }

  llvm::StringRef TargetCPU{"sm_50"};
  llvm::StringRef TargetFeatures{"+sm_50,+ptx76"};
  if (auto *KernelFunc = Mod.getFunction(KernelInfo.Name.c_str())) {
    if (KernelFunc->hasFnAttribute(TARGET_CPU_ATTRIBUTE)) {
      TargetCPU =
          KernelFunc->getFnAttribute(TARGET_CPU_ATTRIBUTE).getValueAsString();
    }
    if (KernelFunc->hasFnAttribute(TARGET_FEATURE_ATTRIBUTE)) {
      TargetFeatures = KernelFunc->getFnAttribute(TARGET_FEATURE_ATTRIBUTE)
                           .getValueAsString();
    }
  }

  // FIXME: Check whether we can provide more accurate target information here
  auto *TargetMachine = Target->createTargetMachine(
      TargetTriple, TargetCPU, TargetFeatures, {}, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Default);

  llvm::legacy::PassManager PM;

  std::string PTXASM;

  {
    llvm::raw_string_ostream ASMStream{PTXASM};
    llvm::buffer_ostream BufferedASM{ASMStream};

    if (TargetMachine->addPassesToEmitFile(
            PM, BufferedASM, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
      return createStringError(
          inconvertibleErrorCode(),
          "Failed to construct pass pipeline to emit output");
    }

    PM.run(Mod);
    ASMStream.flush();
  }

  return &JITCtx.emplaceKernelBinary(std::move(PTXASM), BinaryFormat::PTX);
#endif // FUSION_JIT_SUPPORT_PTX
}

llvm::Expected<KernelBinary *>
KernelTranslator::translateToAMDGCN(SYCLKernelInfo &KernelInfo,
                                    llvm::Module &Mod, JITContext &JITCtx) {
#ifndef FUSION_JIT_SUPPORT_AMDGCN
  (void)KernelInfo;
  (void)Mod;
  (void)JITCtx;
  return createStringError(inconvertibleErrorCode(),
                           "AMDGPU translation not supported in this build");
#else  // FUSION_JIT_SUPPORT_AMDGCN

  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUAsmPrinter();
  LLVMInitializeAMDGPUTargetMC();

  static const char *TARGET_CPU_ATTRIBUTE = "target-cpu";
  static const char *TARGET_FEATURE_ATTRIBUTE = "target-features";

  std::string TargetTriple{"amdgcn-amd-amdhsa"};

  std::string ErrorMessage;
  const auto *Target =
      llvm::TargetRegistry::lookupTarget(TargetTriple, ErrorMessage);

  if (!Target)
    return createStringError(
        inconvertibleErrorCode(),
        "Failed to load and translate AMDGCN LLVM IR module with error %s",
        ErrorMessage.c_str());

  // Set to the lowest tested target according to the GetStartedGuide, section
  // "Build DPC++ toolchain with support for HIP AMD"
  llvm::StringRef TargetCPU{"gfx906"};
  llvm::StringRef TargetFeatures{""};
  if (auto *KernelFunc = Mod.getFunction(KernelInfo.Name.c_str())) {
    if (KernelFunc->hasFnAttribute(TARGET_CPU_ATTRIBUTE)) {
      TargetCPU =
          KernelFunc->getFnAttribute(TARGET_CPU_ATTRIBUTE).getValueAsString();
    }
    if (KernelFunc->hasFnAttribute(TARGET_FEATURE_ATTRIBUTE)) {
      TargetFeatures = KernelFunc->getFnAttribute(TARGET_FEATURE_ATTRIBUTE)
                           .getValueAsString();
    }
  }

  // FIXME: Check whether we can provide more accurate target information here
  auto *TargetMachine = Target->createTargetMachine(
      TargetTriple, TargetCPU, TargetFeatures, {}, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Default);

  std::string AMDObj;
  {
    llvm::legacy::PassManager PM;
    llvm::raw_string_ostream OBJStream{AMDObj};
    llvm::buffer_ostream BufferedOBJ{OBJStream};

    if (TargetMachine->addPassesToEmitFile(PM, BufferedOBJ, nullptr,
                                           llvm::CodeGenFileType::ObjectFile)) {
      return createStringError(
          inconvertibleErrorCode(),
          "Failed to construct pass pipeline to emit output");
    }

    PM.run(Mod);
    OBJStream.flush();
  }

  return &JITCtx.emplaceKernelBinary(std::move(AMDObj), BinaryFormat::AMDGCN);
#endif // FUSION_JIT_SUPPORT_AMDGCN
}
