//===- Translation.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Translation.h"
#include "helper/ConfigHelper.h"

#include "SPIRVLLVMTranslation.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace jit_compiler;
using namespace llvm;

llvm::Expected<JITBinaryInfo> Translator::translate(llvm::Module &Mod,
                                                    JITContext &JITCtx,
                                                    BinaryFormat Format,
                                                    const char *KernelName) {
  llvm::TimeTraceScope TTS{"translate"};
  JITBinary *JITBin = nullptr;
  switch (Format) {
  case BinaryFormat::SPIRV: {
    if (auto Error = translateToSPIRV(Mod, JITCtx).moveInto(JITBin)) {
      return Error;
    }
    break;
  }
  case BinaryFormat::PTX: {
    if (auto Error = translateToPTX(Mod, JITCtx, KernelName).moveInto(JITBin)) {
      return Error;
    }
    break;
  }
  case BinaryFormat::AMDGCN: {
    if (auto Error =
            translateToAMDGCN(Mod, JITCtx, KernelName).moveInto(JITBin))
      return Error;
    break;
  }
  default: {
    return createStringError(
        inconvertibleErrorCode(),
        "Failed to translate module to unsupported output format");
  }
  }

  JITBinaryInfo BinaryInfo;
  BinaryInfo.Format = Format;
  // Output SPIR-V should use the same number of address bits as the input
  // SPIR-V. SPIR-V translation requires all modules to use the same number of
  // address bits, so it's safe to take the value from the first one.
  BinaryInfo.AddressBits = Mod.getDataLayout().getPointerSizeInBits();
  BinaryInfo.BinaryStart = JITBin->address();
  BinaryInfo.BinarySize = JITBin->size();
  return BinaryInfo;
}

llvm::Expected<JITBinary *> Translator::translateToSPIRV(llvm::Module &Mod,
                                                         JITContext &JITCtx) {
  return SPIRVLLVMTranslator::translateLLVMtoSPIRV(Mod, JITCtx);
}

llvm::Expected<JITBinary *> Translator::translateToPTX(llvm::Module &Mod,
                                                       JITContext &JITCtx,
                                                       const char *KernelName) {
#ifndef JIT_SUPPORT_PTX
  (void)Mod;
  (void)JITCtx;
  (void)KernelName;
  return createStringError(inconvertibleErrorCode(),
                           "PTX translation not supported in this build");
#else  // JIT_SUPPORT_PTX
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

  // Give priority to user specified values (through environment variables:
  // SYCL_JIT_AMDGCN_PTX_TARGET_CPU and SYCL_JIT_AMDGCN_PTX_TARGET_FEATURES).
  auto CPUVal = ConfigHelper::get<option::JITTargetCPU>();
  auto FeaturesVal = ConfigHelper::get<option::JITTargetFeatures>();
  llvm::StringRef CPU = CPUVal.begin();
  llvm::StringRef Features = FeaturesVal.begin();

  auto *KernelFunc = KernelName ? Mod.getFunction(KernelName) : nullptr;
  // If they were not set, use default and consult the module for alternatives
  // (if present).
  if (CPU.empty()) {
    CPU = "sm_50";
    if (KernelFunc && KernelFunc->hasFnAttribute(TARGET_CPU_ATTRIBUTE)) {
      CPU = KernelFunc->getFnAttribute(TARGET_CPU_ATTRIBUTE).getValueAsString();
    }
  }
  if (Features.empty()) {
    Features = "+sm_50,+ptx76";
    if (KernelFunc && KernelFunc->hasFnAttribute(TARGET_FEATURE_ATTRIBUTE)) {
      Features = KernelFunc->getFnAttribute(TARGET_FEATURE_ATTRIBUTE)
                     .getValueAsString();
    }
  }

  std::unique_ptr<TargetMachine> TargetMachine(Target->createTargetMachine(
      Mod.getTargetTriple(), CPU, Features, {}, llvm::Reloc::PIC_, std::nullopt,
      llvm::CodeGenOptLevel::Default));

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

  return &JITCtx.emplaceBinary(std::move(PTXASM), BinaryFormat::PTX);
#endif // JIT_SUPPORT_PTX
}

llvm::Expected<JITBinary *>
Translator::translateToAMDGCN(llvm::Module &Mod, JITContext &JITCtx,
                              const char *KernelName) {
#ifndef JIT_SUPPORT_AMDGCN
  (void)Mod;
  (void)JITCtx;
  (void)KernelName;
  return createStringError(inconvertibleErrorCode(),
                           "AMDGPU translation not supported in this build");
#else  // JIT_SUPPORT_AMDGCN

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

  auto CPUVal = ConfigHelper::get<option::JITTargetCPU>();
  auto FeaturesVal = ConfigHelper::get<option::JITTargetFeatures>();
  llvm::StringRef CPU = CPUVal.begin();
  llvm::StringRef Features = FeaturesVal.begin();

  auto *KernelFunc = KernelName ? Mod.getFunction(KernelName) : nullptr;
  if (CPU.empty()) {
    // Set to the lowest tested target according to the GetStartedGuide, section
    // "Build DPC++ toolchain with support for HIP AMD"
    CPU = "gfx906";
    if (KernelFunc && KernelFunc->hasFnAttribute(TARGET_CPU_ATTRIBUTE)) {
      CPU = KernelFunc->getFnAttribute(TARGET_CPU_ATTRIBUTE).getValueAsString();
    }
  }
  if (Features.empty()) {
    if (KernelFunc && KernelFunc->hasFnAttribute(TARGET_FEATURE_ATTRIBUTE)) {
      Features = KernelFunc->getFnAttribute(TARGET_FEATURE_ATTRIBUTE)
                     .getValueAsString();
    }
  }

  std::unique_ptr<TargetMachine> TargetMachine(Target->createTargetMachine(
      Mod.getTargetTriple(), CPU, Features, {}, llvm::Reloc::PIC_, std::nullopt,
      llvm::CodeGenOptLevel::Default));

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

  return &JITCtx.emplaceBinary(std::move(AMDObj), BinaryFormat::AMDGCN);
#endif // JIT_SUPPORT_AMDGCN
}
