//==-------------------------- KernelFusion.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KernelFusion.h"
#include "Kernel.h"
#include "KernelIO.h"
#include "Options.h"
#include "fusion/FusionHelper.h"
#include "fusion/FusionPipeline.h"
#include "helper/ConfigHelper.h"
#include "helper/ErrorHandling.h"
#include "translation/SPIRVLLVMTranslation.h"
#include <llvm/Support/Error.h>
#include <sstream>

using namespace jit_compiler;

using FusedFunction = helper::FusionHelper::FusedFunction;
using FusedFunctionList = std::vector<FusedFunction>;

static FusionResult errorToFusionResult(llvm::Error &&Err,
                                        const std::string &Msg) {
  std::stringstream ErrMsg;
  ErrMsg << Msg << "\nDetailed information:\n";
  llvm::handleAllErrors(std::move(Err),
                        [&ErrMsg](const llvm::StringError &StrErr) {
                          // Cannot throw an exception here if LLVM itself is
                          // compiled without exception support.
                          ErrMsg << "\t" << StrErr.getMessage() << "\n";
                        });
  return FusionResult{ErrMsg.str()};
}

FusionResult KernelFusion::fuseKernels(
    JITContext &JITCtx, Config &&JITConfig,
    const std::vector<SYCLKernelInfo> &KernelInformation,
    const std::vector<std::string> &KernelsToFuse,
    const std::string &FusedKernelName, ParamIdentList &Identities,
    int BarriersFlags,
    const std::vector<jit_compiler::ParameterInternalization> &Internalization,
    const std::vector<jit_compiler::JITConstant> &Constants) {

  // Initialize the configuration helper to make the options for this invocation
  // available (on a per-thread basis).
  ConfigHelper::setConfig(std::move(JITConfig));

  SYCLModuleInfo ModuleInfo;
  // Copy the kernel information for the input kernels to the module
  // information. We could remove the copy, if we removed the const from the
  // input interface, so it depends on the guarantees we want to give to
  // callers.
  ModuleInfo.kernels().insert(ModuleInfo.kernels().end(),
                              KernelInformation.begin(),
                              KernelInformation.end());
  // Load all input kernels from their respective SPIR-V modules into a single
  // LLVM IR module.
  llvm::Expected<std::unique_ptr<llvm::Module>> ModOrError =
      translation::SPIRVLLVMTranslator::loadSPIRVKernels(
          *JITCtx.getLLVMContext(), ModuleInfo.kernels());
  if (auto Error = ModOrError.takeError()) {
    return errorToFusionResult(std::move(Error), "SPIR-V translation failed");
  }
  std::unique_ptr<llvm::Module> LLVMMod = std::move(*ModOrError);

  // Add information about the kernel that should be fused as metadata into the
  // LLVM module.
  FusedFunction FusedKernel{FusedKernelName, KernelsToFuse,
                            std::move(Identities), Internalization, Constants};
  FusedFunctionList FusedKernelList;
  FusedKernelList.push_back(FusedKernel);
  llvm::Expected<std::unique_ptr<llvm::Module>> NewModOrError =
      helper::FusionHelper::addFusedKernel(LLVMMod.get(), FusedKernelList);
  if (auto Error = NewModOrError.takeError()) {
    return errorToFusionResult(std::move(Error),
                               "Insertion of fused kernel stub failed");
  }
  std::unique_ptr<llvm::Module> NewMod = std::move(*NewModOrError);

  // Invoke the actual fusion via LLVM pass manager.
  std::unique_ptr<SYCLModuleInfo> NewModInfo =
      fusion::FusionPipeline::runFusionPasses(*NewMod, ModuleInfo,
                                              BarriersFlags);

  // Get the updated kernel info for the fused kernel and add the information to
  // the existing KernelInfo.
  if (!NewModInfo->hasKernelFor(FusedKernelName)) {
    return FusionResult{"No KernelInfo for fused kernel"};
  }

  SYCLKernelInfo &FusedKernelInfo = *NewModInfo->getKernelFor(FusedKernelName);

  // Translate the LLVM IR module resulting from the fusion pass into SPIR-V.
  llvm::Expected<jit_compiler::SPIRVBinary *> BinaryOrError =
      translation::SPIRVLLVMTranslator::translateLLVMtoSPIRV(*NewMod, JITCtx);
  if (auto Error = BinaryOrError.takeError()) {
    return errorToFusionResult(std::move(Error),
                               "Translation to SPIR-V failed");
  }
  jit_compiler::SPIRVBinary *SPIRVBin = *BinaryOrError;

  // Update the KernelInfo for the fused kernel with the address and size of the
  // SPIR-V binary resulting from translation.
  SYCLKernelBinaryInfo &FusedBinaryInfo = FusedKernelInfo.BinaryInfo;
  FusedBinaryInfo.Format = BinaryFormat::SPIRV;
  // Output SPIR-V should use the same number of address bits as the input
  // SPIR-V. SPIR-V translation requires all modules to use the same number of
  // address bits, so it's safe to take the value from the first one.
  FusedBinaryInfo.AddressBits =
      ModuleInfo.kernels().front().BinaryInfo.AddressBits;
  FusedBinaryInfo.BinaryStart = SPIRVBin->address();
  FusedBinaryInfo.BinarySize = SPIRVBin->size();

  return FusionResult{FusedKernelInfo};
}
